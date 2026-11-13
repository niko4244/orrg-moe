"""Regenerate the report's tables from the run artifacts.

Numbers in the prose drift away from the code; tables generated here cannot. Two
sweeps feed this:

* ``experiments/results/serving_throughput.json`` (both dispatch paths per fused
  arm) -> the seven-model serving table, the k=2 cost table and the batched-vs-
  grouped comparison;
* ``experiments/results/batched_dispatch.json`` -> the row-count crossover table.

Each block is spliced between ``<!-- BEGIN <name> -->`` / ``<!-- END <name> -->``
markers in ``reports/T3-FOUR-PROBES.md``. The script exits non-zero when a batched
arm is materially slower than its grouped baseline, so "no model regresses" is a
check rather than a promise.

    PYTHONPATH=. python experiments/render_report.py [--check-only]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPORT = Path("reports/T3-FOUR-PROBES.md")
RESULTS = Path("experiments/results")
# Metrics the sweep records, in report order.
METRICS = ("prefill_b1", "prefill_b8", "prefill_b64",
           "decode_b1", "decode_b8", "decode_b64")


def fmt(value: float) -> str:
    return f"{value/1e3:.1f}k" if value >= 1000 else f"{value:.0f}"


def with_range(entry: dict) -> str:
    return f"{fmt(entry['median'])} [{fmt(entry['min'])}–{fmt(entry['max'])}]"


def serving_tables(data: dict) -> tuple[str, list[str]]:
    stats, ratios = data["stats"], data["ratios"]

    def med(key: str, metric: str) -> float | None:
        """Per-repeat ratio, whether or not the sweep precomputed that pair."""

        if key in ratios:
            return ratios[key][metric]["median"]
        a, b = key.split("_over_")
        xs, ys = data["samples"].get(a), data["samples"].get(b)
        if not xs or not ys:
            return None
        return statistics.median(x[metric] / y[metric] for x, y in zip(xs, ys))
    names = [n for n in stats if not n.endswith("_grouped")]
    fused = [n for n in names if n.startswith("fused_")]
    lines = [
        "Both dispatch paths on the same weights: `*_grouped` is the fused model with",
        "`BATCHED_BELOW_ROWS` forced to 0, so the batched-vs-grouped rows below either",
        "prove each shape wins or expose it; `BATCHED_BELOW_ROWS = "
        f"{data.get('batched_below_rows', 0):,}` in the shipped build.",
        "",
        "| metric | " + " | ".join(names) + " |",
        "|---" * (len(names) + 1) + "|",
    ]
    for metric in METRICS:
        cells = []
        for name in names:
            entry = stats[name].get(metric)
            cells.append(with_range(entry) if entry else "–")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "| ratio | k=1 | k=2 |",
        "|---|---|---|",
    ]
    for metric in METRICS:
        cells = []
        for k in (1, 2):
            key = f"fused_k{k}_over_fused_k{k}_grouped"
            value = med(key, metric)
            cells.append(f"{value:.2f}" if value else "–")
        lines.append(f"| batched vs grouped, {metric} | " + " | ".join(cells) + " |")

    columns = [
        ("k2/k1, batched", "fused_k2_over_fused_k1"),
        ("k2/k1, grouped", "fused_k2_grouped_over_fused_k1_grouped"),
        ("k2/k1, python", "python_k2_over_python_k1"),
        ("k2/k1, dense-active", "dense_active_k2_over_dense_active_k1"),
    ]
    lines += ["", "| ratio | " + " | ".join(label for label, _ in columns) + " |"]
    lines += ["|---" * (len(columns) + 1) + "|"]
    for metric in METRICS:
        cells = []
        for _, key in columns:
            value = med(key, metric)
            cells.append(f"{value:.2f}" if value else "–")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")

    # Every ratio the prose quotes, so no figure is typed by hand.
    for label, key in (
        ("fused ÷ python", "fused_k{k}_over_python_k{k}"),
        ("fused ÷ dense-active", "fused_k{k}_over_dense_active_k{k}"),
        ("fused ÷ dense-total", "fused_k{k}_over_dense_total"),
    ):
        lines += ["", f"| {label} | k=1 | k=2 |", "|---|---|---|"]
        for metric in METRICS:
            cells = []
            for k in (1, 2):
                value = ratios.get(key.format(k=k), {}).get(metric, {}).get("median")
                cells.append(f"{value:.2f}×" if value else "–")
            lines.append(f"| {metric} | " + " | ".join(cells) + " |")

    # Where the shipped threshold sends both arms down the grouped path the two
    # measurements are the same code, so their spread is this cell's noise floor.
    # Calibrating against it stops identical configurations being called regressions.
    controls = []
    for name in fused:
        k = int(name[-1])
        grouped = f"{name}_grouped"
        if grouped not in stats:
            continue
        for metric in METRICS:
            if not took_batched(metric, k, data):
                controls.append(abs(stats[name][metric]["median"]
                                    / stats[grouped][metric]["median"] - 1.0))
    noise = max(controls, default=0.0)
    allowance = max(0.05, noise + 0.02)
    print(f"noise floor from {len(controls)} same-path cells: {noise:.2f}; "
          f"regression allowance {allowance:.2f}")

    regressions = []
    for name in fused:
        k = int(name[-1])
        grouped = f"{name}_grouped"
        if grouped not in stats:
            continue
        for metric in METRICS:
            if not took_batched(metric, k, data):
                continue
            ours = stats[name][metric]["median"]
            base = stats[grouped][metric]["median"]
            if base and ours / base < 1.0 - allowance:
                regressions.append(
                    f"{name} {metric}: {fmt(ours)} vs grouped {fmt(base)} "
                    f"({ours / base:.2f}x, allowance {allowance:.2f})"
                )
    return "\n".join(lines), regressions


def packed_rows(metric: str, k: int, geometry: dict, prefill_seq: int) -> int:
    """Packed rows a metric produces: one row per shard plus k per group, per token."""

    batch = int(metric.rsplit("b", 1)[1])
    tokens = batch * prefill_seq if metric.startswith("prefill") else batch
    shards = geometry["shared"] + geometry["groups"]
    return tokens * (shards + k * geometry["groups"])


def took_batched(metric: str, k: int, data: dict) -> bool:
    rows = packed_rows(metric, k, data["geometry"], data["prefill_seq"])
    return data["min_batched_rows"] <= rows <= data["batched_below_rows"]


def crossover_table(data: dict, threshold: int) -> tuple[str, list[str]]:
    rows = data["rows"]
    lines = [
        f"Layer-level, `BATCHED_BELOW_ROWS = {threshold:,}`: the last column is the path",
        "the shipped dispatch takes at that size.",
        "",
        "| tokens | packed rows | grouped | batched | batched ÷ grouped | shipped path |",
        "|---|---|---|---|---|---|",
    ]
    regressions = []
    for tokens, row in sorted(rows.items(), key=lambda kv: int(kv[0])):
        ratio = row["batched_ms"] / row["grouped_ms"]
        packed = row.get("rows", 0)
        uses = "bmm" if packed <= threshold else "grouped"
        lines.append(
            f"| {tokens} | {packed} | {row['grouped_ms']:.2f} ms | "
            f"{row['batched_ms']:.2f} ms | {ratio:.2f} | {uses} |"
        )
        # Only sizes the shipped threshold actually routes to bmm can regress; above
        # it the slower batched column is why the threshold is where it is.
        if packed <= threshold and ratio > 1.02:
            regressions.append(f"{tokens} tokens ({packed} rows): batched {ratio:.2f}x slower")
    return "\n".join(lines), regressions


def splice(text: str, name: str, body: str) -> str:
    begin, end = f"<!-- BEGIN {name} -->", f"<!-- END {name} -->"
    start = text.index(begin) + len(begin)
    stop = text.index(end)
    return text[:start] + "\n" + body + "\n" + text[stop:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    serving = json.loads((RESULTS / "serving_throughput.json").read_text())
    crossover = json.loads((RESULTS / "batched_dispatch.json").read_text())

    serving_body, serving_regressions = serving_tables(serving)
    crossover_body, crossover_regressions = crossover_table(
        crossover, serving.get("batched_below_rows", 0)
    )

    if not args.check_only:
        text = REPORT.read_text(encoding="utf-8")
        text = splice(text, "serving-table", serving_body)
        text = splice(text, "crossover-table", crossover_body)
        REPORT.write_text(text, encoding="utf-8")
        print(f"updated {REPORT}")

    print(f"\ncrossover:\n{crossover_body}\n")
    for problem in serving_regressions + crossover_regressions:
        print(f"REGRESSION: {problem}")
    return 1 if serving_regressions + crossover_regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
