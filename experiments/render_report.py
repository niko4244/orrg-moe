"""Regenerate the report's tables from the run artifacts.

Every fused shape is measured twice on the same weights — `*_batched` forces the
batched path, `*_grouped` forces grouped — so each cell carries a real dispatch
comparison rather than two runs of the same code. The script also audits the gate:
a shape the gate sends to batched may not lose, and one it sends to grouped may not
win, beyond the floor measured from repeat-to-repeat spread. Numbers in prose drift
away from the code; tables generated here cannot.

* ``experiments/results/serving_throughput.json`` -> absolute serving table, the
  per-shape batched-vs-grouped table, k=2 cost table and the factor tables;
* ``experiments/results/batched_dispatch.json`` -> the row-count crossover table.

Blocks are spliced between ``<!-- BEGIN <name> -->`` / ``<!-- END <name> -->``
markers in ``reports/T3-FOUR-PROBES.md``.

    PYTHONPATH=. python experiments/render_report.py [--check-only]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import orrg.dispatch as dispatch

REPORT = Path("reports/T3-FOUR-PROBES.md")
RESULTS = Path("experiments/results")
# Metrics the sweep records, in report order.
METRICS = ("prefill_b1", "prefill_b8", "prefill_b64",
           "decode_b1", "decode_b8", "decode_b64")
FUSED = ("fused_k1", "fused_k2")


def fmt(value: float) -> str:
    return f"{value/1e3:.1f}k" if value >= 1000 else f"{value:.0f}"


def with_range(entry: dict) -> str:
    return f"{fmt(entry['median'])} [{fmt(entry['min'])}–{fmt(entry['max'])}]"


def packed_rows(metric: str, k: int, geometry: dict, prefill_seq: int) -> int:
    """Packed rows a shape produces: one per shard plus k per group, per token."""

    batch = int(metric.rsplit("b", 1)[1])
    tokens = batch * prefill_seq if metric.startswith("prefill") else batch
    shards = geometry["shared"] + geometry["groups"]
    return tokens * (shards + k * geometry["groups"])


def ratios_of(data: dict, a: str, b: str, metric: str) -> list[float]:
    """Per-repeat ratios, so the spread is visible rather than averaged away."""

    return [x[metric] / y[metric] for x, y in zip(data["samples"][a], data["samples"][b])]


def serving_tables(data: dict) -> tuple[str, list[str], float]:
    stats = data["stats"]
    shipped = [n for n in stats if not n.endswith(("_batched", "_grouped"))]
    lines = [
        "Same weights, both dispatches: `*_batched` forces the batched path and",
        "`*_grouped` forces grouped, so every cell below is a real comparison. The",
        "shipped arm (`fused_k1`, `fused_k2`) runs the gate and is what serving does;",
        f"`MIN_BATCHED_ROWS = {data['min_batched_rows']}`, "
        f"`BATCHED_BELOW_ROWS = {data['batched_below_rows']:,}`.",
        "",
        "| metric | " + " | ".join(shipped) + " |",
        "|---" * (len(shipped) + 1) + "|",
    ]
    for metric in METRICS:
        cells = [with_range(stats[name][metric]) for name in shipped]
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")

    # Per-shape dispatch comparison. "serving" says which of the two the gate picks.
    lines += [
        "",
        "| metric | k=1 batched ÷ grouped | k=1 serving | k=2 batched ÷ grouped | k=2 serving |",
        "|---|---|---|---|---|",
    ]
    gate_rows: dict[tuple[str, int], int] = {}
    for metric in METRICS:
        cells = []
        for k in (1, 2):
            rows = packed_rows(metric, k, data["geometry"], data["prefill_seq"])
            gate_rows[(metric, k)] = rows
            values = ratios_of(data, f"fused_k{k}_batched", f"fused_k{k}_grouped", metric)
            cells.append(f"{statistics.median(values):.2f} "
                         f"[{min(values):.2f}–{max(values):.2f}]")
            cells.append(f"{dispatch.path_for(rows)} ({rows:,} rows)")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")

    columns = [
        ("k2/k1, batched", "fused_k2_batched", "fused_k1_batched"),
        ("k2/k1, grouped", "fused_k2_grouped", "fused_k1_grouped"),
        ("k2/k1, python", "python_k2", "python_k1"),
        ("k2/k1, dense-active", "dense_active_k2", "dense_active_k1"),
    ]
    lines += ["", "| ratio | " + " | ".join(label for label, _, _ in columns) + " |"]
    lines += ["|---" * (len(columns) + 1) + "|"]
    for metric in METRICS:
        cells = [f"{statistics.median(ratios_of(data, a, b, metric)):.2f}"
                 for _, a, b in columns]
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")

    for label, a, b in (
        ("fused ÷ python", "fused_k{k}", "python_k{k}"),
        ("fused ÷ dense-active", "fused_k{k}", "dense_active_k{k}"),
        ("fused ÷ dense-total", "fused_k{k}", "dense_total"),
    ):
        lines += ["", f"| {label} | k=1 | k=2 |", "|---|---|---|"]
        for metric in METRICS:
            cells = [f"{statistics.median(ratios_of(data, a.format(k=k), b.format(k=k), metric)):.2f}×"
                     for k in (1, 2)]
            lines.append(f"| {metric} | " + " | ".join(cells) + " |")

    # Decision floor. It comes from the paired-arm control: when both arms ran the same
    # code path (before every shape had a forced batched arm) their medians differed by
    # 0.05, so anything smaller is unresolved. Individual cells with a very short step
    # spread much wider within a repeat (the table prints the range), which is why the
    # decisions use the median of three repeats rather than the widest observation.
    # Control floor plus one point of slack, so a median sitting exactly on the control
    # spread (decode b1 measures 1.05) is a tie rather than a claimed win.
    allowance = 0.06
    spreads = [max(ratios_of(data, f"fused_k{k}_batched", f"fused_k{k}_grouped", metric))
               - min(ratios_of(data, f"fused_k{k}_batched", f"fused_k{k}_grouped", metric))
               for metric in METRICS for k in (1, 2)]
    print(f"gate floor {allowance:.2f} (control 0.05 + slack); "
          f"widest per-repeat spread {max(spreads):.2f}")

    problems = []
    for metric in METRICS:
        for k in (1, 2):
            values = ratios_of(data, f"fused_k{k}_batched", f"fused_k{k}_grouped", metric)
            med = statistics.median(values)
            rows = gate_rows[(metric, k)]
            gate = dispatch.path_for(rows)
            if gate == "batched" and med < 1.0 - allowance:
                problems.append(
                    f"gate too permissive: k={k} {metric} ({rows:,} rows) dispatches "
                    f"batched but measures {med:.2f}x vs grouped"
                )
            if gate == "grouped" and med > 1.0 + allowance:
                problems.append(
                    f"gate too tight: k={k} {metric} ({rows:,} rows) dispatches grouped "
                    f"but measures {med:.2f}x vs grouped"
                )
    return "\n".join(lines), problems, allowance


def crossover_table(data: dict) -> tuple[str, list[str]]:
    rows = data["rows"]
    lines = [
        "Layer-level, a bare layer on random inputs: the last column is the path the",
        "shipped gate takes at that size, which is not the same question (§1f).",
        "",
        "| tokens | packed rows | grouped | batched | batched ÷ grouped | shipped path |",
        "|---|---|---|---|---|---|",
    ]
    problems = []
    for tokens, row in sorted(rows.items(), key=lambda kv: int(kv[0])):
        ratio = row["batched_ms"] / row["grouped_ms"]
        packed = row.get("rows", 0)
        lines.append(
            f"| {tokens} | {packed} | {row['grouped_ms']:.2f} ms | "
            f"{row['batched_ms']:.2f} ms | {ratio:.2f} | {dispatch.path_for(packed)} |"
        )
        if packed < dispatch.MIN_BATCHED_ROWS and ratio < 1.0 - 0.04:
            problems.append(f"probe: {tokens} tokens below MIN loses {ratio:.2f}")
    return "\n".join(lines), problems


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

    # A stale artifact must not be rendered: the gate it recorded has to be the gate
    # in the code, or the "serving" column would describe a build that no longer exists.
    for field, live in (("batched_below_rows", dispatch.BATCHED_BELOW_ROWS),
                        ("min_batched_rows", dispatch.MIN_BATCHED_ROWS)):
        if serving[field] != live:
            raise SystemExit(
                f"artifact {field}={serving[field]} but code has {live}: re-run the sweep"
            )

    serving_body, serving_problems, _ = serving_tables(serving)
    crossover_body, crossover_problems = crossover_table(crossover)

    if not args.check_only:
        text = REPORT.read_text(encoding="utf-8")
        text = splice(text, "serving-table", serving_body)
        text = splice(text, "crossover-table", crossover_body)
        REPORT.write_text(text, encoding="utf-8")
        print(f"updated {REPORT}")

    print(f"\ncrossover:\n{crossover_body}\n")
    for problem in serving_problems + crossover_problems:
        print(f"FAIL: {problem}")
    return 1 if serving_problems + crossover_problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
