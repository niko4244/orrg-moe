"""Render experiments/results/p1_real.json as the markdown tables in reports/P1-RUNG.md.

    PYTHONPATH=. python experiments/p1_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PATH = Path("experiments/results/p1_real.json")
OUT = Path("experiments/results/p1_real_table.md")
ARMS = [
    "dense_continued",
    "naive_upcycle",
    "orrg_frozen",
    "orrg_all",
    "orrg_clustered",
    "orrg_distill",
]


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def fmt(xs: list[float], digits: int = 4, scale: float = 1.0, signed: bool = False) -> str:
    if not xs:
        return "–"
    values = [x / scale for x in xs]
    m = mean(values)
    sign = "+" if signed else ""
    if len(values) > 1:
        half = (max(values) - min(values)) / 2
        return f"{m:{sign}.{digits}f} ± {half:.{digits}f}"
    return f"{m:{sign}.{digits}f}"


def main() -> None:
    data = json.loads(PATH.read_text())
    rows = data["rows"]
    args = data["args"]
    by_arm = {name: [r for r in rows if r["arm"] == name] for name in ARMS}
    dense = {r["seed"]: r for r in by_arm["dense_continued"]}

    lines: list[str] = []
    header = (
        "| arm | val before | val after | Δval | Δ vs dense | conv KL | trainable | active | "
        "s/step | imbalance | dead | diversity | cohesion |"
    )
    lines += [header, "|" + "---|" * 13]

    for name in ARMS:
        rs = by_arm[name]
        if not rs:
            continue
        conv = [r["conversion"]["kl"] for r in rs if r["conversion"]]
        metrics = [r["metrics"] for r in rs if r["metrics"]]
        clustering = [r["clustering"] for r in rs if r["clustering"]]
        vs_dense = [
            r["val_after"] - dense[r["seed"]]["val_after"] for r in rs if r["seed"] in dense
        ] if name != "dense_continued" else []
        lines.append(
            "| {name} | {before} | {after} | {delta} | {vs} | {kl} | {trainable} | {active} | "
            "{speed} | {imb} | {dead} | {div} | {coh} |".format(
                name=name,
                before=fmt([r["val_before"] for r in rs]),
                after=fmt([r["val_after"] for r in rs]),
                delta=fmt([r["delta"] for r in rs], signed=True),
                vs=fmt(vs_dense, signed=True) if vs_dense else "–",
                kl=fmt(conv, digits=2) if conv else "–",
                trainable=fmt([r["trainable_params"] for r in rs], digits=1, scale=1e6) + "M",
                active=(
                    fmt([r["active_params"] for r in rs if r["active_params"]], digits=1, scale=1e6)
                    + "M"
                    if any(r["active_params"] for r in rs)
                    else "–"
                ),
                speed=fmt([r["seconds_per_step"] for r in rs], digits=2),
                imb=fmt([m["load_imbalance"] for m in metrics], digits=3),
                dead=fmt([m["dead_experts"] for m in metrics], digits=1),
                div=fmt([m["expert_diversity"] for m in metrics], digits=4),
                coh=(
                    fmt([c["cohesion"] for c in clustering], digits=3) + " / "
                    + fmt([c["baseline"] for c in clustering], digits=3)
                    if clustering
                    else "–"
                ),
            )
        )

    detail = ["", "Per-seed rows:", "", "| seed | arm | val before | val after | Δval | conv KL | s/step |",
              "|---|---|---|---|---|---|---|"]
    for row in rows:
        detail.append(
            "| {seed} | {arm} | {b:.4f} | {a:.4f} | {d:+.4f} | {kl} | {s:.2f} |".format(
                seed=row["seed"],
                arm=row["arm"],
                b=row["val_before"],
                a=row["val_after"],
                d=row["delta"],
                kl=f"{row['conversion']['kl']:.2e}" if row["conversion"] else "–",
                s=row["seconds_per_step"],
            )
        )

    config = (
        f"config: steps={args['steps']} seeds={args['seeds']} batch={args['batch']} "
        f"seq={args['seq']} variants={args['variants']} shared={args['shared']} "
        f"groups={args['groups']} optimizer={args['optimizer']} "
        f"distill_coef={args['distill_coef']} distill_tokens={args['distill_tokens']} "
        f"cluster_tokens={args['cluster_tokens']} lr={args['lr']}"
    )
    text = "\n".join([config, "", *lines, *detail]) + "\n"
    OUT.write_text(text, encoding="utf-8")
    # Windows consoles default to cp1252 and choke on the Δ header.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)


if __name__ == "__main__":
    main()
