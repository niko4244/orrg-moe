"""Render the long-run arms as markdown trajectory tables for the P1 reports.

Reads whichever of the seven long-run arm files exist. Missing arms are skipped so
the script can be run while a run is still landing.

    PYTHONPATH=. python experiments/p1_long_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RES = Path("experiments/results")
ARMS = [
    "dense_continued",
    "orrg_all",
    "orrg_distill",
    "orrg_orth",
    "orrg_sharp",
    "orrg_mag",
    "orrg_orth_mag",
]
METRICS = ["val", "imbalance", "diversity", "residual_share", "orth", "distill"]


def load_rows() -> tuple[list[dict], list[str], dict]:
    rows: list[dict] = []
    present: list[str] = []
    args: dict = {}
    for arm in ARMS:
        path = RES / f"p1_long_{arm}.json"
        if not path.exists():
            print(f"skip missing {path}", file=sys.stderr)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not args:
            args = payload["args"]
        present.append(arm)
        rows.extend(payload["rows"])
    if not rows:
        raise SystemExit("no long-run arm files found")
    return rows, present, args


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "–"
    return f"{value:.{digits}f}"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows, arms, args = load_rows()
    (RES / "p1_long_all.json").write_text(
        json.dumps({"args": args, "rows": rows}, indent=2), encoding="utf-8"
    )

    lines = [
        f"config: steps={args['steps']} seeds={args['seeds']} batch={args['batch']} "
        f"seq={args['seq']} variants={args['variants']} optimizer={args['optimizer']} "
        f"distill_coef={args['distill_coef']} trace_every={args['trace_every']} "
        f"lr={args['lr']}",
        "",
        "## End of run",
        "",
        "| arm | val before | val after | Δval | imbalance | diversity | residual share | orth | distill | s/step |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in arms:
        row = next(r for r in rows if r["arm"] == arm)
        metrics = row.get("metrics") or {}
        last = (row.get("trace") or [{}])[-1]
        lines.append(
            f"| {arm} | {row['val_before']:.4f} | {row['val_after']:.4f} | "
            f"{row['delta']:+.4f} | {fmt(metrics.get('load_imbalance'), 3)} | "
            f"{fmt(metrics.get('expert_diversity'), 4)} | {fmt(last.get('residual_share'), 4)} | "
            f"{fmt(last.get('orth'), 4)} | {fmt(last.get('distill'), 3)} | "
            f"{row['seconds_per_step']:.2f} |"
        )

    steps = sorted({t["step"] for r in rows for t in (r.get("trace") or [])})
    for metric in METRICS:
        lines += [
            "",
            f"## {metric} trajectory",
            "",
            "| step | " + " | ".join(arms) + " |",
            "|---|" + "---|" * len(arms),
        ]
        for step in steps:
            cells = []
            for arm in arms:
                row = next(r for r in rows if r["arm"] == arm)
                entry = next((t for t in (row.get("trace") or []) if t["step"] == step), {})
                cells.append(fmt(entry.get(metric), 4))
            lines.append(f"| {step} | " + " | ".join(cells) + " |")

    text = "\n".join(lines) + "\n"
    (RES / "p1_long_all_table.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
