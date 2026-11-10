"""Compare the two-epoch ORRG arms against the dense two-epoch endpoint.

    PYTHONPATH=. python experiments/p2_compare.py          # print the section
    PYTHONPATH=. python experiments/p2_compare.py --write  # rewrite the report block

The rendered section is inserted between the P2-RESULT markers in
``reports/P2-EPOCHS.md``. Arms that have not landed yet render as "pending".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RES = Path("experiments/results")
REPORT = Path("reports/P2-EPOCHS.md")
START = "<!-- P2-RESULT:START -->"
END = "<!-- P2-RESULT:END -->"
DENSE = "dense_continued"
ARMS = ["orrg_all", "orrg_orth_mag"]
MARGIN = 0.005


def load(name: str) -> dict | None:
    path = RES / f"p2_epochs_{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["rows"][0]


def fmt(value, digits: int = 4) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def row_line(name: str, row: dict | None, dense: dict, label: str | None = None) -> str:
    if row is None:
        return f"| {label or name} | pending | | | | | | | | |"
    metrics = row.get("metrics") or {}
    last = (row.get("trace") or [{}])[-1]
    delta = "–" if name == DENSE else f"{row['val_after'] - dense['val_after']:+.4f}"
    dead = metrics.get("dead_experts")
    return (
        f"| {label or name} | {row['val_before']:.4f} | {row['val_after']:.4f} | {delta} | "
        f"{dead if dead is not None else '–'} | {fmt(metrics.get('load_imbalance'), 3)} | "
        f"{fmt(metrics.get('expert_diversity'), 4)} | {fmt(last.get('residual_share'), 4)} | "
        f"{fmt(last.get('orth'), 4)} | {row['seconds_per_step']:.2f} |"
    )


def verdict(rows: dict[str, dict | None], dense: dict) -> str:
    missing = [name for name in ARMS if rows[name] is None]
    if missing:
        return "_Pending: waiting for " + ", ".join(
            f"`p2_epochs_{name}.json`" for name in missing
        ) + "._"
    deltas = {name: rows[name]["val_after"] - dense["val_after"] for name in ARMS}
    spread = ", ".join(f"`{name}` {delta:+.4f}" for name, delta in deltas.items())
    wins = {name: delta for name, delta in deltas.items() if delta < -MARGIN}
    if wins:
        parts = ", ".join(f"`{name}` by {abs(delta):.4f} nats" for name, delta in wins.items())
        return (
            f"**Verdict.** At 2,660 steps {parts} against dense "
            f"({dense['val_after']:.4f}). The added residual capacity pays at this "
            "budget; with one seed, read the sign as the result and the size as indicative."
        )
    if all(delta > MARGIN for delta in deltas.values()):
        return (
            f"**Verdict.** Neither arm beats dense ({dense['val_after']:.4f}): {spread}. "
            "Residual capacity still does not pay, so the extra parameters buy nothing "
            "at this data budget."
        )
    return (
        f"**Verdict.** No arm clears the {MARGIN:.3f}-nat margin against dense "
        f"({dense['val_after']:.4f}): {spread}. This is a tie at one seed, not "
        "evidence that the capacity pays."
    )


def section(rows: dict[str, dict | None]) -> str:
    dense = rows[DENSE]
    assert dense is not None, "dense endpoint file is missing"
    lines = [
        START,
        "",
        f"Compared against the dense two-epoch endpoint of **{dense['val_after']:.4f}** "
        "at matched steps, schedule and data (2,660 steps x batch 4 x seq 256 = "
        "2.72M tokens).",
        "",
        "| arm | val before | val after | Δ vs dense | dead experts | imbalance | diversity | residual share | orth | s/step |",
        "|---|---|---|---|---|---|---|---|---|---|",
        row_line(DENSE, dense, dense, label="dense"),
    ]
    lines += [row_line(name, rows[name], dense) for name in ARMS]
    lines += ["", verdict(rows, dense), ""]

    steps = sorted(
        {t["step"] for row in rows.values() if row for t in (row.get("trace") or [])}
    )
    if steps:
        lines += [
            "| step | dense | orrg_all | orrg_orth_mag |",
            "|---|---|---|---|",
        ]
        for step in steps:
            cells = []
            for name in [DENSE] + ARMS:
                row = rows.get(name)
                entry = (
                    next((t for t in (row.get("trace") or []) if t["step"] == step), None)
                    if row
                    else None
                )
                cells.append(fmt(entry.get("val")) if entry else "–")
            lines.append(f"| {step} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += [
        "Context: at the 819k-token rung `orrg_all` trailed dense by 0.0085 nats "
        "(2.8289 vs 2.8204), so this comparison asks whether more data changes that sign.",
        "",
        END,
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="rewrite the report block")
    args = parser.parse_args()

    rows = {name: load(name) for name in [DENSE] + ARMS}
    text = section(rows)
    if not args.write:
        print(text)
        return

    report = REPORT.read_text(encoding="utf-8")
    if START not in report or END not in report:
        raise SystemExit(f"markers {START} / {END} missing from {REPORT}")
    head, rest = report.split(START, 1)
    _, tail = rest.split(END, 1)
    REPORT.write_text(head + text + tail, encoding="utf-8")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
