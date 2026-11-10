"""Compare the two-epoch ORRG arms against the dense two-epoch endpoint.

    PYTHONPATH=. python experiments/p2_compare.py          # print the section
    PYTHONPATH=. python experiments/p2_compare.py --write  # rewrite the report block

The rendered section is inserted between the P2-RESULT markers in
``reports/P2-EPOCHS.md``. Arms that have not landed yet render as "pending".
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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
    rows = json.loads(path.read_text(encoding="utf-8")).get("rows") or []
    if not rows:
        raise SystemExit(f"{path} contains no rows")
    return rows[0]


def fmt(value, digits: int = 4) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _metric(row: dict, metrics_key: str, trace_key: str):
    """Prefer the run summary, fall back to the last traced value."""
    metrics = row.get("metrics") or {}
    if metrics.get(metrics_key) is not None:
        return metrics[metrics_key]
    trace = row.get("trace") or []
    if trace and trace[-1].get(trace_key) is not None:
        return trace[-1][trace_key]
    return None


def _minima(row: dict | None):
    """Lowest finite traced val and the step where it occurred."""
    points = [t for t in ((row or {}).get("trace") or []) if _finite(t.get("val"))]
    if not points:
        return None
    point = min(points, key=lambda t: t["val"])
    return point["val"], point["step"]


def row_line(name: str, row: dict | None, dense: dict, label: str | None = None) -> str:
    if row is None:
        return f"| {label or name} | pending | | | | | | | | |"
    last = (row.get("trace") or [{}])[-1]
    delta = "–" if name == DENSE else f"{row['val_after'] - dense['val_after']:+.4f}"
    dead = _metric(row, "dead_experts", "dead")
    return (
        f"| {label or name} | {row['val_before']:.4f} | {row['val_after']:.4f} | {delta} | "
        f"{dead if dead is not None else '–'} | {fmt(_metric(row, 'load_imbalance', 'imbalance'), 3)} | "
        f"{fmt(_metric(row, 'expert_diversity', 'diversity'), 4)} | {fmt(last.get('residual_share'), 4)} | "
        f"{fmt(last.get('orth'), 4)} | {row['seconds_per_step']:.2f} |"
    )


def verdict(rows: dict[str, dict | None], dense: dict) -> str:
    missing = [name for name in ARMS if rows[name] is None]
    if missing:
        return "_Pending: waiting for " + ", ".join(
            f"`p2_epochs_{name}.json`" for name in missing
        ) + "._"
    diverged = [name for name in ARMS if not _finite(rows[name].get("val_after"))]
    if diverged:
        return (
            "**Verdict.** " + ", ".join(f"`{name}`" for name in diverged)
            + " did not produce a finite validation loss at 2,660 steps, so that arm "
            "diverged and cannot be compared against dense."
        )
    deltas = {name: rows[name]["val_after"] - dense["val_after"] for name in ARMS}
    spread = ", ".join(f"`{name}` {delta:+.4f}" for name, delta in deltas.items())
    wins = {name: delta for name, delta in deltas.items() if delta < -MARGIN}
    if wins:
        lead = min(wins, key=wins.get)
        text = (
            f"**Verdict.** `{lead}` ends {abs(deltas[lead]):.4f} nats below dense "
            f"({dense['val_after']:.4f}); {spread}."
        )
        if len(wins) == 1:
            text += (
                f" Only one arm clears the {MARGIN:.3f}-nat margin, so the gain is not "
                "a general property of the conversion."
            )
        else:
            text += (
                " Both arms clear the margin, so the gain is not specific to one "
                "residual variant."
            )
    elif all(delta > MARGIN for delta in deltas.values()):
        text = (
            f"**Verdict.** Neither arm beats dense ({dense['val_after']:.4f}): {spread}. "
            "Residual capacity still does not pay, so the extra parameters buy nothing "
            "at this data budget."
        )
    else:
        text = (
            f"**Verdict.** No arm clears the {MARGIN:.3f}-nat margin against dense "
            f"({dense['val_after']:.4f}): {spread}. This is a tie at one seed, not "
            "evidence that the capacity pays."
        )

    # The endpoint is far above the traced minimum for both curves, so the table
    # compares late-training plateaus rather than attainable losses.
    best = _minima(dense)
    if best and _finite(dense.get("val_after")) and dense["val_after"] - best[0] > 0.01:
        arm_best = []
        for name in ARMS:
            entry = _minima(rows[name])
            if entry:
                arm_best.append(f"`{name}` {entry[0]:.4f} (step {entry[1]})")
        text += (
            f" Read the endpoint as a late-training regime, not as either model's best: "
            f"dense bottoms out at {best[0]:.4f} (step {best[1]}) and gives back "
            f"{dense['val_after'] - best[0]:.4f} nats by the endpoint"
            + (f"; the arms' minima are {', '.join(arm_best)}." if arm_best else ".")
        )
    return text + (
        f" With one seed and a {MARGIN:.3f}-nat margin, the sign is the result and the "
        "size is indicative."
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
        "| arm | val before | val after | Δ vs dense (neg = better) | dead experts | imbalance | diversity | residual share | orth | s/step |",
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
        "(2.8289 vs 2.8204). That rung ran a shorter total schedule, so the movement "
        "between rungs is indicative rather than a clean data-scaling curve.",
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
        # The table header uses Δ; a cp1252 console would otherwise raise.
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass
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
