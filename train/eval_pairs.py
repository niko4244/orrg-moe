"""Paired per-case analysis of two final-suite receipts.

The suite has 50 cases nested in 3 tasks, so a bare total is not evidence: the report
needs the paired 2x2 (both pass / base-only / adapter-only / both fail), the exact
McNemar p on the discordant cells, and the per-task delta. With 3 tasks the smallest
attainable two-sided p at task level is 0.25, which this prints next to the case-level
number so neither can be read as more than it is.

    PYTHONPATH=. python train/eval_pairs.py \
        --base reports/final_suite/qwen3_base_oneshot \
        --adapter reports/final_suite/qwen3_repair_oneshot \
        --out reports/spark_eval_pairs/base_vs_repair.md
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial p for the discordant cells (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def per_case(run: Path) -> dict[tuple[str, str], bool]:
    """(repo, case) -> passed, from the harness's own per-case receipts."""
    out: dict[tuple[str, str], bool] = {}
    for case_file in sorted((run / "cases").glob("*/case.json")):
        record = json.loads(case_file.read_text(encoding="utf-8"))
        repo = record["repo"]
        tests = record.get("tests") or {}
        if not tests:  # older receipts keep only the failing names
            failed = set(record.get("failed") or [])
            tests = {name: ("FAILED" if name in failed else "PASSED")
                     for name in record.get("cases", {}) or failed}
        for name, status in tests.items():
            out[(repo, name)] = status == "PASSED"
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base_dir, adapter_dir = Path(args.base), Path(args.adapter)
    base_receipt = json.loads((base_dir / "receipt.json").read_text(encoding="utf-8"))
    adapter_receipt = json.loads((adapter_dir / "receipt.json").read_text(encoding="utf-8"))

    base, adapter = per_case(base_dir), per_case(adapter_dir)
    keys = sorted(set(base) | set(adapter))
    both = [k for k in keys if base.get(k) and adapter.get(k)]
    base_only = [k for k in keys if base.get(k) and not adapter.get(k)]
    adapter_only = [k for k in keys if not base.get(k) and adapter.get(k)]
    neither = [k for k in keys if not base.get(k) and not adapter.get(k)]

    base_total = sum(1 for k in keys if base.get(k))
    adapter_total = sum(1 for k in keys if adapter.get(k))
    p_case = mcnemar_exact(len(base_only), len(adapter_only))

    repos = sorted({repo for repo, _ in keys})
    tasks = {}
    for repo in repos:
        names = [k for k in keys if k[0] == repo]
        b = sum(1 for k in names if base.get(k))
        a = sum(1 for k in names if adapter.get(k))
        tasks[repo] = {
            "cases": len(names), "base": b, "adapter": a, "delta": a - b,
            "flips_up": [n for r, n in names if not base.get((r, n)) and adapter.get((r, n))],
            "flips_down": [n for r, n in names if base.get((r, n)) and not adapter.get((r, n))],
        }
    up_tasks = sum(1 for t in tasks.values() if t["delta"] > 0)
    down_tasks = sum(1 for t in tasks.values() if t["delta"] < 0)
    p_task = mcnemar_exact(down_tasks, up_tasks)

    # The prompts must be byte-identical between arms or the comparison is meaningless.
    hash_fields = ("contract_sha256", "system_sha256", "prompt_set_sha256")
    hashes_match = all(
        base_receipt.get(field) == adapter_receipt.get(field) for field in hash_fields
    )
    prompt_shas = (base_receipt.get("initial_prompt_sha256") ==
                   adapter_receipt.get("initial_prompt_sha256"))
    identical_prompts = hashes_match and prompt_shas
    identical_config = all(
        base_receipt.get(field) == adapter_receipt.get(field)
        for field in ("temperature", "max_tokens", "max_attempts_per_task", "thinking")
    )

    lines = [
        "# Paired per-case result: base vs repair-format adapter",
        "",
        f"- base run: `{base_dir.as_posix()}` ({base_receipt.get('model')})",
        f"- adapter run: `{adapter_dir.as_posix()}` ({adapter_receipt.get('model')})",
        f"- identical prompts (sha256 of contract/system/all 50 prompts): **{identical_prompts}**",
        f"- identical decoding config (temp/max_tokens/attempts/thinking): **{identical_config}**",
        "",
        "## 2x2",
        "",
        "| | adapter passes | adapter fails |",
        "|---|---:|---:|",
        f"| **base passes** | {len(both)} | {len(base_only)} |",
        f"| **base fails** | {len(adapter_only)} | {len(neither)} |",
        "",
        f"Totals: base {base_total}/{len(keys)}, adapter {adapter_total}/{len(keys)} "
        f"(delta {adapter_total - base_total:+d}).",
        f"Flips: **{len(adapter_only)} fail→pass**, **{len(base_only)} pass→fail**.",
        f"McNemar exact (per case, discordant {len(adapter_only)}+{len(base_only)}): "
        f"p = {p_case:.3f}.",
        "",
        "## Per task",
        "",
        "| task | cases | base | adapter | delta | fail→pass | pass→fail |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for repo, row in tasks.items():
        lines.append(
            f"| {repo} | {row['cases']} | {row['base']} | {row['adapter']} | "
            f"{row['delta']:+d} | {len(row['flips_up'])} | {len(row['flips_down'])} |"
        )
    lines += [
        "",
        f"Task-level McNemar (3 tasks: {up_tasks} up, {down_tasks} down): p = {p_task:.2f}. "
        "With n=3 tasks the smallest attainable two-sided p is 0.25, so no task-level "
        "significance is reachable at any score on this suite.",
        "",
        "## Enumerated flips",
        "",
    ]
    for repo, row in tasks.items():
        if row["flips_up"]:
            lines.append(f"- {repo} fail→pass: " + ", ".join(f"`{n}`" for n in row["flips_up"]))
        if row["flips_down"]:
            lines.append(f"- {repo} pass→fail: " + ", ".join(f"`{n}`" for n in row["flips_down"]))

    payload = {
        "base": str(base_dir), "adapter": str(adapter_dir),
        "base_total": base_total, "adapter_total": adapter_total, "cases": len(keys),
        "both_pass": len(both), "base_only": sorted(f"{r}:{n}" for r, n in base_only),
        "adapter_only": sorted(f"{r}:{n}" for r, n in adapter_only),
        "neither": len(neither), "p_case": p_case, "p_task": p_task,
        "identical_prompts": identical_prompts, "identical_config": identical_config,
        "tasks": tasks,
        "base_config": {k: base_receipt.get(k) for k in
                        ("model", "thinking", "temperature", "max_tokens", "max_attempts_per_task")},
        "adapter_config": {k: adapter_receipt.get(k) for k in
                           ("model", "thinking", "temperature", "max_tokens", "max_attempts_per_task")},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"base {base_total}/{len(keys)}  adapter {adapter_total}/{len(keys)}  "
          f"fail->pass {len(adapter_only)}  pass->fail {len(base_only)}  "
          f"p_case {p_case:.3f}  prompts_identical {identical_prompts}  "
          f"config_identical {identical_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
