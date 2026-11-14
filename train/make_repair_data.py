"""Mine verified "failing test -> fix" repair pairs from the local repos.

The Spark suite scores function replacement given a bug report, so training on
"write a function from an instruction" teaches the wrong task: the first two adapters
regressed (29/50 -> 20/50) doing exactly that, and the repair literature puts the
failing-test context at 2-3x pass@1 over a bare function-to-function prompt.

This builds the matched format without a teacher model: take a real function from
boltons / toolz / more-itertools, inject one mutation, keep the case only if the
repository's own tests fail on the mutant and pass on the original, and record the
real pytest failure text as the bug report. The target is the original function, so
every example is executable and verified in both directions.

    PYTHONPATH=. python train/make_repair_data.py --limit 400 --out train/data/repair.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOS = ROOT / "coding_research/repos"

SYSTEM = (
    "You are a senior Python engineer fixing a bug in a real repository. "
    "Return the complete replacement function in one ```python code block. "
    "Do not include prose or explanation."
)

CONTRACT = """You are fixing one bug in a real repository.
Rules:
1. Read the failing test output and the current function; change only what the bug requires.
2. Preserve public behavior unrelated to the bug.
3. Return a complete, syntactically valid replacement for the target function.
4. Every branch of a conditional that computes a value must assign that value."""

# (name, mutation) pairs applied to one source line.
MUTATIONS = (
    ("comparison flipped", lambda line: re.sub(r"<=|>=|==|!=|<|>", lambda m: {"<=": "<", ">=": ">", "==": "!=", "!=": "==", "<": "<=", ">": ">="}[m.group(0)], line, count=1)),
    ("off-by-one", lambda line: re.sub(r"\b(\d+)\b", lambda m: str(int(m.group(1)) + 1), line, count=1)),
    ("boolean negated", lambda line: re.sub(r"\breturn\b (?!not\b)(.*)$", r"return not (\1)", line, count=1)),
    ("argument dropped", lambda line: re.sub(r",\s*[A-Za-z_][\w.]*\s*\)", ")", line, count=1)),
)
SKIP_HINTS = ("raise ", "import ", "assert ", "print(", '"""', "'''", "#", "yield ")


def functions(tree: ast.Module) -> list[ast.FunctionDef]:
    return [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.end_lineno - n.lineno >= 3]


def test_module_for(repo: Path, module: Path) -> Path | None:
    stem = module.stem
    for cand in (repo / "tests" / f"test_{stem}.py", repo / f"test_{stem}.py",
                 repo / "tests" / f"{stem}_test.py"):
        if cand.exists():
            return cand
    return None


def run_tests(cwd: Path, target: Path, timeout: int = 90) -> tuple[bool, str]:
    # A stripped environment breaks python on Windows (SYSTEMROOT et al.), which made
    # every baseline run look red and the miner yield nothing.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd)
    try:
        done = subprocess.run(
            [sys.executable, "-m", "pytest", str(target.name), "-x", "-q", "--tb=short",
             "-p", "no:cacheprovider"],
            cwd=target.parent, text=True, capture_output=True, timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    return done.returncode == 0, done.stdout + done.stderr


def mutate(source_lines: list[str], start: int, end: int) -> tuple[int, str, str, str] | None:
    """Return (line index, original line, replacement or None for delete, name)."""

    candidates = [i for i in range(start, end) if source_lines[i].strip() and
                  not any(hint in source_lines[i] for hint in SKIP_HINTS)]
    random.shuffle(candidates)
    for index in candidates:
        original = source_lines[index]
        options: list[tuple[str, str | None]] = []
        for name, mutate_fn in MUTATIONS:
            changed = mutate_fn(original)
            if changed != original:
                options.append((name, changed))
        if index > start + 1:  # deleting a body line is a strong, realistic bug
            options.append(("statement deleted", None))
        random.shuffle(options)
        for name, changed in options:
            trial = list(source_lines)
            if changed is None:
                trial.pop(index)
            else:
                trial[index] = changed
            try:
                ast.parse("\n".join(trial) + "\n")
            except SyntaxError:
                continue
            return index, original, changed, name
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--out", default="train/data/repair.jsonl")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--minutes", type=float, default=20.0)
    parser.add_argument("--only", default="", help="comma-separated repo names")
    parser.add_argument("--per-function", type=int, default=1,
                        help="distinct verified mutants to emit per function")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wanted = {name.strip() for name in args.only.split(",") if name.strip()}
    written = tried = 0
    baseline: dict[Path, bool] = {}
    deadline = time.time() + 60 * args.minutes
    with out.open("a" if args.append else "w", encoding="utf-8") as handle:
        for repo in sorted(REPOS.iterdir()):
            if not repo.is_dir():
                continue
            if wanted and repo.name not in wanted:
                continue
            for module in sorted(repo.rglob("*.py")):
                if "test" in module.name or written >= args.limit:
                    continue
                tests = test_module_for(repo, module)
                if tests is None:
                    continue
                lines = module.read_text(encoding="utf-8", errors="replace").splitlines()
                try:
                    tree = ast.parse("\n".join(lines))
                except SyntaxError:
                    continue
                for node in functions(tree):
                    if written >= args.limit or time.time() > deadline:
                        break
                    emitted = 0
                    if tests not in baseline:
                        try:
                            baseline[tests] = run_tests(repo, tests)[0]
                        except Exception:
                            baseline[tests] = False
                    if not baseline[tests]:
                        continue  # suite already red: no clean baseline
                    pick = None
                    attempts = 0
                    # Walk several distinct mutants per function: the first candidate
                    # frequently survives (equivalent mutant) and yields nothing.
                    while pick is None and attempts < args.per_function * 4 and emitted == 0:
                        pick = mutate(lines, node.lineno - 1, node.end_lineno)
                        attempts += 1
                    if pick is None:
                        continue
                    index, original, changed, mutation = pick
                    tried += 1
                    backup = lines[:]
                    if changed is None:
                        lines.pop(index)
                        node_end = node.end_lineno - 1
                    else:
                        lines[index] = changed
                        node_end = node.end_lineno
                    module.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    try:
                        passed, output = run_tests(repo, tests)
                    finally:
                        lines = backup
                        module.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    if time.time() > deadline:
                        break
                    if passed or output == "TIMEOUT":
                        continue
                    failing = re.findall(r"^FAILED (\S+)", output, re.M)[:5]
                    detail = "\n".join(
                        line for line in output.splitlines()
                        if line.startswith(("E ", "assert")) or "Error" in line
                    )[-1200:]
                    buggy = "\n".join(lines[node.lineno - 1: node_end])
                    handle.write(json.dumps({
                        "messages": [
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content":
                                f"{CONTRACT}\n\n## Issue\n\n`{node.name}` in `{module.name}` "
                                f"is wrong ({mutation}). The repository's own tests fail on "
                                f"the current implementation.\n\n## Failing tests\n\n"
                                + "\n".join(f"- `{name}`" for name in failing)
                                + f"\n\n## Observed failure\n\n```text\n{detail}\n```\n\n"
                                f"## Current implementation of `{node.name}`\n\n"
                                f"```python\n{buggy}\n```\n\n"
                                f"Return the complete corrected `{node.name}` function in one "
                                "```python code block."},
                            {"role": "assistant", "content":
                                "```python\n" + "\n".join(lines[node.lineno - 1: node.end_lineno])
                                + "\n```"},
                        ],
                    }) + "\n")
                    written += 1
                    emitted += 1
                    print(f"[{written}] {repo.name}/{module.name}::{node.name} ({mutation})",
                          flush=True)
                    if emitted >= args.per_function:
                        break
    print(f"wrote {written} verified pairs from {tried} mutants to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
