"""Mine verified repair pairs from a repository's own commit history.

Mutation mining only produces synthetic bugs. Real history gives the other half: a
commit that changed a function is normally a fix, so the parent version is a genuine
pre-fix implementation and the child version is the accepted repair. This keeps the
pair only when substituting the parent function into the child tree makes the module's
own tests fail, which supplies the real failing-test text the training prompt needs.

Each worker runs on its own copy of the repository so several can run at once.

    PYTHONPATH=. python train/mine_history.py --repo boltons --copy /tmp/h0 \
        --skip 0 --stride 1 --limit 300 --minutes 20 --out train/data/hist_h0.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
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


def git(cwd: Path, *args: str) -> str:
    # Some historical blobs are not UTF-8; decode leniently instead of crashing the worker.
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True)
    return done.stdout.decode("utf-8", errors="replace")


def top_functions(source: str) -> dict[str, tuple[int, int, str]]:
    """name -> (start line, end line, source text) for module-level functions."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    lines = source.splitlines()
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.end_lineno - node.lineno >= 3:
            out[node.name] = (node.lineno, node.end_lineno,
                              "\n".join(lines[node.lineno - 1: node.end_lineno]))
    return out


def run_module_tests(repo: Path, module: Path, timeout: int = 90) -> tuple[bool, str]:
    tests = None
    for cand in (repo / "tests" / f"test_{module.stem}.py", repo / f"test_{module.stem}.py"):
        if cand.exists():
            tests = cand
            break
    if tests is None:
        return True, ""  # no suite: can never verify, so never emit
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    try:
        done = subprocess.run(
            [sys.executable, "-m", "pytest", tests.name, "-x", "-q", "--tb=short",
             "-p", "no:cacheprovider"],
            cwd=tests.parent, text=True, capture_output=True, timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    return done.returncode == 0, done.stdout + done.stderr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--copy", required=True)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--minutes", type=float, default=20.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source_repo = REPOS / args.repo
    work = Path(args.copy)
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(source_repo, work, ignore=shutil.ignore_patterns(".git", "__pycache__"))

    commits = [line for line in git(source_repo, "log", "--format=%H", "--name-only",
                                    "--", ".").splitlines()]
    # Group: a commit hash is followed by the files it touched.
    history: list[tuple[str, list[str]]] = []
    for line in commits:
        line = line.strip()
        if not line:
            continue
        if re.fullmatch(r"[0-9a-f]{40}", line):
            history.append((line, []))
        elif history:
            history[-1][1].append(line)
    history = history[args.skip::args.stride]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + 60 * args.minutes
    written = tried = 0
    baseline: dict[str, bool] = {}
    with out.open("w", encoding="utf-8") as handle:
        for commit, files in history:
            if written >= args.limit or time.time() > deadline:
                break
            for name in files:
                if not name.endswith(".py") or "test" in name:
                    continue
                parent_src = git(source_repo, "show", f"{commit}^:{name}")
                child_src = git(source_repo, "show", f"{commit}:{name}")
                if not parent_src or not child_src or parent_src == child_src:
                    continue
                parent_fns = top_functions(parent_src)
                child_fns = top_functions(child_src)
                module = work / name
                if not module.exists():
                    continue
                for fn, (start, end, child_text) in child_fns.items():
                    if written >= args.limit or time.time() > deadline:
                        break
                    if fn not in parent_fns:
                        continue  # added by the commit: nothing to repair from
                    parent_text = parent_fns[fn][2]
                    if parent_text == child_text:
                        continue
                    tried += 1
                    child_lines = child_src.splitlines()
                    mutant = child_lines[: start - 1] + parent_text.splitlines() + child_lines[end:]
                    key = name
                    if key not in baseline:
                        module.write_text(child_src, encoding="utf-8")
                        baseline[key] = run_module_tests(work, module)[0]
                    if not baseline[key]:
                        continue  # suite red before any change: unusable baseline
                    module.write_text("\n".join(mutant) + "\n", encoding="utf-8")
                    try:
                        passed, output = run_module_tests(work, module)
                    finally:
                        module.write_text(child_src, encoding="utf-8")
                    if passed or output == "TIMEOUT":
                        continue
                    failing = re.findall(r"^FAILED (\S+)", output, re.M)[:5]
                    detail = "\n".join(
                        line for line in output.splitlines()
                        if line.startswith(("E ", "assert")) or "Error" in line
                    )[-1200:]
                    handle.write(json.dumps({
                        "messages": [
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content":
                                f"{CONTRACT}\n\n## Issue\n\n`{fn}` in `{Path(name).name}` is "
                                "wrong. The repository's own tests fail on the current "
                                "implementation.\n\n## Failing tests\n\n"
                                + "\n".join(f"- `{item}`" for item in failing)
                                + f"\n\n## Observed failure\n\n```text\n{detail}\n```\n\n"
                                f"## Current implementation of `{fn}`\n\n"
                                f"```python\n{parent_text}\n```\n\n"
                                f"Return the complete corrected `{fn}` function in one "
                                "```python code block."},
                            {"role": "assistant", "content":
                                "```python\n" + child_text + "\n```"},
                        ],
                        "origin": {"repo": args.repo, "commit": commit, "path": name,
                                   "function": fn, "source": "history"},
                    }) + "\n")
                    written += 1
                    print(f"[{written}] {args.repo}@{commit[:8]} {name}::{fn}", flush=True)
    print(f"wrote {written} verified history pairs from {tried} changed functions to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
