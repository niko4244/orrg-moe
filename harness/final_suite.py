"""Final 50-case suite: 17 boltons + 17 toolz + 16 more-itertools cases.

One bug-fix task per repo, uniform protocol: the model returns the complete
replacement for the target function, the harness splices it in, and every
test/case is scored.  The model gets up to MAX_ATTEMPTS generations per task;
after each failed attempt it receives the function it just emitted, the names
of the failing cases, and the actual-vs-expected/raw test output.  No repair
hints or solution text are added.  Every generation and every scoring result
is retained under the run directory.

Both runs always emit 50 rows (the last successfully applied state is scored
when a replacement is missing or invalid).
"""
from __future__ import annotations

import ast
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spark_eval as E
from boltons_framing import PRECISE_ISSUE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/final_suite"
CASES = Path(__file__).resolve().parent / "isoparse_cases.py"

MAX_ATTEMPTS = int(os.environ.get("SPARK_MAX_ATTEMPTS", "6"))
FEEDBACK_CHARS = 5000

TOOLZ_ISSUE = """Signature introspection in `toolz.functoolz` is broken.

`_check_sigspec` now unconditionally returns `(None, False)`, so
`is_valid_args`, `is_partial_args`, `num_required_args`, `has_varargs`,
`has_keywords`, and `is_arity` report that every callable is invalid. For
example `num_required_args(lambda x: None)` returns `False` instead of `1`.

Restore `_check_sigspec` so it:
1. accepts an already-supplied `inspect.Signature`;
2. otherwise resolves `inspect.signature(func)`;
3. falls back to the builtin signature registry via
   `builtin_func(*builtin_args)` when no signature is available;
4. still returns `(None, False)` for a callable whose `__signature__` is
   present but invalid.

Do not change the public helpers."""

MORE_ISSUE = """`split_into` returns chunks that are one item too short.

`list(split_into([1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 4]))` returns
`[[1], [3, 4], [6, 7, 8]]` instead of `[[1, 2], [3, 4, 5], [6, 7, 8, 9]]`.
Each requested size must produce exactly that many items (fewer only when the
iterable is exhausted), a `None` size must consume all remaining items, and
the existing behavior for empty inputs, bool sizes, and invalid sizes must be
preserved."""

TASKS = [
    dict(repo="boltons", module="boltons/timeutils.py", fn="isoparse",
         buggy=ROOT / "coding_research/work/boltons_buggy",
         issue=PRECISE_ISSUE, kind="oracle", oracle=CASES, excerpt=(102, 154)),
    dict(repo="toolz", module="toolz/functoolz.py", fn="_check_sigspec",
         buggy=ROOT / "coding_research/work/toolz_buggy",
         issue=TOOLZ_ISSUE, kind="pytest",
         target="toolz/tests/test_inspect_args.py",
         excerpt_fns=["_check_sigspec"]),
    dict(repo="more-itertools", module="more_itertools/more.py", fn="split_into",
         buggy=ROOT / "coding_research/work/more_buggy",
         issue=MORE_ISSUE, kind="pytest",
         target="tests/test_more.py::SplitIntoTests",
         excerpt_fns=["split_into"]),
]


def build_excerpt(task: dict, module: Path) -> str:
    if "excerpt" in task:
        return E.excerpt(module, task["excerpt"][0], task["excerpt"][1])
    spans = [E.find_function_span(module, fn) for fn in task["excerpt_fns"]]
    start = max(1, min(s for s, _ in spans) - 6)
    end = max(e for _, e in spans) + 6
    return E.excerpt(module, start, end)


def read_function(module: Path, name: str) -> str:
    """Return the current source of one top-level function."""
    src = module.read_text(encoding="utf-8")
    tree = ast.parse(src)
    node = next(
        (n for n in tree.body
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name),
        None,
    )
    if node is None:
        raise KeyError(name)
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return "\n".join(src.splitlines()[start - 1:node.end_lineno]) + "\n"


def failure_excerpt(output: str, limit: int = FEEDBACK_CHARS) -> str:
    """Trim pytest output to the failure section plus its short summary."""
    lines = output.replace("\r\n", "\n").splitlines()
    start = next((i for i, line in enumerate(lines)
                  if line.startswith("=") and "FAILURES" in line), None)
    summary = next((i for i, line in enumerate(lines)
                    if "short test summary info" in line), None)
    body = "\n".join(lines[start:summary] if start is not None else lines)
    summary_text = "\n".join(lines[summary:]) if summary is not None else ""
    if len(body) > limit:
        body = body[:limit] + "\n...[failure section truncated]..."
    if summary_text:
        body += "\n" + "\n".join(summary_text.splitlines()[:24])
    return body.strip()


def compact_failure_table(output: str, limit: int = 16) -> str:
    """One short actual-vs-expected line per failing pytest test.

    The full traceback is retained in test_output.txt for audit, but a small
    model needs the failure signal without several kilobytes of repetition.
    """
    current = ""
    entries: dict[str, str] = {}
    for raw in output.replace("\r\n", "\n").splitlines():
        line = raw.strip()
        header = re.match(r"^_{5,}\s+(.+?)\s+_{5,}$", line)
        if header:
            current = header.group(1).strip()
            continue
        if not current or not line.startswith("E   "):
            continue
        text = line[4:].strip()
        if text.startswith(("+", "where ", "and ", "...")):
            continue
        if not text.startswith(("AssertionError:", "assert ", "ValueError:",
                                "TypeError:", "AttributeError:", "KeyError:",
                                "NameError:", "SyntaxError:", "IndexError:")):
            continue
        key = current
        entries.setdefault(key, text)
        if len(entries) >= limit:
            break
    return "\n".join(f"- `{name}`: {text}" for name, text in entries.items())


def write_boltons_feedback_test(dest: Path) -> None:
    """Materialize the 17 oracle cases as a real pytest module.

    Pytest's assertion rewriting turns each failing `assert` into the actual
    and expected values (for example `assert 851 == 851000`).  This file is
    diagnostic feedback only; the authoritative score still comes from
    `isoparse_cases.py`.
    """
    ns = runpy.run_path(str(CASES))
    lines = [
        "from datetime import datetime, timedelta, timezone",
        "from boltons.timeutils import isoparse, dt_to_timestamp",
        "",
    ]
    for index, (name, body) in enumerate(ns["CASES"]):
        lines.append(f"def test_{index:02d}_{name}():")
        body_lines = body.splitlines() or ["pass"]
        lines.extend(("    " + line) if line.strip() else "" for line in body_lines)
        lines.append("")
    dest.write_text("\n".join(lines), encoding="utf-8")


def score_task(task: dict, repo: Path, attempt_dir: Path) -> dict:
    """Run the authoritative scorer and, on failure, collect diagnostics."""
    output = ""
    details: dict[str, str] = {}
    if task["kind"] == "oracle":
        p = subprocess.run([sys.executable, str(task["oracle"])], cwd=repo,
                           text=True, capture_output=True, timeout=180)
        output = p.stdout + "\n" + p.stderr
        try:
            rows = json.loads(p.stdout)
            tests = {row["case"]: row["status"] for row in rows}
            details = {row["case"]: row.get("detail", "") for row in rows}
        except Exception as exc:  # noqa: BLE001 - malformed oracle output is a failure
            tests = {"oracle_error": "FAILED"}
            details = {"oracle_error": f"{type(exc).__name__}: {exc}"}
        failed = sorted(k for k, v in tests.items() if v != "PASSED")
        feedback = ""
        if failed:
            fb_path = (attempt_dir / "test_oracle_feedback.py").resolve()
            write_boltons_feedback_test(fb_path)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
            fb = subprocess.run(
                [sys.executable, "-m", "pytest", str(fb_path), "-q", "--tb=short",
                 "-p", "no:cacheprovider"],
                cwd=repo, text=True, capture_output=True, timeout=180, env=env,
            )
            fb_out = fb.stdout + "\n" + fb.stderr
            output += "\n\n--- diagnostic pytest of the 17 oracle cases ---\n" + fb_out
            feedback = compact_failure_table(fb_out) or failure_excerpt(fb_out)
            if not feedback:
                feedback = "\n".join(
                    f"- {name}: {details.get(name, '')}" for name in failed
                )
    else:
        res = E.run_pytest(repo, task["target"])
        tests = res["results"]
        output = res["output"]
        failed = sorted(k for k, v in tests.items() if v != "PASSED")
        feedback = compact_failure_table(output) or failure_excerpt(output)

    passed = sum(v == "PASSED" for v in tests.values())
    return {
        "tests": tests, "output": output, "feedback": feedback,
        "details": details, "passed": passed, "failed": failed,
    }


def build_repair_prompt(task: dict, current_fn: str, scored: dict) -> str:
    """Rebuild from the prior function and observed failures only.

    There is deliberately no issue restatement, task-specific rule, hint,
    skeleton, or code fragment here.  The model sees its own last function,
    the failing case names, the observed actual-vs-expected lines, and the raw
    pytest failure excerpt.
    """
    failed_lines = "\n".join(f"- `{name}`" for name in scored["failed"])
    raw = failure_excerpt(scored["output"]) or "see failing case names"
    return (
        f"## Current implementation of `{task['fn']}`\n\n"
        "This is the version that produced the results below:\n\n"
        f"```python\n{current_fn}\n```\n\n"
        f"## Test results from that implementation\n\n"
        f"{scored['passed']}/{scored['passed'] + len(scored['failed'])} cases passed. "
        f"Failing cases:\n{failed_lines}\n\n"
        f"Actual vs expected:\n{scored['feedback'] or 'see failing case names'}\n\n"
        f"Raw test output:\n```text\n{raw}\n```\n\n"
        f"## Repair task\n\n"
        f"Diagnose the root cause of every listed failure. Return a complete "
        f"corrected `{task['fn']}` function in one ```python code block. "
        "Keep the parts that already pass working, and do not return a diff. "
        "Re-check the listed failures against your new implementation before "
        "returning."
    ) + "\n"


def run_task(task: dict, run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    repo = run_dir / task["repo"]
    if repo.exists():
        shutil.rmtree(repo)
    shutil.copytree(task["buggy"], repo)
    E.git_repo(repo)

    module = repo / task["module"]
    excerpt = build_excerpt(task, module)
    initial_prompt = E.build_prompt(
        task["issue"], task["repo"], task["module"], excerpt,
        E.FUNCTION_CONTRACT,
        final_instruction=(f"Return the complete corrected `{task['fn']}` function "
                           "in one ```python code block. Do not return a diff."))
    prompt = initial_prompt
    attempts: list[dict] = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        attempt_dir = run_dir / "attempts" / task["repo"] / f"attempt_{attempt}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        reply = E.call_model(E.FUNCTION_SYSTEM, prompt, max_tokens=3000,
                             thinking="disabled")
        fn_code = E.extract_code(reply.text) if reply.text.strip() else ""
        applied, apply_err = False, "empty code block"
        if fn_code.strip():
            try:
                tree = ast.parse(fn_code)
                names = [n.name for n in tree.body
                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            except SyntaxError as exc:
                names = []
                apply_err = f"SyntaxError: {exc}"
            if task["fn"] in names:
                before = module.read_text(encoding="utf-8")
                applied, apply_err = E.replace_function(
                    repo, task["module"], task["fn"], fn_code)
                if applied:
                    try:
                        E.find_function_span(module, task["fn"])
                    except Exception:  # noqa: BLE001 - target disappeared
                        module.write_text(before, encoding="utf-8")
                        applied, apply_err = False, (
                            f"replacement did not define top-level `{task['fn']}`")
            elif not apply_err:
                apply_err = f"code block did not define `{task['fn']}`"

        scored = score_task(task, repo, attempt_dir)
        passed = scored["passed"]
        total = passed + len(scored["failed"])
        current_fn = read_function(module, task["fn"])
        function_payload = {
            "repo": task["repo"], "module": task["module"], "fn": task["fn"],
            "attempt": attempt, "max_attempts": MAX_ATTEMPTS,
            "issue": task["issue"], "excerpt": excerpt,
            "prompt_sha256": E.sha256_text(prompt),
            "prompt_bytes": len(prompt.encode("utf-8")),
            "response_sha256": E.sha256_text(reply.text),
            "response_bytes": len(reply.text.encode("utf-8")),
            "function_sha256": E.sha256_text(current_fn),
            "model_output_sha256": E.sha256_text(fn_code),
            "stop_reason": reply.stop_reason, "usage": reply.usage,
            "latency_s": round(reply.latency_s, 1), "error": reply.error,
            "apply_ok": applied, "apply_error": apply_err,
            "tests": scored["tests"], "cases": total, "passed": passed,
            "failed": scored["failed"], "feedback": scored["feedback"],
            "test_output.txt": scored["output"],
        }
        E.snapshot_case(attempt_dir, {
            **function_payload,
            "prompt.txt": prompt,
            "response.txt": reply.text,
            "function.txt": fn_code,
        })
        attempts.append({
            "attempt": attempt, "prompt_sha256": function_payload["prompt_sha256"],
            "prompt_bytes": function_payload["prompt_bytes"],
            "response_sha256": function_payload["response_sha256"],
            "function_sha256": function_payload["function_sha256"],
            "apply_ok": applied, "apply_error": apply_err,
            "passed": passed, "failed": scored["failed"],
            "stop_reason": reply.stop_reason, "error": reply.error,
        })

        if not scored["failed"]:
            break
        if attempt == MAX_ATTEMPTS:
            break
        prompt = build_repair_prompt(task, current_fn, scored)

    final_fn = read_function(module, task["fn"])
    final_tests = scored["tests"]
    record = {
        "repo": task["repo"], "module": task["module"], "fn": task["fn"],
        "kind": task["kind"], "issue": task["issue"], "excerpt": excerpt,
        "prompt_sha256": E.sha256_text(initial_prompt),
        "prompt_bytes": len(initial_prompt.encode("utf-8")),
        "function_sha256": E.sha256_text(final_fn),
        "stop_reason": attempts[-1]["stop_reason"],
        "apply_ok": attempts[-1]["apply_ok"],
        "apply_error": attempts[-1]["apply_error"],
        "touched_functions": E.touched_functions(repo, task["module"]),
        "attempts": attempts, "tests": final_tests,
        "cases": len(final_tests),
        "passed": sum(v == "PASSED" for v in final_tests.values()),
        "failed": sorted(k for k, v in final_tests.items() if v != "PASSED"),
        "function.txt": final_fn,
        "test_output.txt": scored["output"],
    }
    E.snapshot_case(run_dir / "cases" / task["repo"], record)
    return record


def main(run_label: str):
    run_dir = OUT / run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_task(t, run_dir) for t in TASKS]
    total = sum(r["cases"] for r in rows)
    passed = sum(r["passed"] for r in rows)
    receipt = {
        "run": run_label, "model": E.MODEL, "thinking": "disabled",
        "temperature": 0.0, "max_tokens": 3000,
        "protocol": "function-replacement + bounded self-repair",
        "max_attempts_per_task": MAX_ATTEMPTS,
        "feedback_limit_chars": FEEDBACK_CHARS,
        "contract": E.FUNCTION_CONTRACT,
        "contract_sha256": E.sha256_text(E.FUNCTION_CONTRACT),
        "system": E.FUNCTION_SYSTEM,
        "system_sha256": E.sha256_text(E.FUNCTION_SYSTEM),
        "per_repo": {
            r["repo"]: {
                "passed": r["passed"], "cases": r["cases"],
                "failed": r["failed"], "apply_ok": r["apply_ok"],
                "touched": r["touched_functions"],
                "attempts": len(r["attempts"]),
                "attempt_history": [
                    {"attempt": a["attempt"], "passed": a["passed"],
                     "failed": a["failed"], "apply_ok": a["apply_ok"]}
                    for a in r["attempts"]
                ],
            } for r in rows
        },
        "total_passed": passed, "total_cases": total,
        "initial_prompt_sha256": {r["repo"]: r["prompt_sha256"] for r in rows},
        "initial_prompt_bytes": {r["repo"]: r["prompt_bytes"] for r in rows},
        "function_sha256": {r["repo"]: r["function_sha256"] for r in rows},
        "attempt_prompt_sha256": {
            r["repo"]: [a["prompt_sha256"] for a in r["attempts"]] for r in rows
        },
    }
    (run_dir / "receipt.json").write_text(json.dumps(receipt, indent=2),
                                          encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run1")
