"""Boltons isoparse framing matrix (v2).

Fixes the defect (fractional-second scaling removed) and varies only the input
presentation. Diff variants test targeting; function variants remove
diff-format fragility so the semantic fix can be judged directly.
"""
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spark_eval as E

ROOT = Path(__file__).resolve().parents[1]
BUGGY = ROOT / "coding_research/work/boltons_buggy"
OUT = ROOT / "reports/boltons_input_evidence_v2"
CASES = Path(__file__).resolve().parent / "isoparse_cases.py"

PRECISE_ISSUE = """`isoparse` in `boltons/timeutils.py` mishandles fractional seconds.

`isoparse('1970-01-01T00:00:00.851')` must produce microsecond 851000 (851
thousandths of a second), but it produces 851. The fractional digits after the
decimal point are a decimal fraction of one second and must be scaled to
microseconds: one digit means tenths, three digits mean thousandths, six digits
mean millionths.

More than six fractional digits must truncate to microsecond precision instead
of raising `ValueError: microsecond must be in 0..999999`; for example
`.123456789` must produce microsecond 123456.

Plain timestamps with no fraction, and timezone suffixes such as `Z` and
`+05:30`, must keep working unchanged."""

COMBINED_ISSUE = """The ISO-8601 parser `isoparse` mishandles fractional seconds.

`isoparse('1970-01-01T00:00:00.851')` returns a datetime whose microsecond
field is 851 instead of 851000: the fraction is being read as whole
microseconds rather than as a decimal fraction of a second. Inputs with more
than six fractional digits (for example `.123456789`) raise
`ValueError: microsecond must be in 0..999999` instead of truncating to
microsecond precision.

`dt_to_timestamp` must return the correct epoch seconds for these fractional
ISO-8601 timestamps once parsing is fixed.

Fix the ISO-8601 fractional-second handling in `boltons/timeutils.py`."""

VARIANTS = {
    "A_full_prior_style": dict(start=60, end=154, numbered=True, issue=COMBINED_ISSUE, mode="diff"),
    "B_full_precise": dict(start=60, end=154, numbered=True, issue=PRECISE_ISSUE, mode="diff"),
    "C_target_only": dict(start=102, end=154, numbered=True, issue=PRECISE_ISSUE, mode="diff"),
    "D_full_unnumbered": dict(start=60, end=154, numbered=False, issue=PRECISE_ISSUE, mode="diff"),
    "E_truncated_before_fraction": dict(start=60, end=140, numbered=True, issue=PRECISE_ISSUE, mode="diff"),
    "F_target_function": dict(start=102, end=154, numbered=True, issue=PRECISE_ISSUE, mode="function"),
    "G_prior_function": dict(start=60, end=154, numbered=True, issue=COMBINED_ISSUE, mode="function"),
}


def run_one(variant: str, sample: int, cfg: dict) -> dict:
    run = ROOT / f"coding_research/work/matrix_v2/{variant}_{sample}"
    if run.exists():
        shutil.rmtree(run)
    run.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(BUGGY, run)
    E.git_repo(run)

    module = run / "boltons/timeutils.py"
    code = E.excerpt(module, cfg["start"], cfg["end"], numbered=cfg["numbered"])
    prompt = E.build_prompt(cfg["issue"], "boltons", "boltons/timeutils.py",
                            code, E.DEFAULT_CONTRACT)
    if cfg["mode"] == "function":
        prompt += ("\n\nReturn the complete corrected `isoparse` function in one "
                   "```python code block. Do not return a diff.")
    reply = E.call_model(E.SYSTEM, prompt, max_tokens=3000, thinking="disabled")
    payload = {
        "variant": variant, "sample": sample, "mode": cfg["mode"],
        "excerpt_range": [cfg["start"], cfg["end"]],
        "numbered": cfg["numbered"],
        "issue_kind": "combined" if cfg["issue"] is COMBINED_ISSUE else "precise",
        "prompt_sha256": E.sha256_text(prompt), "prompt_bytes": len(prompt.encode()),
        "issue_sha256": E.sha256_text(cfg["issue"]),
        "response_sha256": E.sha256_text(reply.text), "response_bytes": len(reply.text.encode()),
        "stop_reason": reply.stop_reason, "usage": reply.usage,
        "latency_s": round(reply.latency_s, 1), "error": reply.error,
        "prompt.txt": prompt, "response.txt": reply.text,
        "thinking.txt": reply.thinking, "excerpt.txt": code,
    }
    if cfg["mode"] == "diff":
        diff = E.extract_diff(reply.text)
        applied, apply_err = E.apply_diff(run, diff)
        payload |= {"diff.txt": diff, "diff_sha256": E.sha256_text(diff),
                    "apply_ok": applied, "apply_error": apply_err}
    else:
        fn_code = E.extract_code(reply.text)
        applied, apply_err = E.replace_function(run, "boltons/timeutils.py",
                                                "isoparse", fn_code)
        payload |= {"function.txt": fn_code,
                    "function_sha256": E.sha256_text(fn_code),
                    "apply_ok": applied, "apply_error": apply_err}
    payload["changed_files"] = E.changed_files(run) if applied else []
    payload["touched_functions"] = (E.touched_functions(run, "boltons/timeutils.py")
                                    if applied else [])
    payload["patch"] = subprocess.run(["git", "diff"], cwd=run, text=True,
                                      capture_output=True).stdout if applied else ""
    if applied:
        p = subprocess.run([sys.executable, str(CASES)], cwd=run, text=True,
                           capture_output=True, timeout=120)
        try:
            results = json.loads(p.stdout)
        except Exception:
            results = [{"case": "oracle", "status": "ERROR",
                        "detail": (p.stdout + p.stderr)[:800]}]
        payload["cases"] = results
        payload["passed"] = sum(c["status"] == "PASSED" for c in results)
        payload["total"] = len(results)
        payload["test_output.txt"] = json.dumps(results, indent=1)
    E.snapshot_case(OUT / "cases" / f"{variant}_{sample}", payload)
    return {k: payload.get(k) for k in
            ("variant", "sample", "mode", "apply_ok", "touched_functions",
             "passed", "total", "stop_reason", "latency_s", "error")} | {
        "failed_cases": [c["case"] for c in payload.get("cases", [])
                         if c["status"] != "PASSED"]}


def main():
    import os
    only = [x for x in os.environ.get("MATRIX_VARIANTS", "").split(",") if x]
    samples = int(os.environ.get("MATRIX_SAMPLES", "3"))
    chosen = {k: v for k, v in VARIANTS.items() if not only or k in only}
    jobs = [(v, i, cfg) for v, cfg in chosen.items() for i in range(samples)]
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_one, v, i, cfg) for v, i, cfg in jobs]
        for f in futures:
            row = f.result()
            summary.append(row)
            print(json.dumps(row), flush=True)
    (OUT / "matrix_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
