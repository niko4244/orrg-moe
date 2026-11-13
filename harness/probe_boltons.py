"""Probe: prior-style boltons isoparse prompt against Spark."""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spark_eval as E

ROOT = Path(__file__).resolve().parents[1]
BUGGY = ROOT / "coding_research/work/boltons_buggy"
RUN = ROOT / "coding_research/work/probe_A"

ISSUE = """The ISO-8601 parser `isoparse` mishandles fractional seconds.

`isoparse('1970-01-01T00:00:00.851')` returns a datetime whose microsecond
field is 851 instead of 851000: the fraction is being read as whole
microseconds rather than as a decimal fraction of a second. Inputs with more
than six fractional digits (for example `.123456789`) raise
`ValueError: microsecond must be in 0..999999` instead of truncating to
microsecond precision.

`dt_to_timestamp` must return the correct epoch seconds for these fractional
ISO-8601 timestamps once parsing is fixed.

Fix the ISO-8601 fractional-second handling in `boltons/timeutils.py`."""


def main():
    if RUN.exists():
        shutil.rmtree(RUN)
    shutil.copytree(BUGGY, RUN)
    E.git_repo(RUN)

    module = RUN / "boltons/timeutils.py"
    code = E.excerpt(module, 60, 154, numbered=True)
    prompt = E.build_prompt(ISSUE, "boltons", "boltons/timeutils.py", code,
                            E.DEFAULT_CONTRACT)
    print("prompt bytes", len(prompt.encode()))
    reply = E.call_model(E.SYSTEM, prompt, max_tokens=12000)
    print("error:", reply.error)
    print("usage:", reply.usage, "stop:", reply.stop_reason, "latency", round(reply.latency_s, 1))
    print("=== THINKING (tail) ===")
    print(reply.thinking[-3000:])
    print("=== TEXT ===")
    print(reply.text[:6000])

    diff = E.extract_diff(reply.text)
    ok, err = E.apply_diff(RUN, diff)
    print("=== APPLY ===", ok, err[:1000])
    payload = {
        "variant": "A_full_prior_style",
        "issue": ISSUE,
        "contract": E.DEFAULT_CONTRACT,
        "prompt.txt": prompt,
        "response.txt": reply.text,
        "thinking.txt": reply.thinking,
        "diff.txt": diff,
        "excerpt.txt": code,
        "prompt_sha256": E.sha256_text(prompt),
        "prompt_bytes": len(prompt.encode()),
        "issue_sha256": E.sha256_text(ISSUE),
        "response_sha256": E.sha256_text(reply.text),
        "diff_sha256": E.sha256_text(diff),
        "usage": reply.usage,
        "latency_s": reply.latency_s,
        "request_sha256": reply.request_sha256,
        "apply_ok": ok,
        "apply_error": err,
        "changed_files": E.changed_files(RUN) if ok else [],
        "touched_functions": E.touched_functions(RUN, "boltons/timeutils.py") if ok else [],
    }
    if ok:
        res = E.run_pytest(RUN, "tests/test_timeutils.py")
        payload["tests"] = res["results"]
        payload["test_output.txt"] = res["tail"]
        payload["passed"] = sum(v == "PASSED" for v in res["results"].values())
        payload["total"] = len(res["results"])
    E.snapshot_case(ROOT / "reports/boltons_input_evidence/cases/A_full_prior_style", payload)
    print("=== TESTS ===", payload.get("passed"), "/", payload.get("total"))
    print(json.dumps(payload.get("tests", {}), indent=1))


if __name__ == "__main__":
    main()
