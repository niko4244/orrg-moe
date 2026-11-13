import hashlib
import json
from pathlib import Path

root = Path("I:/.freebuff")
r1 = json.load(open(root / "reports/final_suite/run1/cases/boltons/case.json", encoding="utf-8"))
rec1 = json.load(open(root / "reports/final_suite/run1/receipt.json"))
rec2 = json.load(open(root / "reports/final_suite/run2/receipt.json"))
prompt = r1["prompt.txt"]
prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
receipt1 = hashlib.sha256((root / "reports/final_suite/run1/receipt.json").read_bytes()).hexdigest()
receipt2 = hashlib.sha256((root / "reports/final_suite/run2/receipt.json").read_bytes()).hexdigest()

rows = [
    ("basic_epoch", "PASS", "whole seconds"),
    ("frac_3_digits_851", "FAIL", "parsed 851us, needs 851000us"),
    ("frac_leading_zero_051", "FAIL", "parsed 51us, needs 51000us"),
    ("frac_6_digits_000001", "PASS", "exactly 6 digits: 1us"),
    ("frac_overprecise_truncates", "FAIL", "ValueError: 123456789 > 999999, no truncation"),
    ("frac_1_digit_5", "FAIL", "parsed 5us, needs 500000us"),
    ("frac_2_digits_05", "FAIL", "parsed 5us, needs 50000us"),
    ("frac_4_digits_0001", "FAIL", "parsed 1us, needs 100us"),
    ("isoformat_milliseconds", "FAIL", ".851 parsed as 851us"),
    ("isoformat_microseconds", "PASS", "exactly 6 digits"),
    ("isoformat_default", "PASS", "exactly 6 digits"),
    ("tz_zulu", "PASS", "no fraction"),
    ("tz_offset_plus_0530", "PASS", "no fraction"),
    ("tz_offset_with_fraction", "PASS", "only offset asserted"),
    ("tz_aware_isoformat_roundtrip", "PASS", "exactly 6 digits"),
    ("tz_invalid_offset_raises", "PASS", "raises before parsing"),
    ("dt_to_timestamp_fraction", "FAIL", "0.000851 vs 0.851"),
]
table = "\n".join(
    "| %d | `%s` | %s | %s |" % (i + 1, n, s, why)
    for i, (n, s, why) in enumerate(rows)
)

body = '''# Spark vs Opus 5.5 - final coding evaluation, 2026-11-08

## Verdict

- Model: `claude-spark-x2.6-local` (local Ollama `spark-x2.6-4b:q6_k`, 4.1B Q6_K), served via `http://127.0.0.1:4000`.
- Two complete runs of the rebuilt 50-case suite: **27/50 both**, with matching per-repo results and byte-identical model outputs.
- Boltons root cause: **model capability on the fractional-second scaling step**, decided from input-level evidence. The complete target function was in view; the model emitted the same semantically wrong function in both runs.
- Opus 5.5: **UNMEASURABLE** (HTTP 402, insufficient credits). No repo is claimed as matching Opus.
- Prior artifacts are **not present on this machine**, so the earlier v4.6/v4.7 numbers and the original 16 boltons failures could not be audited or reproduced. See section 1.

## 1. Provenance and what is not verifiable

The workspace contained no `harness/swe_style_realrepo_v2_scaffolded_v4_7_20261108.py`, no `reports/paired_coding_eval_20261108/`, no `coding_research/real_repo_v1/manifest_v2_20261030.json`, no v4.7 prompt template, and no prior receipts. The v4.6/v4.7 34/50 numbers and the 16 original boltons failures therefore cannot be certified from disk, and this report does not claim they are.

Everything below is a fresh, reproducible evaluation built from real upstream repositories at pinned commits:

| repo | commit | test selection | cases |
|---|---|---|---|
| boltons | `d31d821d56c8f7756af9c30a864077f2e5855321` | 17-case `isoparse` oracle | 17 |
| toolz | `451af60dec590a6010e2babdbf391ea8f815122f` | `toolz/tests/test_inspect_args.py` | 17 |
| more-itertools | `f42daddf205f335fea6e200dad46049f33e71221` | `tests/test_more.py::SplitIntoTests` | 16 |

## 2. Final configuration

- Protocol: one bug-fix per repo; the model returns the complete replacement function in a python code block; the harness splices it in and scores every test/case. An invalid replacement leaves the buggy baseline in place, so every run always has 50 scored cases.
- `thinking: disabled`, `temperature: 0.0`, `max_tokens: 3000`.
- Contract SHA-256: `%s`
- System SHA-256: `%s`
- Boltons prompt: **%d bytes**, SHA-256 `%s` (verbatim in Appendix A and `reports/final_suite/BOLTONS_PROMPT_VERBATIM.txt`).

## 3. Two complete runs

| repo | run 1 | run 2 | baseline (buggy) |
|---|---|---|---|
| boltons | 9/17 | 9/17 | 9/17 |
| toolz | 4/17 | 4/17 | 1/17 |
| more-itertools | 14/16 | 14/16 | 4/16 |
| **total** | **27/50** | **27/50** | 14/50 |

Receipts:

- run 1: `reports/final_suite/run1/receipt.json`, SHA-256 `%s`
- run 2: `reports/final_suite/run2/receipt.json`, SHA-256 `%s`

Output hashes (identical in both runs):

- boltons function `%s`
- toolz function `b0260eeb664b33f83ac06d5688bed40dc2a66f601f75dea7a7337267f92959f1`
- more-itertools function `46262255d0be119a925ed2bd9e1f9735c3cd19dfa3867583fc9faf469d183bff`

Prompt hashes: boltons `%s`, toolz `%s`, more-itertools `%s`.

## 4. Boltons: input-level proof

### 4.1 Input actually sent

- Issue names `isoparse`, the exact failing example `.851`, the required 851000 us result, the >6-digit truncation rule, and the requirement that plain and timezone-suffixed inputs keep working. Issue SHA-256 `%s`.
- Excerpt: source lines **102-154**, **%d bytes**, **53 lines**. First line `def isoparse(iso_str):`; last line `return datetime(*dt_args, tzinfo=tzinfo)`. It contains the complete function: signature, docstring, timezone parsing, `parts`/`dt_args`, and the return. No truncation; no needed helper is missing.
- This is the target-only excerpt, so the function is unambiguously in view.

### 4.2 Output actually emitted

The model returned a complete, syntactically valid replacement (2,144 bytes, SHA-256 `%s`). Its functional core:

```python
    parts = _NONDIGIT_RE.split(iso_str)
    dt_args = [int(p) for p in parts]
    microsecond = 0
    if len(parts) > 6:
        microsecond = int(parts[6])
    return datetime(*dt_args[:6], microsecond=microsecond, tzinfo=tzinfo)
```

This is wrong for the issue's own examples: `.851` becomes 851 us, not 851000; `.5` becomes 5 us, not 500000; `.123456789` raises `ValueError` instead of truncating to 123456. It passes exact six-digit fractions only because those digits already equal microseconds.

### 4.3 Per-case results (identical in both runs)

| # | case | result | reason |
|---|---|---|---|
%s

All 8 failures share one root cause: the decimal fraction is never scaled by a power of ten, and over-precise digits are not truncated. No failure is attributable to a missing function, a truncated excerpt, or a mislocated issue. The two passes are identical case-for-case and byte-identical in output.

### 4.4 Alternative explanations tested

- **Mislocation to `dt_to_timestamp`/`total_seconds`:** not reproduced. In a prior-style framing (full `dt_to_timestamp`+`isoparse` excerpt, combined issue naming both), all 3 sampled patches touched `isoparse` only and scored 5/17. The model located the parser and failed the arithmetic.
- **Thinking/token starvation:** real and removed. With thinking enabled and a 12,000-token budget, the model spent 12,000/12,000 tokens in a repeating thinking block (misreading `re.split(r'\\D')`) and emitted a 0-byte answer. Disabling thinking produced complete answers.
- **Diff-format artifact:** real and removed. Several diff-mode outputs had valid bodies but stale `@@` counts; hunk-count normalization made them apply. The final protocol uses function replacement, so patch formatting cannot explain the 8 failures.
- **Prompt contradiction:** found and fixed. `build_prompt` appended "Return only the unified diff" directly before "Do not return a diff" in early passes; those passes are archived as invalid (`run1_invalid_protocol*`). After removal the score moved 22 to 27, which is why the final verdict uses the corrected prompt.

### 4.5 Conclusion

For all 17 reconstructed boltons cases the model received the correct issue and a complete in-view excerpt of `isoparse`, and still emitted the same semantically wrong function in two independent runs. Under the stated criterion this is a **model-capability failure on the fractional-second scaling step**, not a defective input. It cannot certify the original 16 cases from the missing v4.6/v4.7 artifacts.

## 5. Other repositories

- **toolz, 4/17:** emitted `_check_sigspec` returns `(sigspec, True)` instead of `(sigspec, None)`, so every introspection helper reads the second element as failure/unknown; 13 tests fail. The target function was fully in view.
- **more-itertools, 14/16:** the core chunking fix is correct, but `False` in `sizes` reaches `islice(it, -1)` and `[]` raises `TypeError` instead of `ValueError`; `test_bool_in_sizes` and `test_invalid_in_sizes` fail.
- No repo is claimed as matching Opus. Boltons is at its buggy baseline; toolz and more-itertools improve on their baselines but remain below the suite ceiling.

## 6. Opus 5.5 status

A direct request to `POST https://openrouter.ai/api/v1/chat/completions` with model `anthropic/claude-opus-5.5` returned **HTTP 402**, body beginning **"Insufficient credits. This account never purchased credits..."**. It was not re-probed in this turn.

Required user action: fund <https://openrouter.ai/settings/credits> or provide an Anthropic API key with credit. Then the identical 50-case harness can produce the paired baseline. Until then, Spark-vs-Opus parity is **UNMEASURABLE**, not "behind" by a measured margin.

## 7. Limitations

- The original v4.6/v4.7 harness, prompt, manifest, and receipts are absent; this report supersedes them with a fresh suite rather than reproducing them.
- 27/50 is a small three-task suite with injected defects, not a SWE-bench-scale estimate.
- The boltons oracle is a 17-case reconstruction of the `isoparse` contract, not the lost 16-case run.
- Local-model results are hardware- and quantization-specific.

## Appendix A - exact boltons prompt (verbatim)

```text
@@PROMPT@@
```
''' % (
    rec1["contract_sha256"], rec1["system_sha256"], len(prompt.encode()), prompt_sha,
    receipt1, receipt2, r1["function_sha256"],
    rec1["prompt_sha256"]["boltons"], rec1["prompt_sha256"]["toolz"],
    rec1["prompt_sha256"]["more-itertools"],
    hashlib.sha256(r1["issue"].encode()).hexdigest(), len(r1["excerpt"].encode()),
    r1["function_sha256"], table,
)

body = body.replace("@@PROMPT@@", prompt)
out = root / "reports/FINAL_SPARK_VS_OPUS_20261108.md"
out.write_bytes(body.encode("utf-8"))
print("report bytes", len(body.encode()), "sha", hashlib.sha256(body.encode()).hexdigest())
print("prompt bytes", len(prompt.encode()), "sha", prompt_sha)
print("runs", rec1["total_passed"], rec2["total_passed"], rec1["per_repo"] == rec2["per_repo"])
