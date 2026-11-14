# Paired per-case result: base vs repair-format adapter

- base run: `reports/final_suite/qwen3_base_oneshot` (qwen3-4b)
- adapter run: `reports/final_suite/qwen3_repair_oneshot` (qwen3-4b)
- identical prompts (sha256 of contract/system/all 50 prompts): **True**
- identical decoding config (temp/max_tokens/attempts/thinking): **True**

## 2x2

| | adapter passes | adapter fails |
|---|---:|---:|
| **base passes** | 22 | 7 |
| **base fails** | 0 | 21 |

Totals: base 29/50, adapter 22/50 (delta -7).
Flips: **0 fail→pass**, **7 pass→fail**.
McNemar exact (per case, discordant 0+7): p = 0.016.

## Per task

| task | cases | base | adapter | delta | fail→pass | pass→fail |
|---|---:|---:|---:|---:|---:|---:|
| boltons | 17 | 9 | 4 | -5 | 0 | 5 |
| more-itertools | 16 | 16 | 16 | +0 | 0 | 0 |
| toolz | 17 | 4 | 2 | -2 | 0 | 2 |

Task-level McNemar (3 tasks: 0 up, 2 down): p = 0.50. With n=3 tasks the smallest attainable two-sided p is 0.25, so no task-level significance is reachable at any score on this suite.

## Enumerated flips

- boltons pass→fail: `frac_6_digits_000001`, `isoformat_default`, `isoformat_microseconds`, `tz_aware_isoformat_roundtrip`, `tz_offset_with_fraction`
- toolz pass→fail: `toolz/tests/test_inspect_args.py::test_inspect_wrapped_property`, `toolz/tests/test_inspect_args.py::test_introspect_builtin_modules`
