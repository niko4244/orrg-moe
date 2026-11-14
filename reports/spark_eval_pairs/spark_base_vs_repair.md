# Paired per-case result: base vs repair-format adapter

- base run: `reports/final_suite/spark_x26_base_oneshot` (spark-x2.6-4b-hf)
- adapter run: `reports/final_suite/spark_x26_repair_oneshot` (spark-x2.6-4b-hf)
- identical prompts (sha256 of contract/system/all 50 prompts): **True**
- identical decoding config (temp/max_tokens/attempts/thinking): **True**

## 2x2

| | adapter passes | adapter fails |
|---|---:|---:|
| **base passes** | 23 | 6 |
| **base fails** | 1 | 20 |

Totals: base 29/50, adapter 24/50 (delta -5).
Flips: **1 fail→pass**, **6 pass→fail**.
McNemar exact (per case, discordant 1+6): p = 0.125.

## Per task

| task | cases | base | adapter | delta | fail→pass | pass→fail |
|---|---:|---:|---:|---:|---:|---:|
| boltons | 17 | 9 | 4 | -5 | 0 | 5 |
| more-itertools | 16 | 16 | 15 | -1 | 0 | 1 |
| toolz | 17 | 4 | 5 | +1 | 1 | 0 |

Task-level McNemar (3 tasks: 1 up, 2 down): p = 1.00. With n=3 tasks the smallest attainable two-sided p is 0.25, so no task-level significance is reachable at any score on this suite.

## Enumerated flips

- boltons pass→fail: `frac_6_digits_000001`, `isoformat_default`, `isoformat_microseconds`, `tz_aware_isoformat_roundtrip`, `tz_offset_with_fraction`
- more-itertools pass→fail: `tests/test_more.py::SplitIntoTests::test_invalid_in_sizes`
- toolz fail→pass: `toolz/tests/test_inspect_args.py::test_is_valid_curry`
