# Paired per-case result: base vs repair-format adapter

- base run: `reports/final_suite/qwen3_base_oneshot` (qwen3-4b)
- adapter run: `reports/final_suite/qwen3_repair2_oneshot` (qwen3-4b)
- identical prompts (sha256 of contract/system/all 50 prompts): **True**
- identical decoding config (temp/max_tokens/attempts/thinking): **True**

## 2x2

| | adapter passes | adapter fails |
|---|---:|---:|
| **base passes** | 27 | 2 |
| **base fails** | 0 | 21 |

Totals: base 29/50, adapter 27/50 (delta -2).
Flips: **0 fail→pass**, **2 pass→fail**.
McNemar exact (per case, discordant 0+2): p = 0.500.

## Per task

| task | cases | base | adapter | delta | fail→pass | pass→fail |
|---|---:|---:|---:|---:|---:|---:|
| boltons | 17 | 9 | 9 | +0 | 0 | 0 |
| more-itertools | 16 | 16 | 16 | +0 | 0 | 0 |
| toolz | 17 | 4 | 2 | -2 | 0 | 2 |

Task-level McNemar (3 tasks: 0 up, 1 down): p = 1.00. With n=3 tasks the smallest attainable two-sided p is 0.25, so no task-level significance is reachable at any score on this suite.

## Enumerated flips

- toolz pass→fail: `toolz/tests/test_inspect_args.py::test_introspect_builtin_modules`, `toolz/tests/test_inspect_args.py::test_introspect_curry_py3`
