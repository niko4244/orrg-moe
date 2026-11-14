# Repair-format adapter round — 2026-11-14

## Verdict first

A repair-format LoRA adapter trained on 234 **verified** repair pairs now exists and loads
behind the unchanged `/v1/messages` shim. It was scored on the identical 50 cases against
the untuned base, twice (two schedules), and **neither run improved the model**:

| arm | boltons | toolz | more-itertools | total | vs base |
|---|---:|---:|---:|---:|---|
| base `Qwen3-4B-Instruct-2507` (no adapter) | 9/17 | 4/17 | 16/16 | **29/50** | — |
| `repair_v1` — r16, 200 steps, 695,922 tokens | 4/17 | 2/17 | 16/16 | **22/50** | −7 |
| `repair_v2` — r8, 60 steps, 208,763 tokens | 9/17 | 2/17 | 16/16 | **27/50** | −2 |

Paired, per case, both arms on the same prompts (verified by sha256 of all 50 prompt
strings) and the same decoding config (temperature 0.0, max_tokens 3000, attempts 1,
thinking disabled):

| comparison | both pass | base only | adapter only | both fail | McNemar exact |
|---|---:|---:|---:|---:|---|
| base vs `repair_v1` | 22 | 7 | 0 | 21 | p = 0.016 |
| base vs `repair_v2` | 27 | 2 | 0 | 21 | p = 0.50 |

**There were zero fail→pass flips in either arm.** Every case the base model could not
solve, the adapter also could not solve. The only movement was in the other direction:
5 boltons + 2 toolz cases lost by `repair_v1`, and 2 toolz cases lost by `repair_v2`.
The gentler schedule halved the damage (22 → 27) but produced no gain, which rules out
"the first adapter was just over-trained" as the whole explanation.

On the task-level reading, `repair_v2` ties on 2 of 3 tasks (boltons 9 vs 9,
more-itertools 16 vs 16) and loses toolz 4 → 2; at n=3 tasks the smallest attainable
two-sided p is 0.25, so even that cannot be called significant. The per-case McNemar
p = 0.016 for `repair_v1` is the only formally "significant" number here, and it says
the regression is real, not that anything improved.

## What the training data was

`train/data/repair_all.jsonl` — 234 unique pairs, deduplicated on (prompt, target),
sha256 `5eab4b30…a978425f`. Every pair is executable in both directions: the prompt
contains a real failing-test output and the target is the real corrected function.

| source | pairs | how it was verified |
|---|---:|---|
| boltons commit history (4 parallel workers, own repo copies) | 162 | parent function substituted into the child tree makes that module's own tests fail; child passes |
| boltons mutation mining (3 mutants/function) | 39 | injected mutant fails the repo's tests, original passes |
| more-itertools mutation mining | 6 | same |
| earlier `repair.jsonl` set (survivors after dedupe) | 27 | same |

Prompt shape is exactly the suite's shape: contract, issue line, failing test names, a
real traceback excerpt (mean 2,413 chars), the buggy function, and the instruction to
return the corrected function in one code block. Targets mean 1,582 chars. The generic
instruction corpus that produced the earlier 20/50 regression was **not** used.

Toolz and more-itertools produced almost nothing from history (both are single-commit
snapshots here), and their mutation yield was low because most modules have no
per-module test file — 0 pairs for toolz, 6 for more-itertools.

## Training, exactly

| | `repair_v1` | `repair_v2` |
|---|---|---|
| base | `cache/models/qwen3-4b` (Qwen3-4B-Instruct-2507, 4-bit nf4) | same |
| adapter | LoRA r16, α32, all 7 projection types, 33.0M trainable (0.81%) | r8, α16, same targets, 16.5M (0.41%) |
| schedule | OneCycle, lr 2e-4 | OneCycle, lr 1e-4 |
| optimiser steps | **200** | **60** |
| micro-batches | 800 (batch 1 × grad-accum 4) | 240 (batch 1 × grad-accum 4) |
| sequence cap | 1,216 tokens | 1,216 tokens |
| tokens actually trained | **695,922** (padding excluded) | **208,763** |
| epochs over 234 pairs | ≈3.4 | ≈1.0 |
| wall clock | 1,500 s (7.5 s/step) | 458 s (7.6 s/step) |
| final train loss | 0.018 | 0.176 |

`train_meta.json` inside each adapter records all of the above plus the data hash.

## Resource reductions forced by the card

The 16 GB card was held by other tenants for most of this round: 33 compute processes,
15.78 GiB of 16.31 GiB in use, 98–100% utilisation when first measured, and two earlier
attempts at batch 2 × accum 8 × 1,536 tokens produced no optimiser step at all in
200 s. What was cut, and what it cost:

| reduction | from | to | consequence |
|---|---|---|---|
| micro-batch | 2 | 1 | halves activation memory; same effective batch via grad-accum 4 |
| sequence cap | 2,048/1,536 | 1,216 | a few long prompts are truncated at the tail |
| rank | 32 / 16 | 16 / 8 | fewer trainable params, slightly less capacity |
| steps | 150–400 | 200 then 60 | 3.4 epochs then 1.0 epoch |

Measured effect: 7.4 s/optimiser step instead of "no step completes", and both runs
finished (1,500 s and 458 s) rather than dying on `bitsandbytes` illegal memory access
or starving. This is why the step budget is small; it is a hardware limit, not a
hyperparameter choice.

## Why this is not evidence against the approach, and not evidence for it either

Three adapters have now been trained on this base: generic instruction corpus 20/50,
repair-format r16 22/50, repair-format r8 27/50, against a 29/50 base. The pattern is
consistent: fine-tuning a 4B model on 200–6,000 examples at these step counts degrades
this suite rather than improving it, and the degradation is concentrated where the base
was already weak (boltons, toolz) plus a handful of previously-passing cases.

The honest reading is that the limit is **data scale plus task difficulty**, not the
format. With 234 pairs the adapter can only learn the local-edit style of its training
targets — in 162 of those pairs the fix is a small diff — and this suite's cases require
behaviour the base model does not have at all (fractional-second ISO-8601 parsing, sigspec
introspection). 0 fail→pass across 50 cases in both arms is the quantitative statement of
that: there was nothing for the adapter to sharpen, because it never solved these cases
before training either.

## Reproduction

```
# data (parallel, own repo copy each)
PYTHONPATH=. python train/mine_history.py --repo boltons --copy /tmp/h0 --skip 0 --stride 4 \
    --limit 250 --minutes 18 --out train/data/hist_h0.jsonl     # x4 with --skip 1..3
PYTHONPATH=. python train/make_repair_data.py --only boltons --per-function 3 --limit 900 \
    --minutes 26 --out train/data/repair_boltons.jsonl
PYTHONPATH=. python train/merge_repair.py --out train/data/repair_all.jsonl

# train
PYTHONPATH=. python train/qlora_sft.py --steps 60 --batch 1 --accum 4 --max-len 1216 \
    --lr 1e-4 --rank 8 --data train/data/repair_all.jsonl --out train/adapters/repair_v2

# score, unchanged harness and shim
PYTHONPATH=. python train/serve_anthropic.py --model cache/models/qwen3-4b \
    --adapter train/adapters/repair_v2 --port 4012
ANTHROPIC_BASE_URL=http://127.0.0.1:4012 ANTHROPIC_AUTH_TOKEN=x SPARK_MODEL=qwen3-4b \
    SPARK_MAX_ATTEMPTS=1 PYTHONPATH=. python harness/final_suite.py qwen3_repair2_oneshot

# paired analysis
PYTHONPATH=. python train/eval_pairs.py --base reports/final_suite/qwen3_base_oneshot \
    --adapter reports/final_suite/qwen3_repair2_oneshot --out reports/spark_eval_pairs/base_vs_repair_v2.md
```

Receipts committed per case: `reports/final_suite/{qwen3_base_oneshot,qwen3_repair_oneshot,
qwen3_repair2_oneshot}/cases/*/case.json` plus each run's `receipt.json`. Paired
tables: `reports/spark_eval_pairs/base_vs_repair_v1.md` and `..._v2.md`.

## What this round changed in the codebase

- `train/mine_history.py` — history-derived verified pairs (new): walks commits, pairs
  parent→child function versions, keeps only pairs whose parent version fails the repo's
  tests in the child tree, and records the real pytest failure text.
- `train/make_repair_data.py` — gained `--only`, `--per-function N` and multi-mutant
  sampling per function.
- `train/merge_repair.py` — dedupe/shuffle across parallel workers (new).
- `train/eval_pairs.py` — paired 2x2, per-case McNemar, per-task deltas, and an assertion
  that both arms used byte-identical prompts and identical decoding config (new).
- `train/qlora_sft.py` — exact token accounting and a `train_meta.json` beside every
  adapter.

## Still open

1. **Spark itself remains untrainable here** — it is a Q6_K GGUF in Ollama. Everything
   above is measured on the `Qwen3-4B-Instruct-2507` **stand-in**, not on Spark's weights.
2. **Data scale is the binding constraint.** 234 pairs is ~1% of the 30k–65k that LoRA
   recipes report for behavioural change. The miner sustains ~2.5 mutation pairs/min per
   worker and ~12 history pairs/min per worker; reaching 20k pairs needs hours of mining
   and a corpus of repos with fast per-module suites, not three.
3. **No teacher model** is available (no route to a frontier API), so distillation-style
   targets cannot be generated; the correctness ceiling of the training data is the
   repos' own tests.
4. **"Opus 5.5-level" is not reachable on this hardware** and nothing in this round
   should be read as progress toward it. The measured facts are: base 29/50 one-shot,
   adapters 20–27/50, zero fail→pass flips, one 4B card shared with other tenants.
