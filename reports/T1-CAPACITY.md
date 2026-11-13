# T1 capacity-binding precondition: does extra dense capacity pay?

The question the rung answers, from `spec/ORRG-SPEC-v4.md` §5: on this corpus and
this token budget, does **added dense capacity** buy validation loss? If a 1.5×
wider dense model cannot beat the seed on identical batches, then an MoE arm with
extra capacity has nothing to spend it on, and any MoE-vs-dense comparison at that
budget is uninformative. This is the gate whose absence produced the `P2` null
result.

**Verdict: PASS at this scale.** The wider model wins on all three seeds, the
effect grows with tokens, and the regime is not data-bound (train loss sits *below*
validation at the end, so neither arm has started memorizing). Capacity pays here,
so an MoE arm is worth building at this regime.

This is a 2.4M-parameter rung, not the 9B gate in the spec. It certifies the
harness and the corpus regime; it does not say anything about 9B quality.

## Design

The spec's T1, scaled to one RTX 5060 Ti with size and budget as flags:

| item | value |
|---|---|
| arms | `dense` (FFN `inter=768`) vs `wide` (FFN `inter=1152`, exactly 1.5×) |
| everything else matched | hidden 192, 4 layers, 4 heads, vocab 256, tied embeddings, same LR/schedule |
| data | WikiText-103 raw bytes from the local HF cache: 80.0M bytes, 78.4M train / 1.6M held out |
| batches | offsets generated from `(seed, step)` only, so both arms of a seed see **byte-identical** batches |
| budget | 48.2M tokens = 5,886 steps × 32 × 256, identical for both arms |
| optimizer | AdamW, OneCycle, peak lr 2e-3, wd 0.01, grad clip 1.0, bf16 autocast |
| seeds | 0, 1, 2, paired (`torch.manual_seed(seed)` for both arms) |
| evaluation | 12 fixed held-out windows, evaluated at 25 / 50 / 100% of the run |
| cost | 1,175 s total (≈20 min) for all six runs |

Byte-level tokenisation is deliberate: it keeps the embedding out of the parameter
count so the FFN widening is the only difference, and it makes `tokens/param`
literal.

## Results

| arm | seed | params | tok/param | val @25% | val @50% | val final | train (last 50) | train−val |
|---|---|---|---|---|---|---|---|---|
| dense | 0 | 2.41M | 20.0 | 1.9696 | 1.8612 | 1.7702 | 1.7568 | −0.0134 |
| wide | 0 | 3.29M | 14.6 | 1.9605 | 1.8456 | **1.7483** | 1.7338 | −0.0144 |
| dense | 1 | 2.41M | 20.0 | 1.9684 | 1.8525 | 1.7631 | 1.7503 | −0.0128 |
| wide | 1 | 3.29M | 14.6 | 1.9628 | 1.8415 | **1.7493** | 1.7336 | −0.0157 |
| dense | 2 | 2.41M | 20.0 | 1.9756 | 1.8687 | 1.7789 | 1.7723 | −0.0066 |
| wide | 2 | 3.29M | 14.6 | 1.9541 | 1.8413 | **1.7486** | 1.7379 | −0.0107 |

Paired end-validation delta (`wide − dense`; negative = extra capacity pays):

| quantity | value |
|---|---|
| per seed | **−0.0219, −0.0138, −0.0303** |
| mean | **−0.0220 nats** |
| sd across seeds | 0.0083 |
| 2 sd | 0.0165 |
| 95% CI (t, df = 2) | **[−0.0425, −0.0015]** |
| gate `mean < −max(2sd, 0.005)` = −0.0165 | **PASS** |

Trajectory, mean over seeds (wide − dense):

| fraction of budget | tokens/arm | delta |
|---|---|---|
| 25% | 12.1M | −0.0121 |
| 50% | 24.1M | −0.0180 |
| 100% | 48.2M | −0.0220 |

The advantage is present at the first checkpoint and grows monotonically, so it is
not a late-training artifact of one schedule.

## What it says

* **Capacity is the binding constraint at this budget.** The wider model reaches
  the same or better validation at every checkpoint, on every seed, and its final
  loss is below the dense arm by 2.7× the across-seed sd. Nothing about the corpus
  prevents capacity from paying, which is the condition the `P2` rung failed.
* **The regime is capacity-bound, not data-bound.** `train − val` is negative for
  every arm at the endpoint (−0.0066 to −0.0157): both models still generalise
  better on held-out text than on their sampled training windows. Compare `P2`,
  where dense ended 0.213 nats above its own minimum with a +1.29 nat gap while
  memorizing 1.33M repeated tokens. That is the diagnostic the spec's T1 gate asks
  for, and it is the one that was missing before.
* **The wider model is also more stable across seeds** — final-val sd 0.0005 vs
  0.0079 for dense, a 16× difference. One observation at n = 3; it is consistent
  with the wider model being less sensitive to initialization, but it is not a
  claim.
* **The win is not free.** The wide arm has 1.367× the parameters and therefore
  ~1.37× the FLOPs per step; at matched *steps* it spends more compute for its
  0.022 nats. T1 as specified is token-matched, not compute-matched, so this rung
  answers "does the corpus reward capacity", not "is capacity a better use of a
  fixed compute budget than more steps". The compute-matched control (dense trained
  to 1.37× steps) was **not run**.

## What it does not license

* **No 9B claim.** 2.41M parameters and 48.2M byte tokens are five orders of
  magnitude from the target. The spec's T1 at 1B scale is ≈0.9k GPU-hours; this
  run is the same design at 1/400th of the model.
* **No MoE claim.** The rung is deliberately conversion-free. It says the regime
  can reward capacity; it says nothing about whether ORRG's routing, skeleton or
  expert layout captures that reward, which is T2's job (and T2's `B2`
  matched-active baseline is still required, because the wide arm here wins with
  simply more parameters).
* **One corpus, one schedule, three seeds.** WikiText-103 bytes, a single LR and a
  single widening factor (1.5×). Whether the effect saturates, reverses or grows
  at 2× widening is untested.

## Threats to validity

* **Three seeds.** With n = 3 the t-CI is wide ([−0.0425, −0.0015]) and its upper
  bound is only 0.0015 nats from zero. The sign is unanimous and the mean is 2.7×
  the sd, which is the strong part of the result; the magnitude is indicative.
* **Token budget is matched, compute is not.** See above. The wide arm's 14.6
  tok/param also means it is further from the 20 tok/param rule than dense, so if
  anything the test is conservative for capacity.
* **Held-out set is 1.6M bytes** and its windows are evenly spaced, not contiguous
  documents. Validation is comparable across arms but is not a benchmark.
* **Byte-level loss is not comparable to token-level loss** anywhere else in this
  repository. Only the paired delta is meaningful.
* **The dense seed here is random-init, not a pretrained checkpoint.** This rung
  tests a from-scratch capacity regime; a pretrained seed at a fine-tuning budget
  is a different regime, which is why the spec's token/param rule exists.

## Reproduce

```bash
PYTHONPATH=. python experiments/t1_capacity.py --steps 5886 --seeds 3 \
    --hidden 192 --layers 4 --inter 768 --batch 32 --seq 256 \
    --max-bytes 80000000 --out experiments/results/t1_capacity.json
```

Raw rows, per-checkpoint traces and the paired statistics:
`experiments/results/t1_capacity.json`. Log: `logs/t1_capacity.log`.
No checkpoints are written; the whole rung is ~20 minutes on one 16 GB card.
