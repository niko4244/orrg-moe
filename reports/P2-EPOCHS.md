# P2 two-epoch rung: dense saturates, ORRG arms in flight

## Question

At 0.6 epoch the dense baseline reached 2.8204 and every ORRG variant matched it
within 0.01 nats. The open question was whether residual capacity starts to pay
once the dense model is data-bound. This rung trains the same arms for two epochs
on the same corpus.

## Design

| item | value |
|---|---|
| seed model | `HuggingFaceTB/SmolLM2-360M` (361.8M, 32 layers) |
| corpus | WikiText-103 raw, 1.33M train tokens |
| budget | 2,660 steps × 4 × 256 = **2.72M tokens ≈ 2.05 epochs** |
| schedule | Adafactor + OneCycle over 2,660 steps, lr 2e-4, bf16 |
| shape | `shared=2, groups=4, variants=3, k=1` |
| arms | `dense_continued`, `orrg_all`, `orrg_orth_mag` (one process each, sequential) |
| trace | every 100 steps |

Because the OneCycle schedule is stretched over 2,660 steps, intermediate trace
values are **not** comparable to the 800-step rung. Only full endpoints are.

## Dense baseline (complete)

| arm | val before | val after | Δ | s/step |
|---|---|---|---|---|
| dense_continued | 3.0257 | **3.0171** | −0.0086 | 0.46 |

Validation loss by step: 2.8039 (100), 2.8575 (200), 2.8864 (300), 2.8932 (400),
2.9042 (500), 2.9248 (600), 2.9448 (900), 2.9547 (1400), 2.9709 (1500), 2.9986
(1900), 2.9995 (2200), 3.0171 (final).

Dense overfits almost immediately and ends two epochs essentially where it
started: 3.0171 after 2.72M tokens against 3.0257 before training. Compare its
0.6-epoch endpoint on the shorter schedule, 2.8204 — the extra 1.4 epochs cost
about 0.20 nats. The schedule differs between the two runs, so this is not a
step-aligned comparison, but both anneal to zero learning rate and the longer run
is worse: more epochs on this corpus hurt the dense model. That is the
capacity-bound precondition the ORRG test needs.

## ORRG arms (in flight)

`orrg_all` started immediately after the dense endpoint; `orrg_orth_mag` is
queued behind it. Expected wall clock is ~2.5 h and ~2.8 h respectively. Their
JSONs land at:

* `experiments/results/p2_epochs_orrg_all.json`
* `experiments/results/p2_epochs_orrg_orth_mag.json`

The answer to "does residual capacity pay once dense saturates" is the comparison
of those endpoints against 3.0171; the Result section below records it as the arm
files land.

## Checkpoint/resume verification

The 2,660-step arms are long enough that a crash must not silently restart them,
so checkpoints were validated on the live run, not only in unit tests:

* `orrg_all` wrote `experiments/results/ckpt_orrg_all.pt` at step 250 (2.7 GB:
  1,602 trainable tensors, Adafactor state, OneCycle state, torch and CUDA RNG
  states).
* The arm was then killed by PID and relaunched with `--resume --ckpt-every 250`.
  The log prints `resuming from step 250/2660`, and the next trace (step 300) is
  `val=2.9042`, continuing the pre-kill trend (2.8091 at 100, 2.8656 at 200).
* A separate-process test outside the live run (tiny model, interrupted at step 3
  of 6, resumed in a fresh interpreter) reproduces the uninterrupted final
  parameters with `max_abs_diff 0.0`.
* The first live resume attempt exposed a real bug: `map_location=device` moved
  the saved CPU RNG byte states to CUDA and `torch.set_rng_state` rejected them
  (`TypeError: RNG state must be a torch.ByteTensor`). Fixed by normalizing both
  RNG states to CPU on load (`055dd93`); the resumed arm runs on the fixed code.
* A resumed arm skips the fresh-model `trace 0`, so its trace starts at the
  checkpoint step rather than reporting an untrained step 0.

## Result: two-epoch comparison

<!-- P2-RESULT:START -->
_Pending: waiting for `p2_epochs_orrg_all.json` and `p2_epochs_orrg_orth_mag.json`._
<!-- P2-RESULT:END -->

## Reproduce

```bash
PYTHONPATH=. python experiments/p1_real.py --steps 2660 --seeds 1 --batch 4 --seq 256 \
    --shared 2 --groups 4 --variants 3 --optimizer adafactor \
    --trace-every 100 --log-every 100 --down-init-scale 0.05 \
    --orth-step 3e-5 --orth-margin 0.5 --arms orrg_all \
    --out experiments/results/p2_epochs_orrg_all.json
```

## Threats

* **One seed**, and a 0.20-nat endpoint gap is large enough that this is not the
  binding limitation; the ORRG comparison at 0.01-nat resolution will be.
* **Different schedule length** from the 0.6-epoch rung; compare endpoints, not
  intermediate steps.
* **Repeated data.** Two epochs of the same 1.33M tokens measure overfitting, not
  additional data diversity.
* **The central comparison is still open**; nothing here says whether ORRG pays.
