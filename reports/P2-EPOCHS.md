# P2 two-epoch rung: dense overfits early, ORRG arms in flight

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
started: 3.0171 after 2.72M tokens against 3.0257 before training. Its
validation minimum is **2.8039 at step 100**: the run gains 0.2218 nats in the
first 100 steps and gives back 0.2132 over the remaining 2,560. Compare its
0.6-epoch endpoint on the shorter schedule, 2.8204 — the extra 1.4 epochs cost
about 0.20 nats. The schedule differs between the two runs, so this is not a
step-aligned comparison, but both anneal to zero learning rate and the longer run
is worse: more epochs on this corpus hurt the dense model. That is **not** the
capacity-bound precondition the ORRG test needs: it is what a data-bound,
overparameterized model looks like when it memorizes 1.33M repeated tokens.

## The endpoint is a degradation regime, not a capacity test

Two properties of this run decide how the 2,660-step endpoint may be read.

**The minimum sits at peak learning rate.** OneCycle warms up over the first 3%
of steps, peaks at `lr=2e-4` around step 80, and anneals to ~0 at 2,660:

| step | 50 | 100 | 400 | 800 | 1,300 | 1,700 | 2,100 | 2,500 | 2,600 |
|---|---|---|---|---|---|---|---|---|---|
| lr | 1.4e-4 | 2.0e-4 | 1.9e-4 | 1.6e-4 | 1.1e-4 | 6.1e-5 | 2.2e-5 | 1.9e-6 | 2.7e-7 |

The validation minimum at step 100 therefore coincides with the *peak* of the
schedule, and the 0.21-nat climb to the endpoint happens while LR falls by three
orders of magnitude. This is not an instability artifact that a lower LR would
remove: the degradation rate broadly tracks LR (dense gives up 0.121 nats from
step 100 to 600 but only 0.021 from 2,100 to 2,600), so the schedule sets how
fast the model moves into the worse state, not the direction. Annealing to zero
locks that state in rather than recovering the minimum.

**The data repeats, and the train/val gap opens right after the minimum.**
2,660 steps × 1,024 tokens = 2.72M tokens drawn with replacement from 1.33M
unique train tokens, i.e. 2.05 epochs. Dense training loss (100-step mean) and
validation:

| step | 100 | 200 | 400 | 900 | 1,300 | 1,700 | 2,100 | 2,600 |
|---|---|---|---|---|---|---|---|---|
| train | 2.785 | 2.653 | 2.539 | 2.234 | 2.055 | 1.856 | 1.745 | 1.725 |
| val | 2.804 | 2.858 | 2.893 | 2.945 | 2.943 | 2.996 | 2.996 | 3.017 |
| gap | 0.019 | 0.205 | 0.355 | 0.711 | 0.888 | 1.140 | 1.251 | 1.293 |

At step 100 train and validation still agree; afterwards the model fits the
sampled windows while held-out loss rises. `orrg_all` traces the same arc (train
1.708 vs val 3.009 at step 2,100). That is memorization of 1.33M tokens by a
362M-parameter model, not a model running out of capacity.

**Consequence for the comparison.** Dense improves 0.2218 nats from step 0 to
its minimum and then gives back 0.2132 to the endpoint. Through step 2,100,
`orrg_all` bottoms at 2.8091 (step 100) and has given back 0.2003. Across the 21
traced steps from 100 to 2,100, dense is better at every one, by 0.005–0.022
nats (mean +0.0153). Both curves are still on this trajectory, so the 2,660-step
comparison is a small residual — observed gaps are 0.005–0.022 nats — on top of
a ~0.20–0.21-nat drift. That compares how much validation loss each arm gives up
while memorizing the corpus — a stability/interference difference — not how much
representational capacity it adds. A late `orrg_all` win, if any, would be a
reversal in the near-zero-LR tail after both models have already memorized the
data, and should be reported as such.

**What would test capacity.** Keep the steps and schedule but draw 2.72M
*unique* tokens (load ≥13M characters), so the run is one epoch. That gives the
residual experts the full schedule to grow (residual share is 1.9% at step 100
and 6.8% at step 2,100) without the repeated-data memorization that dominates
this endpoint. Early stopping at step 100 would not help: the experts have
barely engaged there, so it would compare two nearly dense models. Any
between-arm sign at 0.005–0.02 nats also needs ≥3 seeds.

## ORRG arms (in flight)

`orrg_all` started immediately after the dense endpoint; `orrg_orth_mag` is
queued behind it. At the observed rate each arm takes roughly six hours, so the
second JSON lands overnight. Their JSONs land at:

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

Compared against the dense two-epoch endpoint of **3.0171** at matched steps, schedule and data (2,660 steps x batch 4 x seq 256 = 2.72M tokens).

| arm | val before | val after | Δ vs dense (neg = better) | dead experts | imbalance | diversity | residual share | orth | s/step |
|---|---|---|---|---|---|---|---|---|---|
| dense | 3.0257 | 3.0171 | – | – | – | – | – | – | 0.46 |
| orrg_all | 3.0258 | 3.0370 | +0.0199 | 14 | 0.423 | 0.0014 | 0.0669 | 0.2486 | 8.21 |
| orrg_orth_mag | 3.0314 | 3.0354 | +0.0182 | 0 | 0.367 | 0.4308 | 0.1188 | 0.0081 | 3.79 |

**Verdict.** Neither arm beats dense (3.0171): `orrg_all` +0.0199, `orrg_orth_mag` +0.0182. Residual capacity still does not pay, so the extra parameters buy nothing at this data budget. Read the endpoint as a late-training regime, not as either model's best: dense bottoms out at 2.8039 (step 100) and gives back 0.2132 nats by the endpoint; the arms' minima are `orrg_all` 2.9042 (step 300), `orrg_orth_mag` 2.8095 (step 100). With one seed and a 0.005-nat margin, the sign is the result and the size is indicative.

| step | dense | orrg_all | orrg_orth_mag |
|---|---|---|---|
| 0 | 3.0257 | – | 3.0314 |
| 100 | 2.8039 | – | 2.8095 |
| 200 | 2.8575 | – | 2.8667 |
| 300 | 2.8864 | 2.9042 | 2.9029 |
| 400 | 2.8932 | 2.9063 | 2.9045 |
| 500 | 2.9042 | 2.9170 | 2.9164 |
| 600 | 2.9248 | 2.9400 | 2.9380 |
| 700 | 2.9179 | 2.9366 | 2.9273 |
| 800 | 2.9324 | 2.9529 | 2.9469 |
| 900 | 2.9448 | 2.9630 | 2.9570 |
| 1000 | 2.9453 | 2.9605 | 2.9569 |
| 1100 | 2.9404 | 2.9590 | 2.9536 |
| 1200 | 2.9879 | 3.0101 | 3.0046 |
| 1300 | 2.9428 | 2.9585 | 2.9568 |
| 1400 | 2.9547 | 2.9689 | 2.9658 |
| 1500 | 2.9709 | 2.9858 | 2.9913 |
| 1600 | 2.9764 | 2.9923 | 2.9921 |
| 1700 | 2.9960 | 3.0072 | 3.0100 |
| 1800 | 2.9807 | 2.9967 | 2.9962 |
| 1900 | 2.9986 | 3.0161 | 3.0146 |
| 2000 | 2.9887 | 3.0051 | 3.0066 |
| 2100 | 2.9959 | 3.0094 | 3.0121 |
| 2200 | 2.9995 | 3.0157 | 3.0149 |
| 2300 | 3.0198 | 3.0401 | 3.0400 |
| 2400 | 3.0181 | 3.0385 | 3.0364 |
| 2500 | 3.0217 | 3.0407 | 3.0396 |
| 2600 | 3.0173 | 3.0368 | 3.0349 |

Context: at the 819k-token rung `orrg_all` trailed dense by 0.0085 nats (2.8289 vs 2.8204). That rung ran a shorter total schedule, so the movement between rungs is indicative rather than a clean data-scaling curve.

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

* **One seed and 8,192 validation tokens.** Between-arm gaps are 0.005–0.022
  nats; one seed and one fixed validation slice make a 0.005-nat sign fragile.
  The 0.21-nat drift is large enough to see, but it is not the quantity the
  verdict is about.
* **Endpoint is post-minimum.** Dense is 0.2132 nats above its validation
  minimum at 2,660, and `orrg_all` is 0.2003 above its own at step 2,100, so the
  endpoint measures degradation under repeated data, not attainable loss or
  capacity (see above).
* **Different schedule length** from the 0.6-epoch rung; compare endpoints, not
  intermediate steps.
* **Repeated data.** Two epochs of the same 1.33M tokens measure overfitting, not
  additional data diversity.
* **The central comparison is still open**; nothing here says whether ORRG pays.
