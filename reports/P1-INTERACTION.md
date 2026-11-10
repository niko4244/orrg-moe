# P1 interaction rung: direction and magnitude together

The last two rungs each moved one axis of the residual problem, and each left
validation loss alone: an explicit push raised diversity 121× without changing
residual share, and a scaled init raised residual share 2.6× without changing
diversity. This rung runs both at once, to test whether the axes interact —
whether differentiated residuals let the router use more magnitude, or whether
nonzero residuals differentiate faster.

## Design

`orrg_orth_mag` combines the two levers:

| lever | setting |
|---|---|
| magnitude | `residual_init="scaled"`, `down_init_scale=0.05` |
| symmetry breaking | `residual_noise=0.02` |
| differentiation push | `orth_step=3e-5`, `orth_margin=0.5` |

The jitter is not optional here. The scaled init multiplies a copy of the shard's
down-projection, but the up-projections remain exact copies, so the cosine hinge
is still stationary at step 0. `residual_noise` is what gives the push a gradient,
exactly as in the orth arm; conversion KL is 1.3e-3, dominated by the scaled init
rather than by the 2% jitter.

Distillation is not included: it was inert at τ = 1 and destabilising at τ = 0.25.

## Protocol

Same model, corpus, shape, seed, schedule and step budget as the previous rungs,
so the three single-axis arms are reused rather than rerun.

| item | value |
|---|---|
| seed model | `HuggingFaceTB/SmolLM2-360M` (361.8M, 32 layers, hidden 960, intermediate 2560) |
| corpus | WikiText-103 raw, 1.33M train / 70k validation tokens |
| budget | 800 steps × 4 × 256 = **819k tokens**, 1 seed |
| optimizer | Adafactor, OneCycle over 800 steps, lr 2e-4, bf16 autocast |
| shape | `shared=2, groups=4, variants=3, k=1` → 676.7M total, 519.4M active |
| trace | every 100 steps: val loss, imbalance, diversity, residual share, cosine |

| arm | change relative to `orrg_all` |
|---|---|
| `orrg_all` | control: lossless zero init |
| `orrg_orth` | jitter + explicit push, zero init |
| `orrg_mag` | scaled init, no push |
| **`orrg_orth_mag`** | **scaled init + jitter + explicit push** |

## Results

| arm | val after | diversity | residual share | imbalance | dead | conv KL | s/step |
|---|---|---|---|---|---|---|---|
| ORRG all (control) | 2.8289 | 0.0004 | 4.18% | 0.203 | 0 | 3.9e-10 | 3.42 |
| ORRG orth | 2.8277 | **0.0486** | 4.20% | 0.211 | 0 | 3.9e-10 | 3.87 |
| ORRG mag | 2.8292 | 0.0004 | **10.88%** | 0.190 | 0 | 1.1e-03 | 3.88 |
| **ORRG orth+mag** | 2.8310 | **0.0484** | **10.91%** | 0.205 | 0 | 1.1e-03 | 3.82 |

Diversity and residual share, per 100 steps:

| step | 0 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 |
|---|---|---|---|---|---|---|---|---|---|
| diversity, orth+mag | 0.0004 | 0.0021 | 0.0052 | 0.0094 | 0.0148 | 0.0215 | 0.0293 | 0.0382 | **0.0484** |
| diversity, orth | 0.0004 | 0.0022 | 0.0052 | 0.0095 | 0.0149 | 0.0216 | 0.0294 | 0.0384 | 0.0486 |
| share, orth+mag | 10.02% | 10.36% | 10.63% | 10.78% | 10.84% | 10.90% | 10.94% | 10.91% | **10.91%** |
| share, mag | 9.98% | 10.34% | 10.57% | 10.73% | 10.83% | 10.83% | 10.87% | 10.87% | 10.88% |

Validation loss, per 100 steps:

| step | 0 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 |
|---|---|---|---|---|---|---|---|---|---|
| orth+mag | 3.0314 | 2.8349 | 2.8622 | 2.8752 | 2.8613 | 2.8449 | 2.8388 | 2.8295 | 2.8310 |
| orth | 3.0258 | 2.8359 | 2.8599 | 2.8725 | 2.8599 | 2.8421 | 2.8357 | 2.8283 | 2.8277 |
| mag | 3.0312 | 2.8352 | 2.8590 | 2.8732 | 2.8599 | 2.8433 | 2.8373 | 2.8289 | 2.8292 |

## What the combination did

**The levers do not interact.** Diversity lands at 0.0484 — 99.6% of the orth
arm's 0.0486 — and residual share lands at 10.91% — 100.3% of the mag arm's
10.88%. Each lever moves its own axis and leaves the other exactly where its
single-axis parent put it. That rules out both plausible interactions: residual
differentiation does not make the router use more of the residuals, and nonzero
residual function does not make the differentiation push more effective.

**The composition costs a little and the cost does not resolve.** Final val
2.8310 is 0.0021 worse than the control, 0.0033 worse than orth and 0.0018 worse
than mag. The combined arm also starts 0.0056 nats behind (3.0314 vs 3.0258)
because of the conversion, and its improvement from its own start is −0.2004 — not
larger than mag's −0.2020. With one seed, a 0.002-nat difference is not
statistically resolvable; the honest reading is "no measurable benefit, possibly
a small cost."

**The combination is stable.** Zero dead experts, entropy 0.998, imbalance 0.205,
unlike the sharp-target arm (0.468 imbalance, 34 dead experts). Differentiating
and magnifying residuals at the same time does not destabilise routing.

## Interpretation

The two levers are independent controls on two independent properties of the
residual block. A deployment that wants either property can have it, or both,
without interference — and none of those choices changes validation loss at this
scale and budget. Together with the sharp-target result, the picture is now
consistent: the router's balance, the residuals' direction, the residuals'
magnitude and the target's sharpness can all be moved substantially, and only a
change that destabilises routing (a sharp target) moves loss — downward.

The capacity question is therefore closed at 819k tokens, not because residuals
cannot be made to do more, but because the model does not need them to. The next
test that can distinguish these recipes is one where the dense baseline is
capacity-bound: multiple epochs, a larger seed, or a task where data rather than
parameters is the constraint.

## Threats to validity

* **One seed.** Interaction claims are differences of differences from single
  runs; only large effects are interpretable.
* **The push rate was calibrated in the zero-init regime** (3e-5). With nonzero
  residuals the same rate may be too weak or too strong; this is one operating
  point, not a tuned combination.
* **The additive expectation is a reference, not a statistical test.** The
  interesting question is whether the combined arm is far outside what either
  single axis produced, not whether it differs by 0.001 nats.
* **819k tokens ≈ 0.6 epoch** of the 1.33M-token train split; validation loss
  bottoms near step 300 and then rises.
* **`k=1` of two residuals** and 76.8% active width: capacity growth, not sparse
  compute.

## Reproduce

```bash
PYTHONPATH=. python experiments/p1_real.py --steps 800 --seeds 1 --batch 4 --seq 256 \
    --shared 2 --groups 4 --variants 3 --optimizer adafactor \
    --trace-every 100 --log-every 50 --down-init-scale 0.05 \
    --orth-step 3e-5 --orth-margin 0.5 --arms orrg_orth_mag \
    --out experiments/results/p1_long_orrg_orth_mag.json
```

Single-axis arms: `reports/P1-ORTHO.md` and `reports/P1-MAGNITUDE.md`.
