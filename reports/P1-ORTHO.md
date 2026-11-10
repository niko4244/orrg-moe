# P1 differentiation rung: breaking the plateau, and what it did not fix

The previous long rung (`reports/P1-LONG.md`) ended with a clear diagnosis: over
819k tokens per arm, residual diversity sat at 0.0004, residual output share
plateaued at ~4.2%, and the contribution target stayed near-uniform
(CE 0.668 against ln 2 = 0.693). Router distillation and task-loss-only routing
were indistinguishable on loss.

This rung adds the two interventions that diagnosis called for — a differentiation
pressure on the residuals, and a sharper contribution target — and pits each
against the already-measured controls at the same budget.

## Protocol

Same model, corpus, shape, seed, schedule and step budget as `P1-LONG.md`, so the
controls are reused rather than rerun.

| item | value |
|---|---|
| seed model | `HuggingFaceTB/SmolLM2-360M` (361.8M, 32 layers, hidden 960, intermediate 2560) |
| corpus | WikiText-103 raw, 1.33M train / 70k validation tokens |
| budget | 800 steps × 4 × 256 = **819k tokens per arm**, 1 seed |
| optimizer | Adafactor, OneCycle over 800 steps, lr 2e-4, bf16 autocast |
| shape | `shared=2, groups=4, variants=3, k=1` → 676.7M total, 519.4M active |
| isolation | one OS process per arm |
| trace | every 100 steps: val loss, imbalance, diversity, residual share, orth, distill CE |

| arm | change relative to `orrg_all` |
|---|---|
| `orrg_all` | control: task-loss-only routing |
| `orrg_distill` | control: contribution-target distillation, τ = 1 |
| `orrg_orth` | `residual_noise=0.02` + explicit differentiation push, `orth_step=3e-5`, `orth_margin=0.5` |
| `orrg_sharp` | distillation with `distill_target_temp=0.25` (4× sharper) |

## Why the pressure needed two pieces

A hinge on the cosine between residual experts is stationary at exact copies:
its gradient is proportional to the sine of the angle, which is zero at
conversion. Measured: with an in-loss penalty at coefficient 1.0, cosine moved by
1e-4 in 80 steps — inert. Optimizer scaling compounds this: Adafactor normalises
per parameter, so multiplying the loss term does not translate into a usable
weight step.

Two things fix it:

1. **Symmetry-breaking init** (`residual_noise=0.02`): a 2%-of-std jitter on the
   residual up-projections. Because the down-projection is still exactly zero,
   the residual output is still exactly zero — conversion KL stays 3.9e-10 — but
   the hinge now has a gradient.
2. **Explicit normalised push** (`orth_step=3e-5`): the hinge gradient is applied
   directly, normalised to unit RMS per weight component, and stops once a pair's
   cosine falls below 0.5. This is independent of the optimizer.

The push acts only on up-projections, which carry no function until the
down-projection grows, so it is free early and a regulariser later.

## Results

Imbalance and diversity are the step-800 trace values; dead experts are the
end-of-run count on the eval batches. The end-of-run imbalance collection reads
0.203 / 0.208 / 0.211 / 0.468 for the four arms, i.e. the same ordering.

| arm | val after | Δ vs dense | diversity | residual share | imbalance | dead | distill CE | s/step |
|---|---|---|---|---|---|---|---|---|
| dense continued | 2.8204 | – | – | – | – | – | – | 0.45 |
| ORRG all (control) | 2.8289 | +0.0085 | 0.0004 | 4.18% | 0.183 | 0 | – | 3.42 |
| ORRG distill (τ = 1) | 2.8275 | +0.0071 | 0.0004 | 4.10% | 0.195 | 7 | 0.668 | 3.45 |
| **ORRG orth** | 2.8277 | +0.0073 | **0.0486** | 4.20% | 0.177 | 0 | – | 3.87 |
| **ORRG sharp** (τ = 0.25) | 2.8366 | +0.0162 | 0.0004 | 3.49% | **0.460** | **34** | **0.536** | 2.88 |

Diversity trajectory (both controls sit at 0.0004 from step 400 onward; the sharp
arm tracks them exactly):

| step | 0 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 |
|---|---|---|---|---|---|---|---|---|---|
| ORRG orth | 0.0004 | 0.0022 | 0.0052 | 0.0095 | 0.0149 | 0.0216 | 0.0294 | 0.0384 | **0.0486** |
| controls / sharp | 0.0000 | 0.0001 | 0.0002 | 0.0003 | 0.0004 | 0.0004 | 0.0004 | 0.0004 | 0.0004 |

For the orth arm the hinge falls 0.2496 → 0.2038, i.e. mean within-group cosine
goes from 0.9996 (copies plus 2% jitter) to **0.9515**. Diversity grows
monotonically and roughly linearly, with no sign of the plateau the controls hit.

Sharp-target trajectory (τ = 0.25), against the τ = 1 control at the same steps:

| step | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 |
|---|---|---|---|---|---|---|---|---|
| val, sharp | 2.8414 | 2.8764 | 2.8905 | 2.8699 | 2.8525 | 2.8471 | 2.8361 | 2.8366 |
| val, τ=1 | 2.8350 | 2.8617 | 2.8730 | 2.8588 | 2.8421 | 2.8357 | 2.8267 | 2.8275 |
| imbalance, sharp | 0.4274 | 0.4592 | 0.4754 | 0.4675 | 0.4714 | 0.4651 | 0.4633 | 0.4599 |
| imbalance, τ=1 | 0.2027 | 0.2028 | 0.1891 | 0.2041 | 0.1969 | 0.1935 | 0.1945 | 0.1954 |
| distill CE, sharp | 0.6349 | 0.5739 | 0.5541 | 0.5464 | 0.5450 | 0.5418 | 0.5344 | 0.5363 |
| distill CE, τ=1 | 0.6830 | 0.6773 | 0.6732 | 0.6704 | 0.6673 | 0.6667 | 0.6679 | 0.6678 |

Router entropy at the end of the run: 0.997 (control), 0.962 (τ = 1), 0.996
(orth), **0.787** (sharp) — the sharp arm is concentrating its choices.

## What each intervention did

**Differentiation is forced, not emerging.** The explicit push takes diversity
from 0.0004 to 0.0486 — 121×, monotone, and in the range the toy run reached
(0.028–0.056) rather than at a trivial value. It costs no loss (2.8277 vs 2.8289,
well inside one-seed noise), leaves 0 dead experts and entropy at 0.996, and adds
~13% wall clock (3.87 vs 3.42 s/step) from the extra backward through the cosine
graph.

**But direction is not the bottleneck.** Residual output share is unchanged:
4.20% against 4.18% for the control, following the same trajectory throughout
(2.34% at step 100, 4.15% at 500, 4.20% at 800). The residuals now point in
different directions and still contribute the same small amount of output norm.
Different downstream residuals that each write back ~4% do not buy capacity.

**A sharper target makes the signal real and the routing worse.** Forcing the
contribution CE from 0.668 to 0.536 (against the 0.6931 ceiling) does exactly what
it says, but it raises imbalance from 0.195 to 0.460, dead experts from 7 to 34,
lowers residual share from 4.10% to 3.49%, and costs 0.009 nats (2.8366 vs
2.8275). Diversity never moves. The norm differences being amplified are mostly
noise: residuals whose output norms differ by a fraction of a percent get a
near-one-hot target, and the router is pushed into over-committing to them.

## Interpretation

The plateau has (at least) two independent causes, and this rung separates them:

1. **Similarity of residual directions** — fixed by the differentiation pressure,
   at no loss cost.
2. **Magnitude of residual contribution** — untouched by either intervention, and
   the one that governs whether the extra parameters matter.

Sharpening the target attacks the wrong variable: it makes the router more
decisive about differences that are too small to be meaningful, which shows up as
collapse rather than specialization.

The next lever is not more differentiation. It is something that grows or
directly rewards contribution: a gate/scale on residual output, wider residuals
so a few percent of share is a larger absolute signal, or a small nonzero
down-projection init that trades exact losslessness for immediate function.

## Threats to validity

* **One seed.** The 0.0012 difference to the control and the 0.009 penalty of the
  sharp arm are not statistically resolved; the diversity and dead-expert results
  are large and structural.
* **One diversity definition** (cosine between the largest parameter of each
  residual, which for these gated FFNs is the gate projection). Functional
  specialization could differ from weight cosine.
* **The push rate was calibrated on a 60-step smoke**, not swept; 3e-5 was chosen
  to move cosine visibly without disturbing the loss, and a rate sweep might
  produce more diversity for the same share.
* **τ = 0.25 is one operating point.** A temperature sweep might find a region
  where the signal sharpens without collapse; nothing here rules that out.
* **819k tokens ≈ 0.6 epoch** of the 1.33M-token train split; validation loss
  bottoms around step 300 and then rises, so this is a mildly overfitting regime.
* **`k=1` of two residuals** and 76.8% active width: capacity growth, not sparse
  compute.
* The sharp arm's s/step (2.88) is lower than both controls; with one process per
  arm this is plausibly allocator variance, and it should not be read as a real
  speedup.

## Reproduce

```bash
for arm in orrg_orth orrg_sharp; do
  PYTHONPATH=. python experiments/p1_real.py --steps 800 --seeds 1 --batch 4 --seq 256 \
      --shared 2 --groups 4 --variants 3 --optimizer adafactor \
      --distill-coef 0.1 --distill-tokens 128 --trace-every 100 --log-every 50 \
      --orth-step 3e-5 --orth-margin 0.5 --residual-noise 0.02 \
      --distill-target-temp 0.25 --arms "$arm" \
      --out "experiments/results/p1_long_${arm}.json"
done
PYTHONPATH=. python experiments/p1_long_report.py
```

Wall clock: 52 min (`orrg_orth`), 38 min (`orrg_sharp`). Controls: `reports/P1-LONG.md`.
