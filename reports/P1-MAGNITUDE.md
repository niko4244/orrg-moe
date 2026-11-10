# P1 magnitude rung: residuals that act from step 0

The previous rung settled the direction question. An explicit differentiation
pressure took residual diversity from 0.0004 to 0.0486 at no loss cost — but
residual output share did not move (4.20% against 4.18% for the control). The
remaining plateau is magnitude: residuals are different, and still contribute
almost nothing.

This rung tests the magnitude lever directly.

## Why a gate or a scale could not work

With the lossless init the residual down-projection is exactly zero. That makes
the residual output zero, and it also makes the gradient with respect to the
residual up-projections zero, because that gradient is the down-projection
transposed times the upstream gradient. A learned gate or a fixed scale
multiplies the residual output, so it multiplies zero — inert for the same
structural reason the in-loss cosine penalty was inert at copies.

The lever therefore has to start with a nonzero down-projection. That is the
option this rung implements.

## Mechanism

`residual_init="scaled"` builds each residual as a copy of its base shard and
multiplies the down-projection (and output bias, where present) by
`down_init_scale = 0.05`. Each residual contributes 5% of its shard's function
from step 0, and its up-projections receive gradient from the first batch. From
there the down-projection is free to grow or shrink, and the router is trained
against residuals that actually produce output.

This is explicitly **not** lossless. The seed model's function changes at
conversion: measured conversion KL 1.1e-3 with 98.8% top-1 agreement, against
3.9e-10 for the zero init. That cost is the thing being traded for immediate
capacity, and it is reported next to every result rather than hidden.

## Protocol

Same model, corpus, shape, seed, schedule and step budget as the previous rungs,
so the control is reused rather than rerun.

| item | value |
|---|---|
| seed model | `HuggingFaceTB/SmolLM2-360M` (361.8M, 32 layers, hidden 960, intermediate 2560) |
| corpus | WikiText-103 raw, 1.33M train / 70k validation tokens |
| budget | 800 steps × 4 × 256 = **819k tokens**, 1 seed |
| optimizer | Adafactor, OneCycle over 800 steps, lr 2e-4, bf16 autocast |
| shape | `shared=2, groups=4, variants=3, k=1` → 676.7M total, 519.4M active |
| init | `residual_init="scaled"`, `down_init_scale=0.05` |
| trace | every 100 steps: val loss, imbalance, diversity, residual share |

| arm | change relative to `orrg_all` |
|---|---|
| `orrg_all` | control: lossless zero init |
| `orrg_mag` | scaled down-projection init, 5% of the shard's function at step 0 |

## Results

Imbalance and diversity are the step-800 trace values; dead experts are the
end-of-run count.

| arm | val after | Δ vs dense | diversity | residual share | imbalance | dead | conv KL | s/step |
|---|---|---|---|---|---|---|---|---|
| dense continued | 2.8204 | – | – | – | – | – | – | 0.45 |
| ORRG all (control) | 2.8289 | +0.0085 | 0.0004 | 4.18% | 0.203 | 0 | 3.9e-10 | 3.42 |
| **ORRG mag** | 2.8292 | +0.0088 | 0.0004 | **10.88%** | 0.190 | 0 | **1.1e-03** | 3.88 |

Residual output share, per 100 steps:

| step | 0 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 |
|---|---|---|---|---|---|---|---|---|---|
| ORRG mag | **9.98%** | 10.34% | 10.57% | 10.73% | 10.83% | 10.83% | 10.87% | 10.87% | **10.88%** |
| ORRG all | 0.00% | 2.33% | 3.23% | 3.73% | 4.00% | 4.14% | 4.17% | 4.18% | 4.18% |

Validation loss, per 100 steps:

| step | 0 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 |
|---|---|---|---|---|---|---|---|---|---|
| ORRG mag | 3.0312 | 2.8352 | 2.8590 | 2.8732 | 2.8599 | 2.8433 | 2.8373 | 2.8289 | 2.8292 |
| ORRG all | 3.0258 | 2.8344 | 2.8596 | 2.8722 | 2.8590 | 2.8424 | 2.8366 | 2.8283 | 2.8289 |

## What the lever did

**The 4% plateau is broken — by the initialization, not by training.** Share
starts at 9.98% and ends at 10.88%: 2.6× the control, with only +0.9 pp of growth
over 819k tokens and a flat second half. The zero-init arm's 4.18% is therefore
not a ceiling imposed by the optimizer or the architecture; it is the distance
covered in 819k tokens when the residuals start with zero function. Start them at
5% of their shard's function and they settle near 11%.

**More share did not buy loss.** 2.8292 against the control's 2.8289 (+0.0003,
one seed) is a tie, even though residuals contribute 2.6× more output. The arm
also had to recover a 0.0054-nat conversion deficit first (3.0312 vs 3.0258 at
step 0); it was within 0.0005 of the control from step 100 onward. At 819k tokens
the model is not short of capacity magnitude either.

**Diversity is unchanged** (0.00037 vs 0.0004), which confirms the two axes are
independent: the previous rung moved direction without magnitude, this one moves
magnitude without direction.

**The exactness trade is small and recoverable.** Conversion KL 1.1e-3, top-1
agreement 98.8%, largest relative logit shift 15%. One seed cannot resolve whether
the final +0.0003 is real, but the step-0 deficit is gone by step 100.

## Interpretation

Together the two rungs close the diagnosis. Residuals can be made different
(orthogonality rung) or made to contribute 2.6× more (this rung), and the 360M
model's validation loss does not respond to either at this budget. The 4% figure
was an artifact of starting from zero, not a resource limit; there is capacity to
spare at 0.6 epoch, and adding differentiable capacity does not help.

That also demotes the losslessness requirement as an explanation: giving up exact
conversion bought nothing here. The next tests should ask where capacity can
pay — longer training where the dense baseline saturates, a larger seed model, or
tasks where capacity rather than data is the constraint. Whether direction and
magnitude interact (a combined `orth` + `mag` arm) is untested, but on this
evidence it is not the highest-value next run.

## Threats to validity

* **One seed.** Loss differences inside ~0.005 nats are not resolved.
* **The conversion is no longer lossless** (KL 1.3e-3). Val loss at step 0 is
  3.0312 against the control's 3.0258, so the arm starts 0.005 nats behind and
  every later comparison includes that debt.
* **`down_init_scale` is one operating point** (0.05), chosen so residuals start
  near the control's plateau rather than above it a priori; the observed start
  is ~10% share, not the ~4% the estimate suggested.
* **One diversity definition** (cosine between the largest residual parameter),
  and `k=1` of two residuals.
* **819k tokens ≈ 0.6 epoch** of the 1.33M-token train split; validation loss
  bottoms near step 300 and then rises, so this is a mildly overfitting regime.

## Reproduce

```bash
PYTHONPATH=. python experiments/p1_real.py --steps 800 --seeds 1 --batch 4 --seq 256 \
    --shared 2 --groups 4 --variants 3 --optimizer adafactor \
    --trace-every 100 --log-every 50 --down-init-scale 0.05 --arms orrg_mag \
    --out experiments/results/p1_long_orrg_mag.json
```

Control: `reports/P1-LONG.md` (`orrg_all`).
