# ORRG-MoE Spec v3

**Oracle-free, lossless dense → MoE conversion, and the protocol to take it to 27B.**

Status: v3 is the first version backed by running code and measured numbers.
Everything in §4 and §5 was produced in this repository. §7 is the scale-up plan;
nothing there has been run at 9B or 27B.

---

## 0. What changed from v2, and why

v2 proposed: carve a skeleton, add zero-output residual experts, force variant 0
while routing is released in phases, and train the router on oracle labels.
Measurement forced four corrections:

1. **The v2 shard layout cannot be lossless.** If variants are allocated
   *disjoint* neuron ranges (`N = shared + groups × variants`), the always-on
   skeleton covers only a fraction of the intermediate axis. v3 shards the
   skeleton into `shared + groups` ranges that tile it exactly, and makes the
   other variants **copies of their group's range**. Measured carve error is now
   2.5e-07–4.2e-07 instead of 0.9.
2. **Residuals must be additive, not a replacement.** v2's "top-1 per group,
   forced variant 0" is lossless only while the scaffold is pinned; selecting a
   zero-output variant deletes that group's contribution. v3 computes
   `group(x) = base(x) + Σ selected_residual(x)`. Losslessness then holds for
   *any* router state, including fully released routing.
3. **Oracle labels do not exist at conversion.** Every expert is identical or
   zero, so an "oracle contribution" target is degenerate. v3 defines a
   measurable substitute (§3.5): the normalized output norm of each residual,
   which starts uniform and becomes informative as experts grow.
4. **The frozen-skeleton recipe is not automatically better.** At toy scale it
   matches naive upcycling rather than beating it (§5). The lossless start pays
   off when the whole model is co-trained.

---

## 1. Targets, and the geometry that actually works

Notation: `N₀` = seed parameters, `N_nonFFN` = embeddings + attention + norms,
`F` = FFN parameters, `D` = intermediate width, `h` = hidden width.

### 9B-A2B ("2B active, 9B total")

A 9B dense model in a Llama-3-8B-like shape has `N_nonFFN ≈ 1.87B` and
`F ≈ 5.64B`. Carving to 2B active leaves **≤ 0.13B for the FFN (2.3 % of it)**.
That is not a MoE; that is deleting the FFN. **9B-A2B is therefore only
reachable by growth**, from a ~2B seed:

| seed shape (h / inter / L / vocab) | seed | total | active k=0 | active k=1 | expansion |
|---|---|---|---|---|---|
| 2560 / 6912 / 24 / 128k | 1.98B | 8.67B | 1.98B | 2.93B | 4.38× |

This is the honest version of the v2 claim: a 2B dense → ~9B-A2B conversion is
feasible **as capacity growth**, and the "active" count is a serving choice, not
a property of the checkpoint.

### 27B-A6B

| shape (h / inter / L / vocab / heads / kv) | seed | total | active k=0 | active k=1 | active k=2 |
|---|---|---|---|---|---|
| 4096 / 10240 / 32 / 128k / 32 / 8 | 5.89B | 27.03B | 5.89B | 8.91B | 11.93B |

Expansion 4.59×, full-finetune state 432 GB, 50B tokens at 25 % MFU ≈ 3.0k
H100-hours at k=1 (2.0k at k=0). This is the recommended first large run because
the seed is a single dense checkpoint and every conversion step is checkable
before any training starts.

---

## 2. Formal setup

### 2.1 The FFN is additive in its hidden neurons

For any per-neuron activation `σ` (GELU, SiLU-gated, squared-ReLU, …):

```
FFN(x) = σ(x W_upᵀ + b_up) W_downᵀ + b_down
       = Σᵣ σ(x W_up[r]ᵀ + b_up[r]) W_down[:, r]ᵀ + b_down
```

**Definition 1 (skeleton).** Ranges `R₀ … R_{S-1}` partition `[0, D)`.
`S = shared + groups`. Shard `i` is the dense block with every up-projection
restricted to `Rᵢ` and every down-projection restricted to the same range.

**Proposition 1 (losslessness).** `Σᵢ shardᵢ(x) = FFN(x)` for all `x`, up to
floating-point summation order. Verified: 2.5e-07 – 4.2e-07 relative error on
five architectures, ≤ 1.2e-07 on logits (Llama, GPT-2, Phi, Mistral, Qwen2).

### 2.2 Residual experts

Variant `v ≥ 1` of group `g` is a copy of shard `shared + g` with
`W_down ← 0` and `b_down ← 0`. Its output is exactly zero at t=0, regardless of
input.

**Proposition 2 (conversion identity).** With `group(x) = base(x) + Σⱼ wⱼ Eⱼ(x)`
and `Eⱼ ≡ 0`, the converted model equals the seed for every router
configuration, including uniform, random, and fully released routing. Measured
conversion KL: **0.0** (naive upcycling: **0.141**) over two seeds.

**Proposition 3 (router bootstrap).** At step 0, `∂L/∂W_router = 0` because every
`Eⱼ(x) = 0`, but `∂L/∂W_down,ⱼ ∝ wⱼ · hᵀ ≠ 0`. Experts therefore leave zero on
the first optimizer step and the router becomes trainable on the second.
Verified: residual gradient 1.2e-04 – 2.9e-02, router gradient exactly 0 at init,
nonzero after one step.

### 2.3 Grouped routing is structurally balanced

**Proposition 4.** With `groups` groups and `k` selections per group per token,
per-group token load is exactly `k · N`, independent of router parameters.
Measured: per-group counts equal `k · N` exactly (test asserts equality), dead
experts = 0.

Balance within a group is *not* structural, so `bias_update_rate > 0` enables
sign-based, aux-loss-free bias updates on the router.

### 2.4 Accounting

```
total   T   = N_nonFFN + S_params + R_params + router_params
active A(k) = N_nonFFN + S_params + router_params + k · groups · P_shard
P_shard     = F / (shared + groups)
```

`A(0)` equals the seed plus routers. This is asserted by a test, not estimated.

---

## 3. Method as implemented

### 3.1 Discovery (`orrg/ffn.py`)

A block is an FFN iff its parameters contain a projection whose output is the
intermediate axis and one whose input is that axis, and `intermediate > hidden`.
Attention is rejected by that inequality (GPT-2's fused qkv is wider, so names
are also filtered), and blocks that own non-FFN parameters are rejected
explicitly: OPT-style `fc1`/`fc2` at layer level fail with a message rather than
silently converting attention.

### 3.2 Sharding (`orrg/shard.py`)

Narrow materialized copies. Up-projection sliced on its output axis, down on its
input axis, up-bias sliced, output bias kept once. transformers `Conv1D`
(`nf`/`nx`) and `nn.Linear` are handled; the block's own forward is used, so
gated and non-gated, any activation, any dtype works.

### 3.3 Router (`orrg/router.py`)

`TokenRouter(hidden → groups × variants)` in fp32, per token. Grouped top-k with
renormalized weights, optional jitter, optional bias balancing, hard-assignment
API for analysis. It reads only the hidden state entering the block, which is why
it is architecture-free — the model-agnostic "tokenizer" for experts.

### 3.4 Training regimes

| regime | trainable | when |
|---|---|---|
| `orrg-all` | everything | default; measured best (§5) |
| `orrg-frozen` | residuals + routers | memory-constrained, seed must stay exact |
| `partition` | routers | compute reduction (MoEfication regime), approximate when sparse |

### 3.5 Router distillation (`orrg/distill.py`)

For a batch, per group:

```
targetⱼ = softmax(‖Eⱼ(x)‖ / τ),   L_router = −Σⱼ targetⱼ log p_router( j | x)
```

At t=0 the target is uniform (correct prior: no expert is better). As experts
grow, the target concentrates. This is a real, cheap signal — it needs no
teacher logits and no labels, only a forward pass through the residuals. Output
KD (forward/reverse) and on-policy KD with prompt masking are also implemented.

`τ < 1` sharpens the target. At `τ = 1` it stays near-uniform because residual
output norms are nearly identical (measured CE 0.668 against a 0.693 ceiling,
§5.1), so the router gets very little signal; `distill_target_temp` exists to
test whether sharpening the same statistic supplies one.

### 3.6 Metrics / gates

Conversion KL and relative max error per layer (forced and deployed), router
entropy, per-group utilization, load imbalance, dead experts, expert diversity
(1 − cosine of residual weights), residual output share, and wall-clock tokens/s.

### 3.7 Differentiation pressure (`residual_noise`, `orth_step`)

Residuals start as exact copies, and diversity is what turns them into a mixture.
Two pieces are needed to pressure them apart:

* **Symmetry-breaking init** (`residual_noise`). A cosine penalty is stationary at
exact alignment, so its gradient is exactly zero at the copy init — measured: coef
1.0 on an in-loss penalty moved cosine by 1e-4 in 80 steps. A small relative
jitter on the residual up-projections makes the hinge differentiable. Because the
down-projection is still zero, residual output is still exactly 0 and conversion
stays lossless (§4).
* **Explicit normalised push** (`orth_step`). A penalty inside the loss is also too
weak under Adafactor's per-parameter normalisation. The push takes the hinge
gradient directly and moves `orth_step` RMS per weight component per step, and
stops once a pair's cosine falls below `orth_margin`.

Both act only on up-projections, which carry no function until the down-projection
grows, so the pressure is free early and a regulariser later. Both are off by
default.

## 4. Evidence: conversion is lossless

`PYTHONPATH=. python -m pytest tests/test_orrg.py -q` → **34 passed**.

| check | result |
|---|---|
| carve error, 5 architectures | 2.5e-07 – 4.2e-07 |
| carve error, activation-clustered shards, real 2560-wide FFN | 1.3e-05 |
| conversion KL, real checkpoint, contiguous / clustered | 3.9e-10 / 3.3e-09 |
| conversion KL with symmetry-breaking residual jitter | 3.9e-10 |
| conversion KL with scaled down-projection init (`down_init_scale=0.05`) | 1.1e-03 |
| clustered cohesion vs random partition of the same sizes | 0.225 – 0.236 vs 0.044 – 0.065 |
| logit max abs diff | ≤ 1.2e-07 |
| partition mode, all shards active | ≤ 1.8e-07 |
| partition mode, 2 of 4 shards | not exact (asserted), by construction |
| residual output at init | exactly 0 |
| residual gradient at step 0 | nonzero (1.2e-04 – 2.9e-02) |
| per-group token counts | exactly `k · N` |
| checkpoint round-trip | bit-identical |
| active-parameter accounting | exact, including routers |

---

## 5. Evidence: P1 rung on a real dense checkpoint

First non-toy rung, and the one that tests the conversion claim on real weights
and real text. Seed `HuggingFaceTB/SmolLM2-360M` (361.8M, 32 layers, hidden 960,
intermediate 2560), WikiText-103, batch 4 × 256 tokens, **80 steps per arm**,
2 seeds, Adafactor. Shape `shared=2, groups=4, variants=3, k=1` → 676.7M total,
519.4M active (76.8% of width; capacity growth, not compute saving).

This is a rung, not a training run: 82k tokens per arm bounds conversion cost and
the first 80 steps only. Full write-up: `reports/P1-RUNG.md`.

| arm | val before | val after | Δ vs dense | conv KL | trainable | imbalance |
|---|---|---|---|---|---|---|
| dense continued | 3.0257 | 2.7473 ± 0.0007 | – | – | 361.8M | – |
| naive upcycling | 3.4359 ± 0.0040 | 2.8490 ± 0.0007 | **+0.1017 ± 0.0015** | 3.1e-01 | 550.8M | 0.212 |
| ORRG frozen | 3.0258 | 2.7573 ± 0.0018 | +0.0101 ± 0.0010 | 3.9e-10 | 314.9M | 0.457 |
| ORRG all | 3.0258 | 2.7488 ± 0.0008 | **+0.0016 ± 0.0015** | 3.9e-10 | 676.7M | 0.250 |
| ORRG clustered | 3.0257 | 2.7488 ± 0.0004 | +0.0015 ± 0.0003 | 3.3e-09 | 676.5M | 0.244 |
| ORRG distill | 3.0258 | 2.7482 ± 0.0000 | +0.0010 ± 0.0007 | 3.9e-10 | 676.7M | **0.177** |

Findings:

* **Lossless start holds on real weights.** 3.9e-10 KL and a 1e-4 difference in
  starting validation loss, against 0.31 KL and **+0.41 nats** for naive
  upcycling on the same batches.
* **Naive upcycling does not recover within the rung.** It ends +0.1017, 66× the
  seed spread; `orrg_all` ends +0.0016, inside it. Lossless conversion plus
  co-training reproduces dense continued training here; copy-upcycling does not.
* **Freezing costs ~6× more than co-training** (+0.0101 vs +0.0016) while
  training 47% of the parameters. Frozen is a memory trade, not a quality win —
  the toy result (§5.2) reproduces on real weights.
* **Activation clustering is free at this scale.** 3.3e-09 KL, cohesion 0.225 vs
  a 0.065 random baseline, and a paired end-loss difference of 0.0001 ± 0.0006
  against contiguous. It buys the option to specialize co-activated neuron sets;
  at equal steps it costs nothing measurable.
* **Router distillation moves routing, not yet loss.** Imbalance falls 0.250 →
  0.177 (29%, ~10× seed spread), but final loss differs by 0.0006 — inside the
  spread. The contribution target is real supervision; it is not yet a quality
  win at this horizon.
* **Diversity is structurally 0 here.** Residual up-projections only receive
  gradient through the zeroed down-projection, so diversity starts at exactly 0
  and needs hundreds of steps to move (toy: 0.03–0.06 at 300 steps). 0.0000 at
  80 steps is not collapse; no dead experts appeared.

### 5.1 Longer rung (819k tokens per arm)

Same model and shape, 800 steps × 4 × 256 = 819k tokens per arm (10× the rung
above), one seed, one process per arm, trajectory recorded every 100 steps. Arms:
dense, `orrg_all`, and `orrg_distill` (`coef=0.1`, 128 target tokens). Full
write-up: `reports/P1-LONG.md`.

| arm | val after | Δ vs dense | imbalance | diversity | residual share | distill CE | s/step |
|---|---|---|---|---|---|---|---|
| dense continued | 2.8204 | – | – | – | – | – | 0.45 |
| ORRG all | 2.8289 | +0.0085 | 0.183 | 0.0004 | 4.18% | – | 3.42 |
| ORRG distill | 2.8275 | +0.0071 | 0.195 | 0.0004 | 4.10% | 0.668 | 3.45 |

Over steps 0 → 800 both MoE arms move identically: diversity 0.0000 → 0.0004,
residual share 0.0% → 4.2%; distill CE 0.683 → 0.668 against ln 2 = 0.693.

Findings:

* **Neither mechanism separates.** Distill beats task-loss-only by 0.0014 nats
  (one seed) and both sit 0.007–0.009 behind dense.
* **The 80-step balance advantage does not persist.** At 80 steps distill's
  imbalance was 0.177 vs 0.250; at 800 it is 0.195 vs 0.183 on the training
  forward and 0.208 vs 0.203 at end of run, with 7 dead experts vs 0.
* **The extra capacity stays mostly inert.** Residual share plateaus at ~4% of
  the layer output and diversity at 0.0004; residuals remain small, similar
  corrections instead of specializing.
* **The distill target stays near-uniform.** Its CE moves from 0.683 to 0.668
  against a 0.693 ceiling, because residual output norms are too similar for the
  contribution target to sharpen. That is why the router gets little signal.
* **The 80-step absolute losses were schedule-defined.** Dense ends at 2.7480
  after 80 steps but 2.8204 after 800: the short rung annealed lr to ~0 inside 80
  steps, while 819k tokens on a 1.33M-token train split begins to overfit. Paired
  comparisons at matched steps stand; absolute losses do not transfer across
  rungs.

### 5.2 Toy-scale head-to-head (method smoke test)

Protocol: one dense 0.67M-parameter byte-level Llama seed, 3 synthetic domains
(prose / code / arithmetic), 400 seed steps, then 300 steps per arm on identical
batches, 2 seeds. `naive_upcycle` and `orrg_frozen` freeze non-MoE weights;
`orrg_all` and dense continue training everything. **This is a method smoke
test, not evidence about 9B models.**

| arm | val loss | prose | code | math | conversion KL | trainable | imbalance | dead | diversity |
|---|---|---|---|---|---|---|---|---|---|
| dense continued | 1.0142 | 1.0201 | 0.5959 | 1.4265 | – | 0.67M | – | – | – |
| naive upcycling | 1.1625 | 1.2344 | 0.7458 | 1.5072 | 1.41e-01 | 1.15M | 0.132 | 0 | 0.028 |
| ORRG, frozen skeleton | 1.1668 | 1.2405 | 0.7501 | 1.5099 | 0.00 | 0.71M | 0.221 | 0 | 0.056 |
| ORRG, all trainable | **1.0098** | 1.0137 | 0.5954 | 1.4204 | 0.00 | 1.38M | 0.162 | 0 | 0.041 |

Findings:

* **Lossless start is real and free.** 0.000 vs 0.141 conversion KL, with no
  recovery step and no teacher.
* **Residuals specialize faster than copies.** Diversity 0.056 vs 0.028 after
  the same number of steps; zero-init experts are free to move in any direction,
  copied experts start at a full-FFN optimum.
* **The gain appears with co-training, not with freezing.** `orrg_all` beats
  naive upcycling by 0.153 nats/byte and dense continued training by 0.004.
  `orrg_frozen` is *not* better than naive upcycling (+0.004, within seed
  spread), so the v2 "freeze the skeleton" recipe is not supported at this scale.
* **Caveat that matters:** `orrg_all` has 2.1× the trainable parameters of the
  dense arm. The comparison shows the lossless conversion does not cost quality
  and that the extra capacity is usable; it does not isolate a method effect.
* No dead experts in any arm; within-group imbalance is higher for ORRG, which
  is expected with zero-init residuals and no bias updates.

Reproduce: `PYTHONPATH=. python experiments/toy_scale.py --steps 300 --seed-steps 400 --seeds 2`
(builds `experiments/results/toy_scale.json`, 200 s on an RTX 5060 Ti).

---

### 5.3 Differentiation pressure and a sharper target (819k tokens/arm)

The rung above ended with residual diversity at 0.0004, residual share at
~4.2% and the contribution target near-uniform (CE 0.668 against ln 2 = 0.693).
This rung adds the two interventions that diagnosis implied, against the same
controls (same seed, schedule, data):

| arm | val after | diversity | residual share | imbalance | dead | distill CE | s/step |
|---|---|---|---|---|---|---|---|
| ORRG all (control) | 2.8289 | 0.0004 | 4.18% | 0.183 | 0 | – | 3.42 |
| ORRG distill, τ=1 | 2.8275 | 0.0004 | 4.10% | 0.195 | 7 | 0.668 | 3.45 |
| ORRG orth (`residual_noise=0.02`, `orth_step=3e-5`) | 2.8277 | **0.0486** | 4.20% | 0.177 | 0 | – | 3.87 |
| ORRG sharp (`distill_target_temp=0.25`) | 2.8366 | 0.0004 | 3.49% | **0.460** | **34** | 0.536 | 2.88 |

Findings:

* **Diversity is forced, not emerging.** The explicit push raises diversity
  0.0004 → 0.0486 (121×, monotone) with within-group cosine 0.9996 → 0.9515. A
  cosine hinge is stationary at copies, so this needs the `residual_noise` init;
  an in-loss penalty at coefficient 1.0 moved cosine 1e-4 in 80 steps.
* **Direction is not the bottleneck.** Residual output share is unchanged
  (4.20% vs 4.18%) and loss is unchanged (2.8277 vs 2.8289), with 0 dead experts.
  Different residuals that still contribute the same ~4% do not buy capacity.
* **A sharper target makes the signal real and the routing worse.** Forcing CE
  0.668 → 0.536 raises imbalance 0.195 → 0.460 and dead experts 7 → 34, lowers
  residual share 4.10% → 3.49%, and costs 0.009 nats. The norm differences being
  amplified are mostly noise at this scale, so the router over-commits.
* The plateau therefore has two independent causes — similar residual direction
  and small residual magnitude — and only the first is fixed by these levers.
  The untested lever is magnitude: a residual gate/scale, wider residuals, or a
  small nonzero down-projection init.

Full write-up: `reports/P1-ORTHO.md`.

### 5.4 Residual magnitude (819k tokens/arm)

The previous rung left magnitude as the open variable. A gate or a scale on the
residual output cannot supply it: with a zeroed down-projection both are
multiplied by zero, and so is the up-projection gradient. `residual_init="scaled"`
instead starts each residual at `down_init_scale` of its shard's function.

| arm | val after | diversity | residual share | imbalance | dead | conv KL | s/step |
|---|---|---|---|---|---|---|---|
| ORRG all (control) | 2.8289 | 0.0004 | 4.18% | 0.203 | 0 | 3.9e-10 | 3.42 |
| ORRG mag (`down_init_scale=0.05`) | 2.8292 | 0.0004 | **10.88%** | 0.190 | 0 | 1.1e-03 | 3.88 |

Findings:

* **The 4% plateau is an artifact of starting at zero.** With a 5% init, residual
  share is 9.98% at step 0 and 10.88% at step 800, flat after step 400. Training
  adds 0.9 pp; the level is set by the initialization.
* **Capacity magnitude is not the binding constraint either.** 2.6× the residual
  share changes loss by 0.0003 nats and recovers its 0.0054-nat conversion debt by
  step 100.
* **Direction and magnitude are independent.** Diversity is unchanged (0.00037),
  so the combined orth+mag arm is the untested interaction.
* The exactness traded is small and visible: KL 1.1e-3, top-1 agreement 98.8%.

Full write-up: `reports/P1-MAGNITUDE.md`.

### 5.5 Interaction: both levers together (819k tokens/arm)

| arm | val after | diversity | residual share | imbalance | dead | conv KL |
|---|---|---|---|---|---|---|
| ORRG all (control) | 2.8289 | 0.0004 | 4.18% | 0.203 | 0 | 3.9e-10 |
| ORRG orth | 2.8277 | **0.0486** | 4.20% | 0.211 | 0 | 3.9e-10 |
| ORRG mag | 2.8292 | 0.0004 | **10.88%** | 0.190 | 0 | 1.1e-03 |
| ORRG orth+mag | 2.8310 | **0.0484** | **10.91%** | 0.205 | 0 | 1.1e-03 |

Findings:

* **The levers are independent.** Diversity lands at 99.6% of the orth arm's
  value and share at 100.3% of the mag arm's, so neither axis amplifies or
  suppresses the other.
* **Combining them is stable** (0 dead experts, entropy 0.998) but not better:
  val 2.8310 is 0.0021 worse than the control, inside one-seed noise.
* Direction, magnitude, balance and target sharpness are all controllable without
  moving loss at 819k tokens; the binding constraint, if any, is elsewhere.

Full write-up: `reports/P1-INTERACTION.md`.

## 6. Serving economics

Active parameters are chosen at load time by `top_residuals`:

| k | total | active | active % | TFLOPs/token |
|---|---|---|---|---|
| 0 | 27.03B | 5.89B | 21.8 % | 0.012 |
| 1 | 27.03B | 8.91B | 33.0 % | 0.018 |
| 2 | 27.03B | 11.93B | 44.1 % | 0.024 |
| 7 | 27.03B | 27.03B | 100 % | 0.054 |

KV cache is unchanged (attention is untouched), which is the main serving
advantage over upcycling a full 27B dense model. Dispatch is dropless with one
batched call per (group, variant) pair.

---

## 7. Scale-up protocol to 27B

Every stage has an entry check and a kill criterion. No stage is skipped.

### P0 — Pre-flight on the real seed (1 GPU-day, no training)

1. `python -m orrg inspect --model <seed>` → block count must equal layer count.
2. Carve with the target config, run 16 real batches through both models:
   * **gate:** per-layer relative carve error < 1e-5, logit KL < 1e-6.
3. Measure tokens/s and peak memory for k = 0, 1, 2; extrapolate serving cost.
4. Save/load round-trip; verify identical logits after reload.
5. **Gate:** all four pass. Any failure is an implementation bug, not a result.

### P1 — 1B-seed rung (≈ 900 GPU-hours, 2 seeds)

Scale the toy protocol up: ~1B dense seed, 4–5B tokens, five arms
(dense continued, naive upcycling, ORRG-frozen, ORRG-all, ORRG-all with
`top_residuals=2`), 2 seeds.

* Primary: validation loss on held-out data.
* Secondary: conversion KL, time-to-seed-parity, diversity, router entropy.
* **Gate:** ORRG-all ≥ naive − 0.01 nats and conversion KL < 1e-3. If the
  lossless start does not survive real data, stop.

### P2 — 10B-token pilot on the 6.4B seed (≈ 1.8k GPU-hours, 1 seed)

Three arms: ORRG-all, naive upcycling, dense continued. Same token budget.

* **Gate:** ORRG-all beats *both* by more than 3× the paired evaluation spread
  measured on 8 held-out shards. Toy spread was 0.002–0.004.

### P3 — 50B-token main run (≈ 3.0k GPU-hours, k=1)

One or two arms, kill criteria pre-registered:

* conversion KL > 1e-3 → bug, stop.
* within-group dead experts > 20 % after 1B tokens → increase `bias_update_rate`
  or restart that arm.
* diversity < 0.01 after 1B tokens → experts never diverged; stop.
* loss worse than naive upcycling by > 0.02 → stop.

### P4 — Evaluation and serving

Held-out loss, MMLU/GSM8K/HellaSwag/IFEval, router health, k-sweep quality
(does k=2 beat k=1 enough to justify 34 % more FLOPs?), and latency at batch
sizes 1/8/32.

### Ablations (single factor, P1 rung)

`groups ∈ {4,6,8}`, `variants ∈ {4,8}`, `shared ∈ {1,2}`, `k ∈ {0,1,2}`,
router distillation on/off, residual noise at init `{0, 0.01}`, activation-
clustered vs contiguous shards, `bias_update_rate ∈ {0, 1e-3}`,
`residual_init=copy` (the naive control).

The 360M rung in §5 already measures three of these — clustered vs contiguous,
router distillation on/off, and the `residual_init=copy` control — and none of
the remaining ones has been run.

### Budget summary (H100, 25 % MFU, formula-based)

| run | tokens | active | GPU-hours |
|---|---|---|---|
| P1 1B arm × 5 × 2 seeds | 4B | ≈1B | ≈900 |
| P2 6.4B pilot × 3 arms | 10B | 8.9B | ≈1.8k |
| P3 6.4B main × 2 arms | 50B | 8.9B | ≈6.0k |

---

## 8. Risks and threats to validity

* **Attention is not compressed.** Active parameters include it in full. The
  9B-A2B carve target is impossible for this reason (§1), and this is the honest
  boundary of the method.
* **Router distillation has not beaten task-loss-only routing, and sharpening it
  backfires.** At τ=1 the target stays near-uniform (CE 0.668 against
  ln 2 = 0.693) and separates by 0.0014 nats at 819k tokens. At τ=0.25 the signal
  is real (CE 0.536) but imbalance rises 0.195 → 0.460, dead experts 7 → 34, and
  loss is 0.009 nats worse: the norm differences being amplified are mostly
  noise.
* **The real-checkpoint result is a rung.** Two seeds, 82k tokens per arm, one
  model family, WikiText only, no downstream evaluations. It bounds conversion
  cost and early training behaviour, not final model quality.
* **Toy-scale caveats still apply.** Two seeds, 0.67M parameters, synthetic
  corpora, 300 steps; the 0.004 win over dense there is within seed noise.
* **Neither residual direction nor residual magnitude buys loss at this scale.**
  Diversity can be forced (0.0004 → 0.0486) and residual share can be raised
  (4.18% → 10.88%, via a scaled down-projection init with conversion KL 1.1e-3),
  each at no measurable loss cost. Combining both changes nothing either
  (diversity and share add exactly; loss 0.0021 worse, inside one-seed noise).
  Capacity is not the binding constraint at 819k tokens; the next tests belong at
  longer budgets or larger seeds.
* **Memory.** Full fine-tuning a 27B-total model needs ≈432 GB of optimizer
  state; the frozen-skeleton recipe cuts this to ≈65 GB but lost at toy scale.
* **Dropless dispatch** trades memory for simplicity; long sequences with
  skewed routing may need capacity-aware variants.

---

## 9. Reproduce what is in this spec

```bash
PYTHONPATH=. python -m pytest tests/test_orrg.py -q
PYTHONPATH=. python experiments/smoke.py
PYTHONPATH=. python experiments/toy_scale.py --steps 300 --seed-steps 400 --seeds 2
PYTHONPATH=. python experiments/p1_real.py --steps 80 --seeds 2 --batch 4 --seq 256 \
    --shared 2 --groups 4 --variants 3 --optimizer adafactor \
    --distill-coef 0.1 --distill-tokens 128 --cluster-tokens 512
PYTHONPATH=. python experiments/p1_report.py
PYTHONPATH=. python experiments/p1_long_report.py
PYTHONPATH=. python experiments/p1_real.py --steps 800 --seeds 1 --batch 4 --seq 256 \
    --shared 2 --groups 4 --variants 3 --optimizer adafactor \
    --distill-coef 0.1 --distill-tokens 128 --trace-every 100 \
    --orth-step 3e-5 --residual-noise 0.02 --distill-target-temp 0.25
PYTHONPATH=. python -m orrg scale --hidden 4096 --inter 10240 --layers 32 \
    --vocab 128000 --heads 32 --kv-heads 8 --shared 2 --groups 6 --variants 8
```
