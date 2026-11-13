# ORRG-MoE Spec v4 — 9B-A2B and 27B-A6B

**Two concrete targets, the arithmetic that decides whether they are reachable, and a
gated plan to test the method before spending scale compute.**

Every model is named `T / A@k` — total parameters / active parameters with `k`
routed experts per group actually firing. The requested labels (`9B-A2B`,
`27B-A6B`) are kept in headings; §2.2 shows what they become once `k` is fixed.

Status: planning document. §1 and §2.1–2.4 are measured (v3 rungs plus the JAX
harness). T1 in §5 has been run at 2.4M scale and passed (`reports/T1-CAPACITY.md`).
§2.5–§4 and §6–§8 are proposals — nothing here has been run at 9B or 27B.

Supersedes `spec/ORRG-SPEC-v3.md` for planning; v3 remains the record of what was
measured on SmolLM2-360M.

---

## 0. Summary

Eleven changes to the previous plan, each driven by a measurement rather than a
preference:

| # | Change | Why |
|---|---|---|
| C1 | **Targets are relabelled by the serving `k`.** The 9B target is `9B-A3B` at `k=1`, not `9B-A2B`. | §2.2: `A(k=0)` equals the seed, and at `k=0` routed experts do not affect the forward pass, so they receive no gradient and cannot help. A2B is only reachable by shrinking the skeleton. |
| C2 | **`9B-A2B` becomes a distillation target, not a conversion target.** | Shrinking the skeleton to `f=0.5` costs a large conversion KL and needs KD recovery. That is a different project with a different gate. |
| C3 | **Regime is the first-order variable.** Every rung adds a capacity-binding precondition. | The 360M rungs sat at 1.2–4 tokens/param on 1.33M repeated tokens (data-bound); the JAX toy sat at a high token/param ratio (capacity-bound). The two disagree because of the regime, not the recipe. |
| C4 | **Stop optimizing diversity.** Drop perturbation, the S3 specialization stage, and sharp contribution targets from the default recipe. | JAX: exact copies match perturbed variants (−0.0006); removing S3 is *better* by 0.011; the library: orth/mag reach 0.048 diversity and 10.9% residual share with zero loss change. |
| C5 | **Keep the cluster router warm-start; it is the one router intervention that pays.** | JAX: random labels 1.7918 vs 1.7225 clustered (+0.064 nats). Contribution targets were inert at τ=1 and harmful at τ=0.25. |
| C6 | **Add a matched-active baseline (`B2`).** | The tied-capacity control shows the JAX win comes from capacity (+0.031 when tied), and every arm so far also had more trainable params than dense. Without `B2`, structure and capacity are confounded. |
| C7 | **Demote `k=0` from a deployment mode to a warm start.** | At `k=0` the experts are multiplied out of the forward pass, so no gradient reaches them; the mode is lossless but cannot improve on the seed. |
| C8 | **Budget in tokens per total parameter, not epochs.** | ≥20 tokens per total parameter is the regime where added capacity is trainable. At 9B total that is ≥180B tokens; at 27B total, ≥540B. |
| C9 | **Token budget drives cost; teacher cost must be capped.** | An online 27B dense teacher over 300B tokens is 54.6k GPU-hours — 3× the student. Use it on ≤10% of steps or offline on a fixed slice. |
| C10 | **Three seeds and a pre-registered MDE on every go/no-go rung.** | P1/P2 used one seed at the long rung and 0.005-nat margins, which cannot resolve the effects being claimed. |
| C11 | **Port the JAX winner (ORRG-A) into the library before the 9B run.** | Additive residuals (`orrg`) and group-top-1 replacement (`ORRG-A`) have never been compared at the same scale. The JAX result is the only measured win. |

The recommended sequence is: fix the two library deltas → prove capacity binds →
compare ORRG-A against additive residuals at 1B → only then 9B. §5 has the gates,
§6 the test protocol, §7 the budgets.

---

## 1. What the evidence now says

Both harnesses are real and they disagree. The disagreement is informative.

| | library (`orrg`, additive residuals) | JAX toy (`ORRG-A`, replace, group top-1) |
|---|---|---|
| seed | SmolLM2-360M, 32 layers, hidden 960 | 64-dim, 3 layers, byte-level |
| training | 82k / 819k / 2.72M tokens per arm | 2.5k seed steps + 1.2k arm steps |
| tokens per **total** param | 0.23 / 1.2 / 4.0 | order 10² |
| capacity binding? | no (`P2`: dense overfits, val rises 0.21 nats) | yes (undertrained) |
| result | no loss separation; residual capacity inert | ORRG-A −0.008…−0.019 vs naive on every seed |

**Reconciliation.** The library's `P2` endpoint is a degradation regime: dense
bottoms out at 2.8039 at step 100 of 2,660 and gives back 0.2132 nats while the
train/val gap opens to 1.29 — memorization of 1.33M tokens by a 362M model, not a
capacity ceiling. Under that regime an arm with three times the parameters has
nothing to spend them on, and the null result is expected. The JAX toy is
capacity-starved, and there the extra capacity is worth 0.008–0.019 nats.

So the measured claim is narrow and defensible: **ORRG's gain is capacity, and it
appears only when capacity binds.** It is not a specialization method, and the
evidence says so directly — JAX routing-to-domain mutual information is 0.052
(max 1.10) and the domain-ablation score (0.0094) is *below* naive upcycling's
(0.0193), yet ORRG-A still wins on loss.

### 1.1 JAX mechanism controls (3 seeds each, paired on the same seed)

| control | val loss (s0/s1/s2) | vs ORRG-A | reading |
|---|---|---|---|
| ORRG-A reference | 1.7171 / 1.7213 / 1.7292 | 0 | baseline |
| exact copies (perturbation 0) | 1.7169 / 1.7203 / 1.7287 | −0.0003 / −0.0010 / −0.0005 | **perturbation is unnecessary**; conversion KL 0.0002 vs 0.0015 |
| no S3 specialization stage | 1.7074 / 1.7099 / 1.7178 | −0.0097 / −0.0114 / −0.0113 | **S3 hurt**; best arm, 0.019 below naive on all seeds |
| tied capacity (variants share weights) | 1.7512 / 1.7520 / 1.7620 | +0.0341 / +0.0307 / +0.0329 | **the gain is capacity**, not the pipeline; worse than dense |

Caveats that stay in the record: the no-S3 arm also receives 300 extra
full-model joint steps, so stage removal and extra steps are not separated; the
raw `results_ctrl.jsonl` / `results_tied.jsonl` files are not in the evidence
pack, so these numbers come from `reports/JAX-TOY.md` tables; and perturbation 0.3
is worse than 0.05 on loss while raising conversion KL 0.0015 → 0.049, so the
zero-perturbation arm is the right operating point.

### 1.2 Library rungs, condensed

| quantity | 360M rung, 819k tok | two-epoch rung, 2.72M tok |
|---|---|---|
| conversion KL | 3.9e-10 (naive: 0.31) | 3.9e-10 |
| ORRG vs dense | +0.0085 nats | +0.0199 nats |
| diversity reached | 0.0004 (forced: 0.0486) | 0.0014 / 0.4308 |
| residual/expert output share | 4.18% (max 10.91%) | 6.69% / 11.88% |
| dead experts | 0 | 14 / 0 |
| verdict | no separation | no separation |

Both rungs are marked *not capacity-bound*. They establish that conversion is
lossless on real weights and that the levers move what they claim to move; they
do not establish that ORRG beats anything.

---

## 2. Targets and the arithmetic that decides them

### 2.1 Definitions

```
T      total parameters
N0     seed (dense) parameters
F      seed FFN parameters            N_nonFFN = N0 - F
D      seed FFN intermediate width
A(k)   active parameters when k residual experts per group are active
```

For a **lossless** conversion the skeleton tiles the FFN exactly, so
`A(0) = N0` and

```
A(k) = N0 + k · g/(s+g) · F        (growth mode, residuals copied from shards)
```

Active parameters are a serving choice, not a property of the checkpoint. Every
number in this document is therefore written `T / A@k`.

### 2.2 The reachability result

Since `A(0) = N0`, the smallest active fraction a lossless conversion can have is

```
A/T ≥ N0/T
```

and at `k=0` the routed experts are multiplied out of the forward pass, so they
receive no task gradient and cannot improve the model. That gives two rules:

1. **A lossless `T`-total model cannot be more sparse than its seed.** If the
   target's `A/T` is below `N0/T`, the skeleton must shrink and the conversion is
   no longer lossless (C2).
2. **`k=0` is a warm start, not a deployment mode.** Serving at `k=0` reproduces
   the seed's quality at best; the extra `T − N0` parameters are inert.

Applying this to the two targets decides them:

| target | `N0/T` | target `A/T` | lossless? | conclusion |
|---|---|---|---|---|
| 9B-A2B | 1.995/9.0 = 0.222 | 2.0/9.0 = 0.222 | **at `k=0` only** | reachable as a label, but `k=0` cannot train the experts; at `k=1` it is 9B-A3B. True A2B needs `f<1` + KD. |
| 27B-A6B | 5.893/27.03 = 0.218 | 6.0/27.0 = 0.222 | **yes, at `k=0`** | the v3 config already *is* 27B-A6B at `k=0`, and 27B-A9B at `k=1`. |

This is the central planning correction (C1, C2, C7). The previous plan implied
`9B-A2B` was an ORRG sweet spot; arithmetically it is a distillation job, and the
lossless path from a 2B seed is `9B-A3B`.

### 2.3 Target A — 9B, lossless path (recommended first)

Seed shape `hidden 2560 / inter 6912 / 24 layers / vocab 128k / 20 heads / 5 kv`,
gated, tied embeddings:

| | value |
|---|---|
| dense seed | **1.995B** (`N_nonFFN` 0.721B = 36.1%, `F` 1.274B = 63.9%) |
| config | `shared=2, groups=6, variants=8` |
| total | **8.68B** (v=9 → 9.64B) |
| conversion KL | 3.9e-10 expected (measured at 360M) |

| `k` | active | A/T | TFLOPs/token |
|---|---|---|---|
| 0 | 2.00B | 23.0% | 0.004 |
| 1 | **2.95B** | 34.0% | 0.006 |
| 2 | 3.91B | 45.0% | 0.008 |
| 3 | 4.86B | 56.0% | 0.010 |

Ship label: **9B-A3B**, with `k=0` retained as the exact-seed checkpoint and `k=2`
as the quality-critical setting.

### 2.4 Target B — 27B, lossless path

Seed shape `hidden 4096 / inter 10240 / 32 layers / vocab 128k / 32 heads / 8 kv`:

| | value |
|---|---|
| dense seed | **5.893B** (`N_nonFFN` 1.867B = 31.7%, `F` 4.027B = 68.3%) |
| config | `shared=2, groups=6, variants=8` |
| total | **27.03B** |
| conversion KL | 3.9e-10 expected |

| `k` | active | A/T | TFLOPs/token |
|---|---|---|---|
| 0 | 5.89B | 21.8% | 0.012 |
| 1 | **8.91B** | 33.0% | 0.018 |
| 2 | 11.93B | 44.1% | 0.024 |
| 3 | 14.95B | 55.3% | 0.030 |

Ship label: **27B-A9B**, with the v3 `27B-A6B` label applying only at `k=0`.

### 2.5 Target C — true `A2B` / `A6B` by shrinking the skeleton (distillation)

If the label matters commercially — a 9B model that serves at 2B active *while
routing* — the only route is to shrink the skeleton to a fraction `f` of `F` and
route over the released neurons. Experts are capped at their group's shard width
(`expert_width_ratio ≤ 1`, §2.5.1), so expert width *is* shard width and the knobs
are the integers `s, g, V, k`:

```
A = N0 + f·F·(1 + g·k/(s+g))
T = N0 + f·F·(1 + g·(V−1)/(s+g))
```

Every row below is produced by `python -m orrg scale` (Appendix B) and accepted
by `MoEConfig`; all use `ratio=1` and contiguous shards, so all are approximate
and need KD recovery (gate G2):

| target | seed | geometry | total | active | experts | always-on |
|---|---|---|---|---|---|---|
| 9B-A2B | 2B (§2.3) | `f=0.5 s4 g4 v25 k=2` | **9.00B** | **2.00B** | 96 | 50% |
| 27B-A6B | 5.89B (§2.4) | `f=0.5 s4 g4 v24 k=2` | **27.03B** | **5.89B** | 92 | 50% |
| 9B, cheapest | Qwen2.5-1.5B | `f=0.5 s4 g4 v29 k=1` | **9.06B** | **1.25B** | 112 | 50% |
| 9B-A2B | Qwen2.5-1.5B | same, `k=4` | **9.06B** | **2.12B** | 112 | 50% |
| 9B-A1.7B | Qwen2.5-1.5B | `f=0.75 s4 g4 v19 k=1` | **9.06B** | **1.69B** | 72 | 75% |

**Measured debt (`reports/T3-FOUR-PROBES.md`).** On Qwen2.5-1.5B the conversion
KL is 1.4e-3 nats at `f=1`, **0.43 at `f=0.9`**, 1.16 at `f=0.75` and 3.18 at
`f=0.5`. The debt is a cliff: losing 10% of the FFN neurons already costs ~300×
the lossless error, so no row below is a fine-tune-scale recovery — they are all
pretraining budgets with a warm start.

Two properties of that solution table matter:

* **`A/T` is set by the target, not the design.** Rows built for 9B-A2B land at
  0.22 whatever `f` and `V` are; spending less of the budget (the 1.25B row)
  simply lowers `A/T`. Choosing `f`, `k` and `g` moves *where* the sparsity sits,
  not how much there is.
* **`f` is the only lever that buys routing below the seed's active count, and it
  costs conversion fidelity.** At `f=0.5` the skeleton retains half the seed's FFN
  neurons, so the converted model starts far from the teacher and needs KD
  recovery. `f=0.75` halves that debt and needs fewer, wider experts (72 vs 112 at
  9.06B) but keeps 75% of the FFN always on, so the serving win shrinks with it.

#### 2.5.1 Why experts never exceed their shard

An earlier version of this section solved for `e = 8.40%` of `D` against a `6.25%`
shard, i.e. `expert_width_ratio = 1.34`. That geometry cannot be built, and the
cap is semantic rather than a missing feature:

* A residual is a copy of a prefix of exactly one group's shard, and `MoELayer`
  asserts that the shard index sets **partition** the retained intermediate axis
  (`union.numel() == skeleton_width` and `unique().numel() == skeleton_width`). A
  wider expert would have to take neurons owned by another shard, so there is no
  disjoint set for it to copy — and under `cluster_shards` there is not even a
  natural rule for which foreign neurons to reach for.
* It would break the parameter identity. `T = skeleton + g·(V−1)·e` only counts
  distinct capacity because shards are disjoint; overlap counts the same neuron
  twice, so a "9.06B" checkpoint would hold less than 9.06B of distinct function.
* It would leave the KD target undefined. Residual targets are the group base's
  output (§3), which is the base *restricted to its own neurons*; an expert whose
  range crosses a group boundary is no longer a restriction of any base.

So the budget is reached with integer knobs instead: `k` and `V` move in whole
shards, which is why the table above lands at 2.00B rather than an exact 2.0B by
construction.

**Recommendation: do not start here.** Treat §2.5 as Phase 3, gated on the
lossless 9B run showing that routed capacity pays at all (G5). If it does, the
additional active budget can also be bought more cheaply by compressing attention
(§8, risk 3) than by shrinking the FFN skeleton.

**Superseded in part by §2.6.** The "A2B needs distillation" conclusion above
holds only for the 2B seed assumed in §2.3 (FFN = 64% of params). Active budget is
`A(1) = N0 + r·F` with `r = g/(s+g)`, so a *smaller* seed with a *larger* FFN
fraction reaches A2B losslessly. With Qwen2.5-1.5B (`F/N0 = 74.9%`) and `s=g=4`,
`r=0.5`: 9.06B total / **2.12B active**, conversion KL 3.9e-10. No KD, no `f<1`.

---

### 2.6 Seed selection is a design variable, not a given

Active budget is set by the seed and the routed share:

```
A(1) = N0 + r · F ,      r = g/(s+g)      (lossless growth)
```

So the seed is chosen to fit the *active* target and the expert count to fit the
*total*. Sizes needed, at `F/N0 ≈ 0.67`:

| target | `r = 0.75` (`s2 g6`) | `r = 0.50` (`s4 g4`) |
|---|---|---|
| 9B-A2B | ~1.33B | ~1.50B |
| 9B-A3B | ~2.00B | ~2.25B |
| 27B-A6B | ~4.00B | ~4.49B |
| 27B-A9B | ~6.00B | ~6.74B |

### 2.7 Local seed inventory (2026-11-13)

Exact counts from meta-device instantiation; FFN blocks discovered 1:1 with layers
on all three, so `orrg` can convert them.

| model | params | FFN `F` | `nonFFN` | weights on disk |
|---|---|---|---|---|
| `HuggingFaceTB/SmolLM2-360M` | 0.362B | — | — | 0.72 GB |
| `Qwen/Qwen2.5-1.5B-Instruct` | **1.544B** | 1.156B (74.9%) | 0.388B | 3.09 GB |
| `Qwen/Qwen3-1.7B` | **1.721B** | 1.057B (61.4%) | 0.664B | 4.06 GB |
| `Qwen/Qwen3-8B` | 8.191B | 5.436B (66.4%) | 2.755B | **none** (config only) |
| Qwen3.8-9B / 27B repos (5) | — | — | — | **none** |

Nothing with weights exists between 1.72B and 8.19B, and no project directory holds
a checkpoint (the only `.pt` under `outputs/` is a 241 KB round-trip test).

With D1–D5 implemented, the reachable configurations are (all numbers from
`orrg scale`, Appendix B):

| seed | seed params | geometry | total | A@k=0 | A@k=1 | A@k=2 | A/T@1 | lossless | role |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 1.544B | `s4 g4 v14` | 9.06B | 1.54B | **2.12B** | 2.70B | 0.23 | yes | 9B target |
| Qwen2.5-1.5B | 1.544B | `s4 g4 v14` replace | 9.06B | 1.54B | **1.54B** | 1.54B | 0.17 | yes | 9B, flat active |
| Qwen3-1.7B | 1.721B | `s4 g4 v15` | 9.12B | 1.72B | **2.25B** | 2.78B | 0.25 | yes | 9B target |
| Qwen2.5-1.5B | 1.544B | `s4 g4 v29 f=0.5` | 9.06B | 0.97B | **1.25B** | 1.54B | 0.14 | no (KD) | 9B as A~1.3B |
| Qwen3-8B | 8.191B | `s4 g4 v8` | 27.22B | 8.19B | **10.91B** | 13.63B | 0.40 | yes | 27B target |
| Qwen3-8B | 8.191B | `s16 g4 v18` | 26.67B | 8.19B | **9.28B** | 10.37B | 0.35 | yes | 27B, 80% skeleton |
| Qwen3-8B | 8.191B | `s4 g4 v17 f=0.5` | 27.22B | 5.47B | **6.83B** | 8.19B | 0.25 | no (KD) | 27B as A~6.8B |
| 5.89B (v3 shape) | 5.893B | `s2 g6 v8` | 27.03B | 5.89B | **8.91B** | 11.93B | 0.33 | yes | v3 reference |

* **9B is runnable now, losslessly.** Qwen2.5-1.5B at `s4 g4 v14` gives 9.06B
  total / 2.12B active (52 experts); Qwen3-1.7B at `s4 g4 v15` gives 9.12B /
  2.25B. Both are `f=1`, additive, zero-init: conversion KL 3.9e-10. In replace
  mode active stays at the seed's 1.54B while total is still 9.06B.
* **True A2B on 9B costs KD, not pruning.** `f=0.5` with `v=29` gives 9.06B /
  1.25B, but the skeleton drops half the FFN neurons, so the conversion is
  approximate and needs KD (gate G2).
* **27B is not reachable as A9B from local weights** (no ~6B seed). Qwen3-8B
  (config only, 16 GB to download) gives 27.22B / 10.91B lossless, or 26.67B /
  9.28B with 20 shards (80% of the FFN always on — barely an MoE).
* **27B-A6B is now a KD target, not a pruning project.** `f=0.5`, `s4 g4 v17`
  from the 8B seed gives 27.22B / 6.83B at `f=1`'s neuron level — no weight
  pruning, but a real conversion debt. This is why D3 exists.
* **`expert_width_ratio` is capped at 1.0**, i.e. experts are never wider than
  their group shard. This is an invariant, not a gap: shards partition the axis
  and a residual copies a prefix of one shard, so there is no disjoint set of
  foreign neurons to take (§2.5.1). `orrg scale` enforces the same `(0, 1]` bound
  as `MoEConfig`, and §2.5 now lists rows that need no ratio above 1.

**Measured on the real checkpoint (2026-11-13).** The `s4 g4 v14` 9B geometry
was run on the cached Qwen2.5-1.5B in bf16 (`reports/T2-REAL-CONVERSION.md`):
total **9,060,309,936** params and active **2.124B** at `k=1`, matching the
arithmetic to the parameter. Conversion KL **1.324e-3** nats, logit max
`|diff|` 1.062, block carve error 5.4e-3 at layer 0, ~2e-2 through the middle of
the stack and 5.0e-2 worst at layer 27 (dense-vs-converted forward drift, not a
per-layer conversion error), router warm-start agreement **0.988**
(0.967 min) over 28 layers. The construction itself is exact: splitting one real
MLP into the same 8 shards costs 2.7e-3 in bf16 — the same as bf16's own
representation error — and 3.3e-7 in fp32. So the "lossless" claim is a claim
about exact arithmetic, and at bf16 the residual cost is per-shard rounding.

**Hardware reality.** 64 GB RAM, 16 GB GPU, 36 GB free disk. The 9B conversion is
CPU-feasible (18.1 GB bf16 output + 3 GB seed) and its lossless check can run on
CPU, but a GPU forward pass does not fit, full-finetune state is ~145 GB, and the
save leaves ~18 GB of disk. Local work is therefore inspect → convert → verify;
**training the 9B stays on rented compute (§7)**.

---

## 3. Method v4: what to build

### 3.1 Conversion

Unchanged and proven: the skeleton is `shared + groups` shards that tile the
intermediate axis exactly; each group's residuals are copies of that group's
range. Measured carve error 2.5e-07–4.2e-07 relative and ≤1.2e-07 on logits
across five architectures, conversion KL 3.9e-10 on real weights.

New parameter: `skeleton_scale f ∈ (0,1]`. `f=1` is today's behaviour and stays
the default. `f<1` retains the first `f·D` neurons (contiguous or clustered) and
records the measured conversion KL in the report; it is **not** lossless and must
not be described as such.

### 3.2 Routing

Two modes, both first-class:

| mode | forward | lossless at `k=0` | evidence |
|---|---|---|---|
| `additive` (today) | `group = base(x) + Σ selected residuals(x)` | yes, for any router state | v3 rungs; no loss win |
| `replace` (ORRG-A) | `group = selected variant(x)`, one per group | only when variant 0 is selected | JAX win, −0.008…−0.019 |

Grouped routing keeps structural balance (per-group load is exactly `k·N`), which
is why no load-balancing penalty is needed to prevent starvation. Additive mode
keeps the lossless guarantee under any router state, including fully released
routing; replace mode gives the router a real choice and was the only arm that
beat naive upcycling. **Both run in the 9B comparison (`A1` vs `A2`); this is the
highest-value unresolved question in the project.**

### 3.3 Router initialisation and training

| mechanism | status in v4 | evidence |
|---|---|---|
| cluster warm-start (balanced k-means on seed hidden states) | **default on** | JAX: 1.7225 vs 1.7918 random (+0.064); library clustering is free (cohesion 0.225 vs 0.065, KL 3.3e-09) |
| task-loss routing only | default baseline | v3: distillation separated by 0.0014 nats at 819k tokens (inside noise) |
| contribution-target distillation, τ=1 | **ablation, not default** | inert: target CE 0.668 against ln 2 = 0.693 |
| sharpened target τ=0.25 | **off** | real signal (CE 0.536) but imbalance 0.195→0.460, dead experts 7→34, +0.009 nats |
| aux-loss-free bias updates | optional, `rate ∈ {0, 1e-3}` | untested at scale; cheap |
| random router init | control only | JAX: clearly worse |

Cluster labels are a *warm start*, not a target: they tell the router which
experts to try first, then the task loss takes over. They should not be
maintained as a classification objective.

### 3.4 What to drop

| dropped | measured reason |
|---|---|
| perturbation on variant init | exact copies match or beat it (−0.0006) and convert 7.5× better (KL 0.0002 vs 0.0015) |
| S3 specialization stage (anchored variant training) | removing it was better on all 3 seeds by ~0.011 nats |
| orthogonality push (`orth_step`) | raises diversity 121× (0.0004 → 0.0486) with **zero** loss change; keep as a diagnostic knob |
| scaled residual init (`down_init_scale`) | raises residual share 2.6× (4.18% → 10.88%) with **zero** loss change and KL 1.1e-3; keep for the magnitude ablation only |
| sharp contribution targets | destabilizes routing, costs 0.009 nats |

These are not wrong, they are *inert*: every one of them moves its target
statistic substantially without moving loss at the budgets tested. Carrying them
into a 9B run would multiply the arm count for no measured benefit.

### 3.5 Training regime

* **Continued pretraining**, not SFT-style fine-tuning. The whole thesis is that
  capacity must bind; short low-token rungs cannot show it.
* **Token budget ≥ 20 tokens per total parameter** for the decisive rung
  (C8): 180B for 9B-total, 540B for 27B-total.
* **OneCycle over the full token budget**, `lr 2e-4` for the 9B scale (2e-4 was
  used at 360M; re-tune at T2), `bf16`, Adafactor for arms that fit, FSDP/EP for
  the 27B.
* **Frozen vs co-trained skeleton is settled**: co-training is ~6× better than
  freezing at 360M (+0.0016 vs +0.0101). Co-train. Keep frozen only as a
  memory-constrained fallback, documented as a quality trade.
* **Distillation from the dense seed is optional at `f=1`** (conversion is already
  lossless) and **mandatory at `f<1`**. Where used, apply it on-policy at ≤10% of
  steps to bound teacher cost (C9).

---

## 4. Implementation deltas required before the 9B run

None of these is large; all are testable in the existing pytest suite.

| id | delta | acceptance test |
|---|---|---|
| D1 | `expert_mode: "additive" \| "replace"` in `MoEConfig` + `MoEGroup.forward` | at `replace` with `V=1`, output equals the base shard exactly; at `replace` with all weights tied, equals the dense block |
| D2 | `expert_width_ratio` (expert width as a fraction of the group shard, default 1.0) + slicing in `shard.narrow_module` | `ratio=1` reproduces today's numbers bit-for-bit; every expert's up/down shapes agree; active-param accounting matches `scale.py` within 0.5% |
| D3 | `skeleton_scale f` + a `conversion_kl` field in `ConversionReport` | `f=1` is bit-identical to today; `f<1` reports measured KL and refuses to print "lossless" |
| D4 | `router_warm_start: "none" \| "cluster"` wired from `cluster.balanced_kmeans` into `TokenRouter` init | clustered init beats random on a fixed toy seed; init is deterministic given `cluster_seed` |
| D5 | `scale.py`/CLI accounting for `f` and `expert_width_ratio`; `python -m orrg scale` reproduces the §2.3–2.5 tables | a test asserts the 9B/27B rows in §2.3/2.4 exactly |
| D6 | Arm runner: one JSONL record per arm per trace step, 8-shard evaluation, `k`-sweep, throughput and peak memory at `k ∈ {0,1,2}` | a 3-step smoke run produces a complete record with no missing fields |
| D7 | Pre-registration file per rung: hypothesis, primary metric, MDE, seeds, stop rules, git SHA | the report renderer refuses to emit a verdict without it |

D1–D4 are prerequisites for the `A1`/`A2` comparison; D5–D7 are prerequisites for
trusting any 9B number.

**Status: D1–D5 implemented** (`orrg/config.py`, `moe.py`, `shard.py`,
`router.py`, `cluster.py`, `convert.py`, `scale.py`, `cli.py`). Tests:
`tests/test_geometry.py`, 16 cases, including the default-geometry bit-identity
check. D6–D7 remain open.

Beyond the deltas, `orrg/dispatch.py` provides `dispatch="fused"` (default
`"python"`): a grouped-GEMM inference path that reproduces the reference forward
exactly and has no autograd, so it raises outside `torch.no_grad()`. Additive
growth only; uneven (clustered) shard widths are zero-padded. Measured in
`reports/T3-FOUR-PROBES.md` §1b.

---

## 5. Plan: stage ladder

Each stage has an entry check, an exit gate and a kill criterion. No stage is
skipped, and no stage's result is reported as evidence about the next one.

### T0 — Numerical and accounting (hours, no GPU)

`pytest tests -q` (55 tests today, including the D1–D5 geometry cases).

| gate | threshold |
|---|---|
| carve error, 5 architectures | < 1e-5 relative |
| logit max abs diff | < 1e-5 |
| checkpoint round-trip | bit-identical |
| active-param accounting vs `scale.py` | exact, including routers |
| per-group token counts | exactly `k·N` |

### T1 — Capacity-binding precondition (**new, decisive**)

Purpose: prove the corpus and budget can reward added capacity before comparing
any MoE. This is the gate whose absence produced the `P2` null result.

Arms: dense seed vs the same seed widened by 1.5× (FFN 1.5×D, same attention), on
identical batches, same token budget, 3 seeds.

| gate | threshold |
|---|---|
| `dense_wide` vs `dense`, paired val | Δ ≥ **2σ_pair** and ≥ 0.005 nats, CI excluding 0 |
| train/val gap at end | < 0.1 nats (not in a memorization regime) |
| tokens per total param | ≥ 20 |

**If this fails, stop.** A corpus where extra dense capacity does not pay cannot
show that extra MoE capacity pays; change corpus or raise the token budget rather
than running the arms.

Cost: 2 arms × 3 seeds at 1B scale, 20B tokens ≈ **0.9k GPU-hours** (§7).

**Result at 2.4M scale (run 2026-11-13, `reports/T1-CAPACITY.md`): PASS.** 2.41M
vs 3.29M params, 48.2M identical byte tokens, 3 paired seeds: wide − dense =
−0.0219 / −0.0138 / −0.0303, mean **−0.0220**, sd 0.0083, 2 sd 0.0165, 95% CI
[−0.0425, −0.0015]. The delta grows monotonically over the run (−0.0121 → −0.0180
→ −0.0220) and train−val is negative in every arm, so the regime is capacity-bound
rather than data-bound. Caveats: matched steps, not matched compute (the wide arm
spends 1.37× FLOPs/step); three seeds; one corpus. This certifies the regime and
the harness — it is not evidence about 9B.

### T2 — 1B rung, mechanism comparison (≈ 4.2k GPU-hours)

Seed ~1B dense, ≥20B tokens (20 tokens/param), 3 seeds, paired data order.

Arms: `B0` dense, `B1` naive upcycling, `B2` naive matched-active, `A1` ORRG
additive, `A2` ORRG-A replace, `A3` A2 with random router init, `A4` A2 tied
capacity. (§6.1.)

| gate | threshold |
|---|---|
| `A2` vs `B1` | paired Δ ≤ −0.01 nats, 95% CI excluding 0 |
| `A2` vs `B2` (matched active) | Δ < 0 — structure must beat capacity alone to claim a method effect |
| conversion KL (`A1`, `A2`) | ≤ 1e-3 |
| router health | dead experts ≤ 5%, load CV ≤ 0.15 |
| `A2` vs `A1` | reported, not gated: this picks the mode for T3 |

**Kill:** `A2` not better than `B1`, or the only win is over `B0` while `B2`
matches it. In the second case the honest conclusion is "capacity, not ORRG",
and the 9B run is not justified.

### T3 — 9B pilot (≈ 7.3k GPU-hours)

Seed 1.995B, `s=2 g=6 v=8`, 100B tokens (11.5 tokens/param — below the ≥20
ideal, so this is a pilot, not the decisive rung), 2 seeds. Arms: `B0`, `B1`,
`A2`, plus `A5` (`f=0.5` + KD, true A2B) if T2 supports it.

| gate | threshold |
|---|---|
| conversion KL | ≤ 1e-3 (`f=1`); `f=0.5` must state its measured KL |
| `A2` vs `B1` | ≥ 0.01 nats, CI excluding 0 |
| `A2` vs `B0` | not worse than +0.005 nats |
| `k`-sweep | `A(2)` beats `A(1)` by ≥ the FLOP-justified margin, else serve at `k=1` |
| throughput | measured at `k ∈ {0,1,2}`, batch 1/8/32 |

### T4 — 9B main (≈ 12k GPU-hours at 100B tokens, ≈ 24k at 200B)

180–200B tokens (≥20 tokens/param), 3 seeds, arms `B1`, `A2` and the winner of
`A1`/`A2`. This is the first run whose result may be quoted as a 9B-quality
claim.

### T5 — 27B pilot and main (≈ 54k GPU-hours pilot, ≈ 130k decisive)

Only after T4 passes. 27.03B total, `A(1)=8.91B`; 300B tokens for the pilot
(11 tokens/param), 540B for the decisive rung. Teacher used on ≤10% of steps.

| gate | threshold |
|---|---|
| full-finetune memory | ≈432 GB optimizer state → ≥32×H100 with FSDP+EP; the frozen fallback is ≈65 GB but is a documented quality trade |
| `A2` vs `B1` | ≥ 0.01 nats at 300B, CI excluding 0 |
| `A2` vs dense continued | not worse, and better on ≥2 of 5 downstream tasks |

### Decision tree

```
T0 pass?
 ├─ no  → fix the bug; no results exist yet
 └─ yes → T1 capacity-binding
           ├─ fail → change corpus/budget; do NOT run MoE arms
           └─ pass → T2 1B, 3 seeds
                     ├─ A2 ≤ B1 − 0.01 and A2 < B2 → T3 9B pilot (A1 vs A2 settled here)
                     ├─ A2 beats B1 but B2 matches A2 → capacity, not ORRG: stop, report
                     └─ A2 not better than B1 → ORRG does not scale from this seed: stop
T3 pass → T4 9B main (3 seeds) → T5 27B
T3 fail on k-sweep only → serve at k=1, drop k=2 claims, continue
```

---

## 6. Test plan

### 6.1 Arms and what each controls for

| id | arm | `T` | `A@k=1` | isolates |
|---|---|---|---|---|
| `B0` | dense continued | 1.995 | 1.995 | no conversion; the quality reference |
| `B1` | naive upcycling (copy ×8, random router, top-1) | 8.68 | 2.95 | the baseline the literature reports |
| `B2` | **naive upcycling, matched active** (narrow experts, top-1) | 8.68 | **2.00** | structure vs capacity — the missing control |
| `B3` | drop-upcycling | 8.68 | 2.95 | diversity bought at a conversion cost |
| `A1` | ORRG additive, zero init | 8.68 | 2.95 | current library recipe |
| `A2` | ORRG-A port: exact copy, group top-1 **replace**, cluster warm-start, no S3 | 8.68 | 2.95 | the only measured win |
| `A3` | `A2` with random router init | 8.68 | 2.95 | value of the cluster warm-start |
| `A4` | `A2` with tied variants | 8.68 | 2.95 | capacity vs pipeline (replicates JAX `tied`) |
| `A5` | `A2` with `f=0.5` + KD | 9.06 | **2.00** | true 9B-A2B; conversion debt vs active budget |

`B2` and `A4` are the two controls that decide whether ORRG is a method or a
parameter-count effect. Every previous comparison lacked at least one of them.

### 6.2 Ablations — only on a winning arm, single factor

`groups ∈ {4,6,8}`, `k ∈ {1,2}`, `expert_width_ratio ∈ {0.5,1.0}`,
`skeleton_scale ∈ {0.75,1.0}`, cluster vs contiguous shards, `bias_update_rate ∈
{0,1e-3}`, router warm-start on/off, S3 on/off (to confirm the JAX result at
scale, without the step-budget confound), perturbation `{0,0.05}` (expected null),
`residual_init ∈ {zero,scaled}` (expected null on loss), output KD on/off.

Ten single-factor runs at T2, each ~200 GPU-hours, is ~2.0k GPU-hours — cheaper
than one wrong 9B run. Ablate only if T2 produced a win; otherwise the ablation
matrix is measuring noise.

### 6.3 Seeds, statistics, pre-registration

* **3 seeds** at T1, T2 and T4; 2 at T3 (pilot); 1 at T5 pilot, 2 at T5 main.
* Paired by seed: same data order, same seed checkpoint, same conversion RNG.
* Primary metric: validation loss on **8 held-out shards**, mean over shards;
  report per-domain as secondary.
* Report paired differences with a 95% bootstrap CI over seeds×shards, not raw
  means.
* **Pre-register the MDE** from the T2 measured `σ_pair`: claim a win only if
  |Δ| > 3σ_pair and the CI excludes 0. The 360M rungs had `σ_pair ≈ 0.0015`
  nats, so 0.005-nat claims at one seed were never resolvable — do not repeat that.
* Verdict is written only by the renderer, from the JSONL plus the
  pre-registration file, so the sign cannot be chosen after the fact.

### 6.4 Instrumentation (every 100 steps, JSONL)

| metric | definition | warn | stop |
|---|---|---|---|
| `val_loss` | 8-shard mean | — | primary |
| `conv_kl` | seed vs converted, fixed 2k-token probe | > 1e-3 (`f=1`) | > 1e-2 drift |
| `top1_agree` | next-token argmax agreement with seed | < 0.95 | < 0.85 |
| `expert_share` | ‖expert output‖ / ‖layer output‖ | < 2% after 1B tok | inert capacity |
| `dead_experts` | < 1% tokens per expert | > 5% | > 20% after 1B tok |
| `load_cv` | per-expert load coefficient of variation | > 0.15 | > 0.35 |
| `router_entropy` | normalised entropy of routing distribution | < 0.9 | < 0.7 |
| `cluster_agree` | router top-1 vs cluster label | < 0.4 early | — (diagnostic) |
| `diversity` | 1 − mean cosine of expert weights | — | diagnostic only (C4) |
| `train_val_gap` | 100-step mean train − val | > 0.5 | > 1.0 (memorization) |
| `tokens_per_s`, `peak_mem` | per `k` | — | — |

`diversity` and `router_entropy` stay in the log because they diagnose collapse,
but neither is a target any more. The target statistic is `expert_share` — output
share is what the model actually uses.

### 6.5 Evaluation suite

* **Held-out loss** on 8 shards from a corpus that was never in the token stream;
  report mean and per-domain.
* **Downstream** at T3+: MMLU, GSM8K, HellaSwag, ARC-Challenge, IFEval, plus one
  domain-matched set. 0-shot and 5-shot where standard.
* **`k`-sweep**: evaluate the same checkpoint at `k=0,1,2,3`. This is the single
  most informative evaluation for the whole project: it measures whether added
  active capacity converts into quality, independently of training.
* **Serving**: prefill and decode latency at batch 1/8/32, KV-cache size,
  expert-dispatch overhead, peak memory.
* **Honesty check**: report the seed's own scores, so a win can be attributed to
  conversion+training rather than to the seed.

### 6.6 Artifacts

```
experiments/results/<rung>/<arm>_s<seed>.jsonl     per-step trace
experiments/results/<rung>/<arm>_s<seed>.json      endpoint summary
experiments/results/<rung>/manifest.json           configs, git SHA, token counts, seeds
experiments/results/<rung>/PRE_REGISTRATION.md     hypothesis, MDE, stop rules
experiments/results/<rung>/eval/<task>.json        downstream scores
reports/<RUNG>.md                                  rendered verdict
```

### 6.7 What would falsify the thesis

1. `B2` (matched active) matches `A2` — the gain is capacity, not ORRG. Report it.
2. T1 fails — the corpus cannot reward capacity; no MoE conclusion is available.
3. `A(2)` never beats `A(1)` on quality at any scale — routed capacity is not
   usable, and the elastic knob is decorative.
4. `f=0.5` conversion KL cannot be recovered by ≤10% of the budget — true A2B is
   not affordable, and the label should be dropped rather than fudged.
5. Downstream scores move opposite to validation loss — the loss proxy is wrong
   for this conversion, and the plan needs a new primary metric.

---

## 7. Budgets

H100, 25% MFU (unmeasured at this scale), `6·A·tokens`; activation remat,
dispatch overhead and evaluation excluded. Costs are **per run** and **per rung**
separately, because a rung is `arms × seeds` runs.

| run | arms × seeds | tokens | `A(k=1)` | GPU-h / run | GPU-h total |
|---|---|---|---|---|---|
| T1 precondition: dense + 1.5× wide | 2 × 3 | 20B | 1.0 / 1.25B | 135 / 169 | **0.9k** |
| T2 core: `B0 B1 B2 A1 A2` | 5 × 3 | 20B | ~1.5B | 202 | **3.0k** |
| T2 controls: `A3 A4` | 2 × 3 | 20B | 1.5B | 202 | **1.2k** |
| T3 pilot: `B0 B1 A2` | 3 × 2 | 50B | 2.95B | 995 | **6.0k** |
| T3 add `A5` (`f=0.5` + KD) | 1 × 2 | 50B | 2.00B | 674 | **1.3k** |
| T4 main: `B1 A2` at 100B | 2 × 3 | 100B | 2.95B | 1,989 | **11.9k** |
| T4 main: `B1 A2` at 200B (≥20 tok/param) | 2 × 3 | 200B | 2.95B | 3,978 | **23.9k** |
| T5 pilot: `B0 B1 A2` | 3 × 1 | 300B | 8.91B | 18,024 | **54.1k** |
| T5 main: `B1 A2` | 2 × 2 | 540B | 8.91B | 32,443 | **129.8k** |

Read this honestly rather than as a menu:

* **A plan that fits ~25k GPU-hours** is T1 + T2 + T3 + T4 at 100B tokens
  (0.9 + 4.2 + 7.3 + 11.9). That buys the first 9B evidence at 11.5 tokens/param,
  which is *below* the ≥20 rule — so a null result is not decisive and must be
  reported as a pilot.
* **The decisive 9B rung is T4 at 200B tokens: ~24k GPU-hours**, ≈1,000
  GPU-days, ≈42 days on 24 H100s or ≈21 days on 48.
* **The decisive 27B rung is ~130k GPU-hours (~15 GPU-years)** and is probably
  out of reach. The realistic 27B evidence is the 300B-token pilot at ~54k
  GPU-hours (≈6 GPU-years), read with the same "pilot" caveat.
* **Teacher cost dominates if left uncapped.** Online distillation from the 27B
  dense seed is `2·27B·tokens` = **54.6k GPU-hours at 300B tokens — 3× the
  student**. Use it on ≤10% of steps (5.5k) or precompute top-k logits on a fixed
  slice: top-32 over 10B tokens is ~0.6 TB (feasible); over 300B it is 18 TB
  (not).
* **Data.** 9B at 200B tokens and 27B at 540B need corpora that large with
  licensing settled before T4/T5. A 1.33M-token corpus cannot be reused with more
  epochs — that is exactly the `P2` failure mode.

Data: 9B at 200B tokens and 27B at 540B tokens need corpora that large with
licensing settled before T4/T5. A 1.33M-token corpus cannot be reused with more
epochs — that is exactly the `P2` failure mode.

Storage per 9B checkpoint: 8.68B total; trainable-only saves are ~2.95B×2 bytes
plus optimizer state, ~35–70 GB per save with Adafactor. Budget 3 TB per arm for
traces, saves and eval artifacts.

---

## 8. Risks and threats to validity

1. **The only measured win is a tiny JAX model.** 0.008–0.019 nats on a 64-dim
   byte model is a mechanism hint, not a scaling result. It is also confounded:
   the no-S3 arm got 300 extra joint steps.
2. **The library's real-checkpoint evidence is a null result at two budgets.**
   Lossless conversion is proven; a quality win is not. The plan above exists
   because the null result was in a data-bound regime, not because it was wrong.
3. **Attention is untouched and counts toward active.** `N_nonFFN` is 32–36% of
   the active budget at both targets. Compressing attention (GQA is already in
   the seed; MLA/linear attention is not researched here) buys active budget more
   cheaply than shrinking the FFN skeleton, and is out of scope.
4. **Conversion exactness is not free at `f<1`.** The KL is unmeasured for the
   §2.5 geometry; the plan gates it rather than assuming recovery.
5. **Same-family teacher limit** applies to any KD stage: the dense seed is the
   natural teacher, which is fine for `f<1` but rules out cross-family transfer.
6. **Memory.** Full fine-tuning 27.03B needs ≈432 GB of optimizer state. The
   frozen-skeleton fallback (≈65 GB) is 6× worse on loss at 360M.
7. **Dropless dispatch** trades memory for simplicity; expert-parallel sharding
   at 27B may need capacity-aware dispatch at long sequence lengths.
8. **Statistics.** 3 seeds and 8 shards is the minimum that can resolve a
   0.01-nat effect; a 0.005-nat claim needs more of both.
9. **Regime dependence is the whole story.** If T1 passes but the 9B run sits
   below 20 tokens/param, a null result is uninformative and must be reported as
   such rather than as evidence against the method.
10. **Evidence-pack gap.** The JAX control numbers are quoted from
    `reports/JAX-TOY.md`; the raw `results_ctrl.jsonl` and `results_tied.jsonl`
    were not in the archive. Re-run or recover them before T2 is cited.

---

## 9. Open decisions

1. **Confirm 27B-A6B as the second target** — and accept that at `k=1` it is
   27B-A9B, or fund the `f=0.5` distillation path (§2.5).
2. **Seed checkpoints — 9B resolved, 27B open (§2.7).** Local seeds with weights
   are Qwen2.5-1.5B (1.544B) and Qwen3-1.7B (1.721B); either yields a lossless
   ~9B model at 2.1–2.25B active, so the 9B path does **not** need prune-then-
   convert. The 27B path does: no ~6B seed is present, Qwen3-8B is config-only and
   would give 27B-A11B. Decide between (a) download a ~4B seed (for A6B) or ~6B
   (for A9B), (b) prune Qwen3-8B to ~6B and convert, or (c) relabel the 27B target
   to A11B and use the 8B seed.
3. **Compute envelope.** The decisive 9B rung is ~24k GPU-hours and the decisive
   27B rung ~130k (§7). Which is funded, and is a 300B-token 27B *pilot* the
   intended ceiling? Is `k=0` serving acceptable if routed capacity fails?
4. **Corpus.** ≥200B unique tokens with licensing for T4; ≥540B for T5.
5. **Whether true A2B is a product requirement.** If it is, the `f=0.5` path is
   mandatory and its conversion-KL gate should be measured at T2, not T3.
6. **Whether attention compression is in scope.** It is the cheapest source of
   active budget and the only way to keep the full skeleton while routing at 2B.

---

## Appendix A — algebra

```
N_nonFFN = N0 − F
lossless:      A(k) = N0 + k·g/(s+g)·F                     A(0) = N0
shrunk (f<1):  A    = N_nonFFN + F·(f + g·k·e)
               T    = N_nonFFN + F·(f + g·(V−1)·e)
               ⇒ e      = (A − N_nonFFN − f·F) / (g·k·F)
               ⇒ V − 1  = k·(T − N_nonFFN − f·F) / (A − N_nonFFN − f·F)
GPU-hours      = 6·A·tokens / (MFU · 989e12) / 3600
```

`A/T` is invariant to `f`, `k` and `g` at fixed `(A, T)`: the three parameters
choose where the sparsity sits, not how much exists.

## Appendix B — reproduce the tables

```bash
# 9B from the local Qwen2.5-1.5B seed (lossless), and the same in replace mode
python -m orrg scale --hidden 1536 --inter 8960 --layers 28 --vocab 151936 \
    --heads 12 --kv-heads 2 --shared 4 --groups 4 --variants 14 --tokens 200e9 --gpus 24
python -m orrg scale --hidden 1536 --inter 8960 --layers 28 --vocab 151936 \
    --heads 12 --kv-heads 2 --shared 4 --groups 4 --variants 14 \
    --expert-mode replace --tokens 200e9 --gpus 24
# true A2B on 9B: approximate, needs KD
python -m orrg scale --hidden 1536 --inter 8960 --layers 28 --vocab 151936 \
    --heads 12 --kv-heads 2 --shared 4 --groups 4 --variants 29 \
    --skeleton-scale 0.5 --expert-ratio 1.0 --tokens 200e9 --gpus 24
# 27B from the 8B seed, lossless, and the A6B KD variant
python -m orrg scale --hidden 4096 --inter 12288 --layers 36 --vocab 151936 \
    --heads 32 --kv-heads 8 --shared 4 --groups 4 --variants 8 --tokens 300e9 --gpus 24
python -m orrg scale --hidden 4096 --inter 12288 --layers 36 --vocab 151936 \
    --heads 32 --kv-heads 8 --shared 4 --groups 4 --variants 17 \
    --skeleton-scale 0.5 --tokens 300e9 --gpus 24
```

The CLI assumes gated FFN and tied embeddings; for Qwen3-8B (untied `lm_head`)
add `--untied`.

# §2.5 KD rows: f=0.5, ratio=1; the printed sweep gives every k, the table
# quotes the k=2 line
python -m orrg scale --hidden 2560 --inter 6912 --layers 24 --vocab 128000 \
    --heads 20 --kv-heads 5 --shared 4 --groups 4 --variants 25 \
    --skeleton-scale 0.5 --tokens 200e9 --gpus 24
python -m orrg scale --hidden 4096 --inter 10240 --layers 32 --vocab 128000 \
    --heads 32 --kv-heads 8 --shared 4 --groups 4 --variants 24 \
    --skeleton-scale 0.5 --tokens 300e9 --gpus 24
```

All §2.5 rows come from the commands above (the 2B and 5.89B shapes are §2.3's
and §2.4's seeds); `python -m orrg convert` builds the same geometry, since the
config accepts any `(f, s, g, V, k)` the planner accepts.

## Appendix C — evidence map

| claim | source |
|---|---|
| lossless conversion, 5 architectures | `tests/test_orrg.py`; v3 §4 |
| D1–D5 wiring, default-preserving | `tests/test_geometry.py`; 47-test suite |
| conversion KL 3.9e-10 on real weights | `reports/P1-RUNG.md` |
| no loss win at 82k / 819k / 2.72M tokens | `reports/P1-RUNG.md`, `P1-LONG.md`, `P2-EPOCHS.md` |
| diversity and magnitude are inert levers | `P1-ORTHO.md`, `P1-MAGNITUDE.md`, `P1-INTERACTION.md` |
| dense overfits; endpoint is a degradation regime | `P2-EPOCHS.md` |
| exact copies ≥ perturbed; S3 hurts; tied capacity removes the gain | `reports/JAX-TOY.md` §mechanism controls |
| cluster warm-start matters (+0.064) | `reports/JAX-TOY.md` |
| ORRG-A −0.008…−0.019 vs naive, capacity-bound regime | `reports/JAX-TOY.md` |
| 9B/27B geometry and budgets | `orrg/scale.py`, Appendix B |
| new geometry is wired and default-preserving | `tests/test_geometry.py`, 47-test suite |
