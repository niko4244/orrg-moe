# T3 — Four probes: serving, clustering, residual dynamics, conversion debt

Four independent measurements run on 2026-11-13 to price the next stage. Each is
a single harness with a JSON artifact; none of them trains at scale.

## 1. Does "A2B" cash out in wall-clock? (`experiments/serving_throughput.py`)

RTX 5060 Ti 16 GB, bf16, `s4 g4 v14`, f=1, hidden 768 / inter 3072 / 8 layers.
Seed 81.8M → ORRG 450.2M total, 110.4M active at k=1 (5.50×). Dispatch is the
current unfused path (`index_add_` per selected variant).

| model | params | prefill b1 / b8 / b64 (k tok/s) | decode b1 / b8 / b64 (k tok/s) |
|---|---|---|---|
| ORRG @k=1 | 450.2M | 0.7 / 4.6 / 29.2 | 0.010 / 0.048 / 0.27 |
| dense, same active | 110.4M | 7.3 / 78.7 / 122.0 | 0.060 / 0.53 / 3.79 |
| dense, same total | 450.2M | 7.8 / 31.7 / 36.2 | 0.070 / 0.51 / 4.15 |
| dense seed | 81.8M | 8.8 / 83.0 / 155.3 | 0.070 / 0.45 / 3.83 |

Ratios, ORRG ÷ dense with the same **active** params: prefill 0.10× / 0.06× /
0.24×; decode 0.22× / 0.10× / 0.07×.
Ratios, ORRG ÷ dense with the same **total** params: prefill 0.09× / 0.15× /
0.81×; decode 0.14× / 0.10× / 0.065×.

### 1b. How much of that is the implementation? (`orrg/dispatch.py`)

The fused path is now library code: `MoEConfig(dispatch="fused")`, default
`"python"`. It replaces the per-segment Python loop with one sorted token list and
three `torch._grouped_mm` calls per layer (gate, up, down), skeleton shards as
always-full segments, and zero-padded stacks so uneven (clustered) shard widths
still fuse. `test_fused_dispatch_matches_reference_forward` asserts it reproduces
the reference forward in fp32 and bf16 and raises when autograd is enabled;
`test_fused_dispatch_follows_the_routing_map` repeats the check with
`residual_init="copy"`, pairwise-distinct residual weights and a k=2 router, so a
wrong segment id, group offset or scale gather cannot hide behind zeroed
residuals. Four injected dispatch bugs (dropped group offset, misaligned scales,
shifted gather, dropped residual scale) each fail that test.

Same geometry (ORRG 450.2M total / 110.4M active), four models, **three repeats
each after a discarded warm-up sweep**; cells are median [min–max] tok/s, ratios
are taken per repeat then summarised.

| metric | orrg_fused | orrg_python | dense_active | dense_total |
|---|---|---|---|---|
| prefill b1 | 2.1k [1.7k–2.2k] | 690 [621–738] | 9.6k [9.4k–10.4k] | 9.7k [7.9k–9.8k] |
| prefill b8 | 12.8k [12.2k–14.2k] | 4.1k [3.9k–4.2k] | 79.4k [76.8k–81.2k] | 31.7k [30.1k–32.3k] |
| prefill b64 | 58.7k [54.2k–65.1k] | 27.6k [22.8k–30.3k] | 123.5k [122.4k–123.5k] | 36.0k [35.9k–36.2k] |
| decode b1 | 22 [21–22] | 12 [9–12] | 63 [62–65] | 61 [51–65] |
| decode b8 | 131 [116–135] | 49 [46–54] | 511 [489–513] | 467 [460–519] |
| decode b64 | 795 [683–811] | 268 [261–269] | 4.2k [4.1k–4.2k] | 4.0k [3.0k–4.1k] |

| metric | fused ÷ python | fused ÷ dense_active | fused ÷ dense_total |
|---|---|---|---|
| prefill b1 | 2.92× [2.42–3.40] | 0.20× [0.18–0.22] | 0.22× |
| prefill b8 | 3.13× [3.13–3.36] | 0.16× [0.15–0.19] | 0.40× |
| prefill b64 | 2.13× [1.79–2.86] | 0.48× [0.44–0.53] | **1.63×** |
| decode b1 | 1.82× [1.80–2.37] | 0.34× [0.34–0.35] | 0.36× |
| decode b8 | 2.50× [2.35–2.85] | 0.26× [0.23–0.27] | 0.28× |
| decode b64 | 2.97× [2.62–3.01] | 0.19× [0.16–0.20] | 0.20× |

Run-to-run spread is 4–20% on the fused arm (worst cell: b64 prefill, 54.2k–65.1k)
and under 5% on both dense arms, so the ratio brackets above are meaningful but
any single-shot number from the earlier harness over-read differences under 10%.

**Verdict: the prefill gap was mostly implementation; the decode gap is not.**
Fused dispatch is 1.8–3.4× faster than the reference path everywhere, and at batch
64 prefill it *beats* the dense model with the same total parameters by 1.63×
[1.49–1.81]. But against the dense model with the same **active** parameters — the
comparison the A2B label actually promises — it reaches only 0.48× at b64 prefill
and 0.19× at b64 decode. Roughly two-thirds of the prefill shortfall was dispatch;
the decode shortfall is structural: at 64 sequences a step routes ~4.6 rows per
segment, and 56 tiny segments per layer is launch- and utilisation-bound no matter
the kernel. Closing that needs CUDA graphs, larger routing granularity (fewer,
wider experts), or cross-layer fusion.

## 2. Clustered vs contiguous shards (`experiments/cluster_shards_real.py`)

Qwen2.5-1.5B, `s4 g4 v14`, f=1, 512 calibration tokens, k-means over post-gate
activations.

| arm | cohesion | random baseline | conversion KL | max carve | group-output cosine |
|---|---|---|---|---|---|
| contiguous | — | — | 1.73e-3 | 4.99e-2 | 0.040 (0.414 max) |
| clustered | **0.345** | 0.085 | 1.30e-3 | 6.36e-2 | 0.037 (0.652 max) |

**Structure exists.** Clustering finds groups 4× more coherent than random
assignment, and conversion stays exact (KL ~1e-3, same bf16 floor as contiguous).
So clustered shards are not worse on fidelity, and they do capture real neuron
structure.

### 1c. What the second expert costs (k=2)

`MoEConfig` now makes the router follow `top_residuals`, so k=2 really routes two
residual experts per group. Seven models at the same geometry — fused, reference
and dense-at-active rebuilt per k, one dense-at-total — three repeats after a
discarded warm-up. Active: k=1 110.4M → k=2 138.7M (+28.3M, +25.6%), so a purely
compute-bound second expert should read 1/1.256 = **0.80×**.

The fused path then went from **three** `torch._grouped_mm` calls per layer to
**two**: gate and up are stacked along N as one N=2I GEMM. Both stacks also share
one padded width, which is what makes `expert_width_ratio < 1` work (it could not
concatenate shard and residual segments before).

| metric | fused k2/k1, 3 GEMMs | fused k2/k1, 2 GEMMs | python k2/k1 | dense-active k2/k1 |
|---|---|---|---|---|
| prefill b64 | 0.82 [0.77–0.85] | **0.70** [0.70–0.73] | 0.55 | 0.82 |
| prefill b8 | 0.90 [0.86–0.97] | 0.90 [0.37–0.97] | 0.48 | 0.95 |
| prefill b1 | 0.89 [0.70–0.95] | 0.91 [0.43–0.95] | 0.34 | 1.11 |
| decode b64 | 0.72 [0.60–0.87] | **1.01** [0.99–1.03] | 0.56 | 1.07 |
| decode b8 | 0.84 [0.71–0.88] | 0.89 [0.73–0.90] | 0.64 | 1.04 |
| decode b1 | 0.88 [0.73–0.92] | 0.89 [0.79–0.95] | 0.36 | 1.01 |

**The decode penalty was launch overhead, and one fewer kernel per layer removes
it.** At b64 decode the second expert now costs nothing measurable (1.01, against
a 0.80 arithmetic ideal and a 1.07 dense control), where it previously cost 28%.
A 25.6% compute increase cannot legitimately read 1.01 — the saving is the removed
`_grouped_mm` call plus the removed second `[M, I]` materialisation, which
dominated a decode step that is already tiny per layer.

**The cost moved to large-batch prefill: 0.72 at b64, down from 0.82 — and §1d
shows that is the GEMM structure, not the extra expert's arithmetic.** Measured
against each other at the same batch, the k2/k1 prefill b64 ratio is **0.72 with
the fused N=2I path and 0.87 with two N=I GEMMs**; the historical 0.82 came from
the three-GEMM path. Row arithmetic alone predicts 1/1.33 = 0.75, so the split
arm clears it (more rows per segment tile better) and the fused arm misses it by
15 points because its `[M, 2I]` intermediate is 201 MB at 131k rows. b8 is
unchanged (0.90) and b1 slightly better (0.91). Re-run of the same seven-model sweep on the shipped build (the size branch of
§1d was removed after it failed at real widths) reproduces the ratio at **0.70
[0.70–0.73]** and keeps decode b64 at 1.01 [0.95–1.02], so nothing in the table
below §1c is stale.

Absolute: fused k=2 reaches 0.51× of dense-at-active at b64 prefill (k=1: 0.53×)
and 1.41× of dense-at-total (k=1: 1.79×). Fused ÷ python at k=2 is 5.8× / 6.8× /
3.2× on prefill b1/b8/b64 and 2.4× / 4.5× / 7.0× on decode — against 2.3–3.2×
at k=1 in table 1b, so the implementation gap widened as well.

On this harness +25.6% active parameters cost ~9% at b8 prefill, ~28% at b64
prefill and nothing at b64 decode. For the real 9B (+0.578B, 2.124 → 2.702B
active) the transferable part is the shape of that curve, not the magnitudes: the
real model's k=2 throughput is not measured.

The reference path still loses 45–65%, because doubling the slots doubles its host
syncs and per-segment launches. 12 segments per layer at k=1, 16 at k=2.

### 1d. The harness crossover does not transfer to real widths: no size branch

The layer probe below measures **k=1 only**, and its largest shape (98k rows) has a
151 MB `[M, 2I]` intermediate — just short of the knee, which is why it read as
"the fused form always wins". The width is the stacked 2 x per-segment width (768
here), not 2 x the model's FFN width: conversion splits the FFN across shards and
variants. `experiments/prefill_structure.py` measures both
structures in the full model at both k, with the arms alternated and the split arm
asserted equal to the library (identical logits, max|diff| 0):

| k | batch | packed rows | intermediate | one N=2I | two N=I | two ÷ one |
|---|---|---|---|---|---|---|
| 1 | 16 | 24,576 | 38 MB | 26,613 | 23,739 | 0.89 |
| 1 | 32 | 49,152 | 75 MB | 55,726 | 44,483 | 0.80 |
| 1 | 48 | 73,728 | 113 MB | 69,223 | 57,017 | 0.82 |
| 1 | 64 | 98,304 | 151 MB | 71,472 | 65,822 | 0.92 |
| 1 | 96 | 147,456 | 226 MB | 68,211 | 69,206 | **1.01** |
| 2 | 16 | 32,768 | 50 MB | 26,740 | 21,445 | 0.80 |
| 2 | 32 | 65,536 | 101 MB | 47,225 | 40,771 | 0.86 |
| 2 | 48 | 98,304 | 151 MB | 51,532 | 53,078 | **1.03** |
| 2 | 64 | 131,072 | 201 MB | 51,509 | 54,821 | **1.06** |
| 2 | 96 | 196,608 | 302 MB | 57,238 | 58,297 | **1.02** |

The k2/k1 column reconciles the history: fused path 0.72 at b64, split 0.83,
against the 0.82 the three-GEMM run measured. On **this** geometry the fused form
wins by 10–20% below ~150 MB and the split by 1–6% from ~200 MB, and the k=2
shapes turn over earlier than the k=1 ones (151 MB: 1.03 against 0.92).

**At real widths the sign flips, so the branch was removed.**
`experiments/real_threshold.py` repeats the comparison on Qwen2.5-1.5B's own
weights and widths (segment 1120, gate/up output 2240, 60 segments), on 4 of its
28 layers because the 9.06B conversion needs ~18 GB in bf16 and does not fit a
16 GB card. Prefill throughput, batch 8–48 at seq 128:

| batch | packed rows | intermediate | one N=2240 | two N=1120 | two ÷ one |
|---|---|---|---|---|---|
| 8 | 12,288 | 55 MB | 20,995 | 19,835 | 0.94 |
| 16 | 24,576 | 110 MB | 28,641 | 27,249 | 0.95 |
| 32 | 49,152 | 220 MB | 30,887 | 29,717 | 0.96 |
| 48 | 73,728 | 330 MB | 30,775 | 30,001 | 0.97 |

The GEMM alone was measured the same way and agrees in sign: 0.98 at 55 MB,
1.08 at 220 MB and 1.18 at 440 MB, i.e. the split loses at real widths where it
won on the harness at the same bytes. A threshold in bytes is therefore a
property of the harness's 384-wide segments, not of the method: 1120-wide real
segments hit different tile shapes. The split arm reproduces the layer exactly
(max|diff| 0 over four layers), so the change is safe; it simply is not faster,
and `orrg/dispatch.py` keeps the single N=2I GEMM with a comment recording why.

`experiments/dispatch_crossover.py` times a single MoE layer with the two gate/up
GEMM structures interleaved, so ordering cannot bias either arm:

| tokens | packed rows | one N=2I | two N=I | two ÷ one |
|---|---|---|---|---|
| 16 | 192 | 4.34 ms | 5.68 ms | 1.31 |
| 64 | 768 | 5.12 ms | 6.93 ms | 1.35 |
| 512 | 6,144 | 5.17 ms | 7.18 ms | 1.39 |
| 2,048 | 24,576 | 5.94 ms | 7.59 ms | 1.28 |
| 4,096 | 49,152 | 7.05 ms | 9.39 ms | 1.33 |
| 8,192 | 98,304 | 11.11 ms | 12.99 ms | 1.17 |

**At k=1 up to 98k rows there is no crossover**: the N=2I form is 20–39% faster at
every size measured here, which is what the layer probe saw. That range stops at
the knee, so the table above (full model, both k) is the one to read for the
branch question.

**Skeleton replication.** Eight of the twelve rows per token are the skeleton's
`flat.repeat(n_shards, 1)`. Replacing them with one dense FFN over concatenated
shard weights — identical arithmetic, no replication, eight fewer segments —
saves:

| tokens | library forward | dense-skeleton | saved | `repeat()` alone |
|---|---|---|---|---|
| 64 | 5.35 ms | 5.02 ms | 6.2% | 0.088 ms |
| 512 | 5.21 ms | 5.10 ms | 2.1% | 0.091 ms |
| 2,048 | 6.10 ms | 5.64 ms | 7.5% | 0.093 ms |

Run-to-run spread on the batch-64 figure is 0.9–6.2%, so read it as "a few
percent", not as 6.2%. The copy itself is 1.7% of a decode call; the rest is the row-multiplied GEMM
shape — eight N=768 segments instead of one N=3072. So the 8x replication accounts
for about 6% of a batch-64 decode layer and cannot explain the remaining gap:
fused k=2 decode sits at 0.25× of dense-at-active, which a perfect skeleton would
move to 0.27× at best. The gap is the per-segment cost inside `_grouped_mm` (§1e),
not the gather/argsort/`index_add` scaffolding. Nor is
it implemented: the dense skeleton needs an extra copy of the skeleton weights,
which on the real 9B is ~3 GB for a 6% win on this card.

### 1e. CUDA graphs: blocked by `_grouped_mm`, worth ~5% of the gap where they apply

Prototype: `experiments/cuda_graph_decode.py`.

**Capture of the dispatch is impossible as written.** `torch._grouped_mm` reads
`offs` back to the host to build its tile schedule, so under capture it raises
`Cannot copy between CPU and CUDA tensors during CUDA graph capture`; passing a
CPU-resident `offs` is rejected by the op instead (`Expected all tensors to be on
the same device`). Both failures are reproduced by the script.

A prerequisite bug fell out of this. `torch.bincount(..., minlength=...)` also
syncs — it reads the maximum index to size its output — at 209 µs against 74 µs
for a fixed-size `scatter_add_`, at every row count. `orrg/dispatch.py` now uses
`scatter_add_`: 135 µs per layer, ~1.1 ms per 8-layer step off the eager path,
and the change that makes the prologue capturable at all.

**What a graph can cover: the prologue.** Router, argsort, segment and row
build, counts and the gather, at 64 tokens: 0.998 ms eager → 0.080 ms replayed
(92%); at 512 tokens 0.925 → 0.135 ms.

**What it cannot: 70% of the layer.** Budget at 64 tokens — total 5.44 ms =
prologue 0.95 + two `_grouped_mm` 3.79 + epilogue 0.70.

**And the grouped GEMM is schedule-bound, not arithmetic-bound.** One call over
52 segments:

| rows/segment | total rows | `_grouped_mm` | plain GEMM | ratio |
|---|---|---|---|---|
| 4 | 224 | 1.68 ms | 0.09 ms | 18.7× |
| 8 | 448 | 1.74 ms | 0.08 ms | 22.4× |
| 16 | 896 | 2.32 ms | 0.11 ms | 21.7× |
| 32 | 1,792 | 1.77 ms | 0.10 ms | 17.3× |
| 64 | 3,584 | 1.68 ms | 0.15 ms | 11.3× |
| 128 | 7,168 | 1.74 ms | 0.24 ms | 7.3× |

Flat in rows: ~30 µs per segment whatever the rows, so 60 segments × 2 calls ×
8 layers ≈ 1,000 segment overheads ≈ 30 ms of a 61 ms decode step. Replaying a
graph cannot elide it, because the sync is inside the op's launcher.

**End-to-end, graphing the capturable part and leaving the GEMMs eager:**

| batch | eager | prologue-graphed | dense-at-active | graph ÷ dense | eager ÷ dense |
|---|---|---|---|---|---|
| 8 | 135 [131–139] | 171 [159–197] | 516 | 0.33 | 0.26 |
| 64 | 958 [885–1,031] | 1,172 [1,042–1,267] | 4,033 | 0.29 | 0.24 |

Medians of three alternating runs per arm; individual b64 runs ranged from +5%
to +22% because capture reallocates the memory pool, so the honest figure is
+15% and 0.25 → 0.29× — consistent with the prologue being 17.5% of a MoE layer
and MoE 70% of a decode step.

So graphs do close some of the gap, but little of it: 0.25 → 0.29× at b64 is
about 5% of the distance to parity (0.26 → 0.33× at b8, ~9%). The rest is the
per-segment launch cost of `_grouped_mm`, and
the fix is a kernel that walks all segments in one launch (a Triton grouped GEMM,
or an ATen path that takes a device-resident schedule), or fewer segments: 52 of
the 60 per layer are residual experts, one per (group, variant). Neither is
reachable by capturing the current dispatch.

### 2b. Routing accuracy instead of cosine

The group-output cosine saturated (0.037 clustered vs 0.040 contiguous) because
random 1120-column slices of an 8960-wide sum are near-orthogonal regardless of
structure, so it could not discriminate. Replaced with a held-out linear probe:
label each token by the group whose base contributes the largest norm, fit a
ridge classifier on one document, score another.

| arm | probe accuracy | chance (1/4) | majority baseline |
|---|---|---|---|
| contiguous | 0.344 | 0.25 | 0.38 |
| clustered | 0.321 | 0.25 | 0.55 |

**Neither arm has routing signal at group granularity.** Both probes sit at or
below the majority baseline, i.e. the hidden state does not predict which group's
neurons matter for a token — knowing the hidden state buys nothing over always
naming the commonest group. Clustering creates real *neuron-level* structure
(cohesion 0.345 vs 0.085 random) but not *token-level* routing structure, and it
makes the contribution label much more imbalanced (majority 0.38 → 0.55), so a
router trained on these groups would collapse toward one or two of them.

So the cosine was not hiding a signal: there is nothing at group level to
specialise into yet. Residual targets remain one per group (56 experts share 4
targets per layer), and the label distribution over groups is nearly flat in
difficulty terms. Any expert-level specialisation would have to come from the
orthogonality push, not from the targets.

## 3. Can a zero-init residual leave zero? (`experiments/residual_dynamics.py`)

Tiny converted model (hidden 128, inter 512, 2 layers), 150 steps, 5 arms. Step-0
gradient mass from the task loss, then residual output norm relative to its group
base and cosine to the base.

| arm | grad gate/up | grad down | grad router | norm@0 → @25 → @150 | cos@150 |
|---|---|---|---|---|---|
| zero | **0.0** | 45.3 | 0.0 | 0.00 → 0.95 → 1.04 | +0.31 |
| scaled | 4.47 | 45.2 | 0.038 | 0.05 → 1.05 → 1.10 | +0.30 |
| noise | 0.0 | 45.3 | 0.0 | 0.00 → 0.99 → 1.10 | +0.27 |
| distill | 0.0 | 45.3 | 0.0 | 0.00 → 0.87 → 0.66 | +0.33 |
| ortho | 0.0 | 45.3 | 0.0 | 0.00 → 1.90 → 1.26 | +0.27 |

**The "dead residual" theory is false.** The gate/up gradient is exactly zero at
step 0, as predicted, and so is the task-loss router gradient — but the
**down-projection gradient is not** (45.3): `dL/dW_down = dL/dout ⊗ y` does not
depend on `W_down` being nonzero. One Adam step moves `W_down` off zero, and from
step 2 everything trains. The zero arm reaches norm 0.95 by step 25 on its own,
indistinguishable from `scaled` and `noise`. No symmetry breaker is needed; the
optimizer does it. `distill` ends smaller (0.66) and `ortho` overshoots (1.90)
then settles.

## 4. Conversion-debt curve (`experiments/conversion_debt.py`)

Qwen2.5-1.5B, `s4 g4 v14`, 192-token probe, bf16.

| `f` | total | conversion KL | logit max diff | carve median | carve first / last |
|---|---|---|---|---|---|
| 1.00 | 9.06B | 1.39e-3 | 1.06 | 1.5e-2 | 2.7e-3 / 5.0e-2 |
| 0.90 | 8.19B | **4.31e-1** | 15.9 | 3.6e-1 | 2.0e-1 / 7.9e-1 |
| 0.75 | 6.89B | **1.16** | 18.3 | 6.2e-1 | 2.6e-1 / 6.2e-1 |
| 0.50 | 4.73B | **3.18** | 18.9 | 8.1e-1 | 4.1e-1 / 7.1e-1 |

**The debt is a cliff, not a curve.** Dropping 10% of the FFN neurons multiplies
KL by ~310 (1.4e-3 → 0.43 nats/token); half the neurons costs 3.18 nats, worse
than a uniform-distribution predictor. There is no cheap `f<1` regime: 27B-A6B is
not a KD-recovery job on top of a converted checkpoint, it is a pretraining job
with a warm start. §2.5 should be priced accordingly, and `f=0.9` is not a
"gentle" setting.

## What this changes

* Serving: fused dispatch recovers 2–3× and makes large-batch prefill better than
  dense-at-total, but at matched active params the model is still 0.54× (prefill)
  and 0.21× (decode). Stop quoting A2B as a latency claim; it is a parameter
  count on this code path.
* Clustering: free on fidelity, real neuron structure, but no token-level routing
  signal and a much more imbalanced label — keep it for fidelity, not for
  specialisation, and expect router collapse if trained as-is.
* Residual init: the default is fine; the symmetry problem was misdiagnosed, so
  `orthogonal` machinery is optional rather than load-bearing.
* Debt: the KD track needs a pretraining-scale budget, not a recovery-scale one.

Artifacts: `experiments/results/serving_throughput.json`,
`cluster_shards_real.json`, `residual_dynamics.json`, `conversion_debt.json`.
