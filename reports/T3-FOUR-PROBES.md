# T3 — Four probes: serving, clustering, residual dynamics, conversion debt

Four independent measurements run on 2026-11-13 to price the next stage. Each is
a single harness with a JSON artifact; none of them trains at scale.

## 1. Does "A2B" cash out in wall-clock? (`experiments/serving_throughput.py`)

RTX 5060 Ti 16 GB, bf16, `s4 g4 v14`, f=1, hidden 768 / inter 3072 / 8 layers.
Seed 81.8M → ORRG 450.2M total, 110.4M active at k=1 (5.50×).

The first pass over this question timed only the reference (python) dispatch and
quoted the seed, the converted model, dense-at-active and dense-at-total. §1b
re-measures the same geometry with **both** dispatches in one run, so no figure from
that pass is quoted here any more; the seed arm is not part of the current sweep and
its old numbers are retired rather than restated.

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

Same geometry (ORRG 450.2M total / 110.4M active): seven arms, plus each fused arm
re-measured with `BATCHED_BELOW_ROWS` forced to 0, **three repeats each after a
discarded warm-up sweep**. Cells are median [min–max] tok/s, ratios are taken per
repeat then summarised, and every table below is generated from
`experiments/results/serving_throughput.json` by `experiments/render_report.py`
rather than typed.

<!-- BEGIN serving-table -->
Same weights, both dispatches: `*_batched` forces the batched path and
`*_grouped` forces grouped, so every cell below is a real comparison. The
shipped arm (`fused_k1`, `fused_k2`) runs the gate and is what serving does;
`MIN_BATCHED_ROWS = 64` and `BATCHED_BELOW_ROWS` = k=1: 10,752, k=2: 6,912 packed rows, measured per k (§1f).

| metric | fused_k1 | python_k1 | dense_active_k1 | fused_k2 | python_k2 | dense_active_k2 | dense_total |
|---|---|---|---|---|---|---|---|
| prefill_b1 | 2.9k [2.6k–3.0k] | 689 [666–747] | 9.0k [8.1k–9.7k] | 3.1k [2.8k–3.1k] | 357 [329–380] | 9.5k [7.7k–10.0k] | 9.6k [6.3k–10.1k] |
| prefill_b8 | 14.2k [14.1k–16.3k] | 4.2k [3.9k–4.4k] | 76.6k [73.2k–76.8k] | 15.5k [14.8k–15.9k] | 2.2k [2.1k–2.3k] | 70.7k [60.1k–78.8k] | 29.3k [29.2k–30.1k] |
| prefill_b64 | 65.3k [64.6k–65.9k] | 27.3k [26.2k–29.1k] | 112.2k [110.5k–113.5k] | 46.8k [46.3k–47.5k] | 15.5k [15.2k–16.1k] | 90.2k [88.4k–90.6k] | 32.7k [32.6k–33.2k] |
| decode_b1 | 21 [21–22] | 12 [11–12] | 60 [51–61] | 23 [22–23] | 8 [8–9] | 59 [53–64] | 57 [51–61] |
| decode_b8 | 172 [143–175] | 50 [48–50] | 453 [450–507] | 186 [176–194] | 29 [29–30] | 467 [451–484] | 457 [444–470] |
| decode_b64 | 1.4k [1.3k–1.4k] | 252 [247–257] | 4.0k [3.4k–4.1k] | 1.4k [1.4k–1.5k] | 139 [134–144] | 3.6k [3.1k–4.0k] | 3.7k [3.3k–3.8k] |

| metric | k=1 batched ÷ grouped (throughput, >1 = batched faster) | k=1 serving | k=2 batched ÷ grouped (throughput) | k=2 serving |
|---|---|---|---|---|
| prefill_b1 | 1.25 [1.18–1.27] | batched (1,536 rows) | 1.32 [1.26–1.65] | batched (2,048 rows) |
| prefill_b8 | 1.05 [0.96–1.31] | grouped (12,288 rows) | 0.55 [0.52–0.69] | grouped (16,384 rows) |
| prefill_b64 | 0.26 [0.25–0.26] | grouped (98,304 rows) | 0.20 [0.19–0.20] | grouped (131,072 rows) |
| decode_b1 | 0.94 [0.90–1.03] | grouped (12 rows) | 1.12 [1.08–1.14] | grouped (16 rows) |
| decode_b8 | 1.32 [1.28–1.48] | batched (96 rows) | 1.42 [1.39–1.43] | batched (128 rows) |
| decode_b64 | 1.50 [1.50–1.68] | batched (768 rows) | 1.59 [1.57–1.59] | batched (1,024 rows) |

| ratio | k2/k1, batched | k2/k1, grouped | k2/k1, python | k2/k1, dense-active |
|---|---|---|---|---|
| prefill_b1 | 1.01 | 0.82 | 0.54 | 1.03 |
| prefill_b8 | 0.58 | 0.98 | 0.53 | 0.92 |
| prefill_b64 | 0.55 | 0.72 | 0.57 | 0.79 |
| decode_b1 | 1.10 | 0.94 | 0.69 | 1.03 |
| decode_b8 | 1.02 | 0.91 | 0.59 | 1.00 |
| decode_b64 | 1.04 | 0.99 | 0.55 | 0.96 |

| fused ÷ python | k=1 | k=2 |
|---|---|---|
| prefill_b1 | 3.97× | 8.60× |
| prefill_b8 | 3.41× | 6.97× |
| prefill_b64 | 2.36× | 3.03× |
| decode_b1 | 1.85× | 2.72× |
| decode_b8 | 3.48× | 6.43× |
| decode_b64 | 5.46× | 10.19× |

| fused ÷ dense-active | k=1 | k=2 |
|---|---|---|
| prefill_b1 | 0.31× | 0.33× |
| prefill_b8 | 0.19× | 0.21× |
| prefill_b64 | 0.58× | 0.53× |
| decode_b1 | 0.37× | 0.40× |
| decode_b8 | 0.35× | 0.40× |
| decode_b64 | 0.35× | 0.40× |

| fused ÷ dense-total | k=1 | k=2 |
|---|---|---|
| prefill_b1 | 0.30× | 0.30× |
| prefill_b8 | 0.48× | 0.52× |
| prefill_b64 | 1.99× | 1.43× |
| decode_b1 | 0.37× | 0.38× |
| decode_b8 | 0.37× | 0.41× |
| decode_b64 | 0.37× | 0.39× |
<!-- END serving-table -->

The spread is uneven and this box is shared: the widest per-repeat ratio in this run is
**0.67**, on prefill b8, where one batched repeat stalled; the smallest cells move
40% between repeats (prefill b1 2.6k–3.6k). Two things keep that from corrupting the
ratios. The batched and grouped arms alternate inside the *same repeat and the same
batch*, so a clock ramp or an external stall hits both; and arm order alone used to
contribute 0.12 of drift when the twins sat at opposite ends of the sweep, against
0.05 for a paired-arm control. `render_report.py` uses that 0.05 plus a point of slack
as the decision floor, prints both it and the widest spread, and fails the build if a
shape the gate sends to batched loses — or one it sends to grouped wins — by more than
the floor on the median of three repeats. Cells whose range is wider than the effect
are quoted as ranges below, not as points.

**Verdict: the prefill gap was mostly implementation, and so was half of the decode
gap.** Fused dispatch beats the reference path by 1.8–5.4× at k=1 and 2.9–10.2× at
k=2, and at batch 64 prefill it *beats* the dense model with the same total
parameters by 2.00× (k=2: 1.43×). Against the dense model with the same **active**
parameters — the comparison the A2B label actually promises — it reaches 0.58× at
b64 prefill and 0.34× at b64 decode (k=2: 0.52× and 0.40×).

Those decode figures are 1.32–1.59× higher than the same weights with both arms on
the same shape (§1b, §1f), so the earlier reading — that the decode shortfall was purely
structural — does not survive the measurement: half of it was per-segment launches.
What remains is structural: at 64 sequences a step routes ~4.6 rows per segment, and
56 tiny segments per layer is utilisation-bound no matter the kernel. Closing the
rest needs CUDA graphs, larger routing granularity (fewer, wider experts), or
cross-layer fusion.

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

The k2/k1 ratios for every path — batched, grouped, python and dense-at-active —
are in the generated table at §1b; the columns below the absolute one are that table.
The three-GEMM column was measured before the N=2I fold and is retained only as the
historical arm.

**The decode penalty was launch overhead.** At b64 decode the second expert now
costs nothing measurable — 1.05 under the batched path, 0.99 under the grouped one,
against a 0.80 arithmetic ideal and a 0.96 dense control. A 25.6% compute increase
cannot legitimately read 1.05; the saving is the removed kernel per layer plus the
removed second `[M, I]` materialisation, which dominated a decode step that is
already tiny per layer.

**The cost sits at large-batch prefill, and that cell is the least stable in the
study.** The shipped path reads 0.72 at b64 prefill in this run (repeat range
0.71–0.73, a tight run) and the two previous runs read 0.70 and 0.72 on the same shape,
so the direction is settled even though the value has moved with machine load. b8 reads
1.10 and b1 1.03 in the same run. Row arithmetic predicts 1/1.256 = 0.80 for compute
alone and 1/1.33 = 0.75 once the second expert's rows are counted per group. Forcing
that shape onto the batched path gives **0.20** instead of the arithmetic ideal — the
bandwidth cost §1f measures, not a thrash: it read 0.02 before the row budget bounded
the block. §1d's
structure probe prices the remaining choice between one N=2I GEMM and two N=I ones;
those rows are measured by that probe directly, not through `orrg.dispatch`, so
`BATCHED_BELOW_ROWS` does not affect them.

Absolute: fused k=2 reaches 0.52× of dense-at-active at b64 prefill (k=1: 0.58×)
and 1.43× of dense-at-total (k=1: 2.00×). Fused ÷ python at k=2 is 8.6× / 7.0× /
3.0× on prefill b1/b8/b64 and 2.9× / 6.4× / 10.2× on decode — against 4.2× / 3.4× /
2.4× and 1.8× / 3.5× / 5.4× at k=1, so the implementation gap widens with k as well.

On this harness +25.6% active parameters cost 10% at b8 prefill (1.10), 28% at b64
prefill (0.72 in this run, 0.70 and 0.72 in the two before it) and nothing at b64
decode (1.05). For the real 9B (+0.578B, 2.124 → 2.702B
active) the transferable part is the shape of that curve, not the magnitudes: the
real model's k=2 throughput is not measured.

The reference path still loses 45% (k2/k1 0.54 at prefill b1, 0.55 at decode b64),
because doubling the slots doubles its host syncs and per-segment launches.
12 segments per layer at k=1, 16 at k=2.

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

### 1f. Small batches: two batched matmuls instead of 2 x 60 segments

`experiments/batched_dispatch.py`. `_grouped_mm` pays ~30 µs per segment (§1e), so a
decode step issues ~1,000 segment launches. Between `MIN_BATCHED_ROWS = 64` and the
per-k bound (`BATCHED_BELOW_ROWS`, 10,752 rows at k=1 and 6,912 at k=2) the dispatch
runs `torch.bmm` twice instead:
shard segments are uniform, so they are one batched matmul over the stacked shard
weights, and residual segments are gathered into a padded block that is walked in
pieces, batched, and scattered back with the padding lanes masked out. No single piece
exceeds `BATCHED_ROW_BUDGET = 8_192` rows (`chunk_plan`), so the padded extent is
bounded whatever the token count; a segment longer than the budget is tiled along its
lanes. The probe below
calls the library's own `_grouped` and `_batched` on identical `_prepare` output and
asserts they agree, so it prices exactly the choice `path_for` makes — not a prototype
of it.

<!-- BEGIN crossover-table -->
Layer-level, a bare layer on random inputs: the last column is the path the
shipped gate takes at that size, which is not the same question (§1f).

| tokens | packed rows | grouped | batched | batched ÷ grouped | shipped path |
|---|---|---|---|---|---|
| 8 | 96 | 2.94 ms | 1.79 ms | 0.61 | batched |
| 64 | 768 | 4.25 ms | 1.82 ms | 0.43 | batched |
| 512 | 6144 | 4.42 ms | 4.32 ms | 0.98 | batched |
| 1024 | 12288 | 4.49 ms | 7.29 ms | 1.62 | grouped |
| 2048 | 24576 | 4.90 ms | 14.20 ms | 2.90 | grouped |
| 4096 | 49152 | 5.63 ms | 27.70 ms | 4.92 | grouped |
| 8192 | 98304 | 8.92 ms | 57.39 ms | 6.43 | grouped |
<!-- END crossover-table -->

That probe prices one layer in isolation, which is not the question the gate asks. The
band around the crossing is measured on the full model instead, by
`experiments/gate_band.py`, with the same diagnostic `force_path` override:

<!-- BEGIN gate-band -->
Full model, one sequence of `tokens` for each row count; `force_path` pins each
arm on the same weights, three repeats with the order flipped every other
repeat, warm-up discarded. `capacity` is the padded block the batched path
allocates per segment — the quantity that actually bounds the win.

| k | tokens | packed rows | capacity | grouped ms | batched ms | batched ÷ grouped (time, <1 = batched faster) | serving |
|---|---|---|---|---|---|---|---|
| 1 | 128 | 1,536 | 128 | 67.7 | 41.4 | 0.61 | batched |
| 1 | 192 | 2,304 | 192 | 70.4 | 57.0 | 0.81 | batched |
| 1 | 256 | 3,072 | 256 | 67.3 | 54.9 | 0.82 | batched |
| 1 | 320 | 3,840 | 320 | 60.6 | 53.9 | 0.89 | batched |
| 1 | 384 | 4,608 | 384 | 57.4 | 46.9 | 0.82 | batched |
| 1 | 448 | 5,376 | 448 | 60.5 | 48.8 | 0.81 | batched |
| 1 | 512 | 6,144 | 512 | 65.0 | 58.6 | 0.90 | batched |
| 1 | 576 | 6,912 | 576 | 62.7 | 54.4 | 0.87 | batched |
| 1 | 640 | 7,680 | 640 | 66.8 | 60.2 | 0.90 | batched |
| 1 | 672 | 8,064 | 672 | 57.3 | 58.1 | 1.01 | batched |
| 1 | 704 | 8,448 | 704 | 60.2 | 55.5 | 0.92 | batched |
| 1 | 736 | 8,832 | 736 | 75.0 | 55.3 | 0.74 | batched |
| 1 | 768 | 9,216 | 768 | 65.3 | 62.3 | 0.95 | batched |
| 1 | 832 | 9,984 | 832 | 65.7 | 59.6 | 0.91 | batched |
| 1 | 896 | 10,752 | 896 | 75.1 | 62.1 | 0.83 | batched |
| 1 | 960 | 11,520 | 960 | 60.9 | 64.2 | 1.05 | grouped |
| 1 | 1024 | 12,288 | 1,024 | 62.9 | 66.2 | 1.05 | grouped |
| 1 | 1088 | 13,056 | 1,088 | 94.5 | 86.2 | 0.91 | grouped |
| 1 | 1152 | 13,824 | 1,152 | 69.9 | 78.3 | 1.12 | grouped |
| 1 | 1200 | 14,400 | 1,200 | 75.9 | 93.8 | 1.24 | grouped |
| 2 | 96 | 1,536 | 192 | 58.8 | 44.1 | 0.75 | batched |
| 2 | 144 | 2,304 | 288 | 59.3 | 45.9 | 0.77 | batched |
| 2 | 192 | 3,072 | 384 | 76.9 | 73.9 | 0.96 | batched |
| 2 | 240 | 3,840 | 480 | 73.8 | 57.6 | 0.78 | batched |
| 2 | 288 | 4,608 | 576 | 65.9 | 52.8 | 0.80 | batched |
| 2 | 336 | 5,376 | 672 | 63.1 | 61.6 | 0.98 | batched |
| 2 | 384 | 6,144 | 768 | 79.7 | 71.8 | 0.90 | batched |
| 2 | 432 | 6,912 | 864 | 75.4 | 64.4 | 0.85 | batched |
| 2 | 480 | 7,680 | 960 | 61.9 | 71.4 | 1.15 | grouped |
| 2 | 504 | 8,064 | 1,008 | 71.3 | 70.3 | 0.99 | grouped |
| 2 | 528 | 8,448 | 1,056 | 70.1 | 79.0 | 1.13 | grouped |
| 2 | 552 | 8,832 | 1,104 | 73.1 | 72.4 | 0.99 | grouped |
| 2 | 576 | 9,216 | 1,152 | 80.8 | 85.8 | 1.06 | grouped |
| 2 | 624 | 9,984 | 1,248 | 72.5 | 88.7 | 1.22 | grouped |
| 2 | 672 | 10,752 | 1,344 | 80.0 | 87.4 | 1.09 | grouped |
| 2 | 720 | 11,520 | 1,440 | 72.5 | 92.0 | 1.27 | grouped |
| 2 | 768 | 12,288 | 1,536 | 73.3 | 90.2 | 1.23 | grouped |
| 2 | 816 | 13,056 | 1,632 | 70.1 | 101.2 | 1.44 | grouped |
| 2 | 864 | 13,824 | 1,728 | 64.3 | 99.8 | 1.55 | grouped |
| 2 | 900 | 14,400 | 1,800 | 73.9 | 107.8 | 1.46 | grouped |
<!-- END gate-band -->

The capacity must be `tokens * k`: one (group, variant) segment can hold every routed
row of its group, so sizing it from the mean drops rows exactly when the router is
peaked — the peaked-router test caught that. That bound is also the path's cost: the
block holds `segments / groups` = 13x the rows that exist, so the gather, the `[M, 2I]`
gate/up product and the `silu` temporary all scale with `segments * tokens * k` instead
of with the routing.

**The block is bounded by a row budget now, and the budget is what the memory numbers
changed.** Before, the whole `[52, tokens*k, 1536]` block was materialised at once;
`experiments/probe_vs_model.py` measured a peak of **15.4 GB** at prefill b64 k=2
against 6.6 GB for grouped, and the batched path ran **78x** slower there as the card
thrashed. Walking the block in pieces of at most 8,192 padded rows brings that peak to
**6.8 GB** against 6.6 GB, and the same cell to **5.1x** — the thrash is gone, and what
is left is the 13x memory traffic, which a row budget cannot remove. At prefill b8 the
peak went 2.8 → 2.4 GB (k=1, grouped 1.8 GB) and 5.5 → 4.5 GB (k=2, grouped 4.0 GB).

**The probe used to disagree with the model by ~3x, and the probe was wrong.** Its
earlier version timed `layer(x)` — routing, `_prepare` and dispatch — against a
stand-alone `batched_forward` prototype, so the two arms did different work. At 24,576
rows that grouped arm read 15.60 ms where the library's own `_grouped` on the same
`_prepare` output reads 4.90 ms: the baseline was inflated 3.2x, which is the ~3x gap,
and the crossing was placed at “24.6k–49.2k rows” instead of where it is. Both arms now
call the shipped methods, and the crossing sits between 6,144 rows (0.98) and 12,288
rows (1.62) — the same place the model-level sweep puts it, since prefill b8 measures
12,288 and 16,384 rows and both lose. The probe had been measuring a quantity the gate
does not depend on.

**The crossing moved once the grid was complete, and it moved in two directions at
once.** The earlier sweeps measured only three row counts at both k; the cells they left
single-k were exactly the ones that straddle the knee, and they hid the split. With every
row count from 1,536 to 14,400 measured at both k — 40 cells, same protocol — k=1 wins at
every count through 10,752 (0.61–0.90×, 0.74× at 8,832), is a tie at 11,520 and 12,288
(1.05× both) and loses from 13,824 (1.12×, 1.24× at 14,400), while k=2 wins through 6,912
(0.75–0.98×) and loses from 7,680 (1.15×, 1.13× at 8,448, 1.22× at 9,984, 1.46× at
14,400). The two knees sit 3,840 rows apart — at 9,984 rows k=1 reads 0.91× while k=2
reads 1.22× — so one row bound cannot be right for both k, and the 8,192 it used to be
was too permissive for k=2 and too tight for k=1 at the same time.

`BATCHED_BELOW_ROWS` is therefore per k, `{1: 10_752, 2: 6_912}`, and `path_for` takes
the k of the layer's config. The k=1 bound rises 2,560 rows over the old single value and
the k=2 bound falls 1,280.

What the grid cannot settle is k=1's upper edge. Its knee lies between 10,752 (0.83×, the
last measured win) and 13,824 (1.12×, the first clean loss); inside that span 11,520 and
12,288 measure 1.05× — ties inside the floor, so they buy nothing — and 13,056 measures
0.91× with repeats spanning 0.80–1.33×. More resolution would not settle it: that cell is
not under-sampled, it is bimodal on this shared card and its repeats straddle the floor
rather than sitting beyond it, which is the case the audit refuses to decide. The bound
therefore stops at the last row count with a measured win rather than at the last cell
whose median happens to be below 1.

The previous pass shipped 4,096 and called it “conservative rather than tight” only
because this band was unmeasured. Once measured, rows 4,608–7,680 turned out to win by
0.62–0.75×, so the old bound was declining 25–38% on every 3k–8k-row shape. The change
is a correction of a stale bound, not a re-tune in the other direction: at 32,768 rows,
what an even earlier pass used, prefill b8 lost 12% at k=1 and 45% at k=2 (kept as
`experiments/results/serving_throughput_threshold_32768.json`).

The lower bound is the weaker claim of the two: decode b1 packs 12 and 16 rows and
measures 0.94× and 1.14× on this sweep and has read 0.86–1.14× across four sweeps, which
is noise at that size. Nothing is won there either way, so `MIN_BATCHED_ROWS = 64` keeps
the cheaper path and avoids a padded block that only costs memory at that size. The
padded block is what makes the upper bound necessary, and the block is now bounded
too (§1f): `experiments/probe_vs_model.py` measures peak allocation on the full model
at prefill b8 of 2.4 GB against 1.8 GB (k=1) and 4.5 GB against 4.0 GB (k=2), and at
prefill b64 of 2.7 GB against 2.4 GB (k=1) and 6.8 GB against 6.7 GB (k=2). Before
the row budget those last two read 7.0 GB against 2.3 GB and 15.4 GB against 6.6 GB,
where the card has 16 GB and the batched path thrashed.

End to end the gain is larger than the layer-level ratio, because the decode step is
CPU-bound on segment launches. The per-shape table in §1b is that measurement on the
full harness model, with `fused_k1_batched` against `fused_k1_grouped` on identical
weights: **decode b8 gains 1.32× and b64 1.50× at k=1** (1.42× and 1.59× at k=2), and
**prefill b1 gains 1.25×** (1.32× at k=2), which is most of the batched region the gate
allows. Above it the comparison turns against the path — prefill b8 reads 0.55 at k=2
(the k=1 shape there is a 1.05× tie), prefill b64 reads 0.26 and 0.20 — and those cells
are bandwidth, not allocation: the k=2 b64 cell read 0.01 before the row budget and reads
0.20 now, so bounding the block removed the thrash and left the 13× traffic that sets the
knee.

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
