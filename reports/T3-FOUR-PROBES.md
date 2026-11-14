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
`MIN_BATCHED_ROWS = 64`, `BATCHED_BELOW_ROWS = 8,192`.

| metric | fused_k1 | python_k1 | dense_active_k1 | fused_k2 | python_k2 | dense_active_k2 | dense_total |
|---|---|---|---|---|---|---|---|
| prefill_b1 | 2.9k [2.6k–3.6k] | 660 [572–762] | 9.3k [8.0k–9.6k] | 3.0k [2.2k–3.4k] | 334 [331–349] | 8.9k [6.8k–9.8k] | 8.2k [5.3k–9.4k] |
| prefill_b8 | 12.9k [9.2k–15.6k] | 3.8k [3.4k–3.9k] | 66.7k [61.6k–71.1k] | 15.7k [14.6k–16.3k] | 2.1k [2.1k–2.1k] | 71.0k [66.7k–76.0k] | 29.0k [29.0k–31.0k] |
| prefill_b64 | 65.0k [24.6k–67.2k] | 25.4k [24.1k–26.7k] | 111.7k [109.4k–112.1k] | 46.5k [46.2k–47.0k] | 15.1k [13.9k–15.3k] | 90.7k [90.5k–93.0k] | 33.0k [32.7k–33.2k] |
| decode_b1 | 21 [19–24] | 10 [10–12] | 56 [54–61] | 21 [16–22] | 7 [7–8] | 55 [54–60] | 55 [54–57] |
| decode_b8 | 177 [174–201] | 48 [45–49] | 472 [303–482] | 185 [160–203] | 27 [25–29] | 408 [379–449] | 438 [430–470] |
| decode_b64 | 1.3k [1.2k–1.6k] | 246 [217–253] | 3.6k [2.9k–3.8k] | 1.5k [1.3k–1.6k] | 129 [124–131] | 3.4k [3.4k–3.5k] | 3.5k [3.4k–3.6k] |

| metric | k=1 batched ÷ grouped (throughput, >1 = batched faster) | k=1 serving | k=2 batched ÷ grouped (throughput) | k=2 serving |
|---|---|---|---|---|
| prefill_b1 | 1.35 [1.21–1.41] | batched (1,536 rows) | 1.60 [1.20–1.63] | batched (2,048 rows) |
| prefill_b8 | 1.00 [0.36–1.03] | grouped (12,288 rows) | 0.56 [0.52–0.60] | grouped (16,384 rows) |
| prefill_b64 | 0.22 [0.17–0.23] | grouped (98,304 rows) | 0.01 [0.01–0.01] | grouped (131,072 rows) |
| decode_b1 | 1.04 [1.03–1.10] | grouped (12 rows) | 1.04 [1.03–1.04] | grouped (16 rows) |
| decode_b8 | 1.26 [1.19–1.52] | batched (96 rows) | 1.50 [1.48–1.62] | batched (128 rows) |
| decode_b64 | 1.49 [1.46–1.55] | batched (768 rows) | 1.47 [1.44–1.53] | batched (1,024 rows) |

| ratio | k2/k1, batched | k2/k1, grouped | k2/k1, python | k2/k1, dense-active |
|---|---|---|---|---|
| prefill_b1 | 0.98 | 0.93 | 0.51 | 0.96 |
| prefill_b8 | 0.56 | 1.01 | 0.56 | 1.07 |
| prefill_b64 | 0.02 | 0.48 | 0.60 | 0.81 |
| decode_b1 | 0.98 | 0.97 | 0.71 | 1.03 |
| decode_b8 | 1.07 | 0.86 | 0.61 | 0.87 |
| decode_b64 | 0.96 | 0.98 | 0.52 | 0.93 |

| fused ÷ python | k=1 | k=2 |
|---|---|---|
| prefill_b1 | 4.79× | 9.19× |
| prefill_b8 | 3.39× | 7.36× |
| prefill_b64 | 2.55× | 3.07× |
| decode_b1 | 2.00× | 2.60× |
| decode_b8 | 3.89× | 6.38× |
| decode_b64 | 6.00× | 11.79× |

| fused ÷ dense-active | k=1 | k=2 |
|---|---|---|
| prefill_b1 | 0.32× | 0.38× |
| prefill_b8 | 0.18× | 0.23× |
| prefill_b64 | 0.59× | 0.51× |
| decode_b1 | 0.38× | 0.38× |
| decode_b8 | 0.43× | 0.45× |
| decode_b64 | 0.44× | 0.45× |

| fused ÷ dense-total | k=1 | k=2 |
|---|---|---|
| prefill_b1 | 0.35× | 0.37× |
| prefill_b8 | 0.45× | 0.51× |
| prefill_b64 | 1.95× | 1.41× |
| decode_b1 | 0.39× | 0.39× |
| decode_b8 | 0.41× | 0.42× |
| decode_b64 | 0.36× | 0.45× |
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
gap.** Fused dispatch beats the reference path by 2.0–6.0× at k=1 and 2.6–11.8× at
k=2, and at batch 64 prefill it *beats* the dense model with the same total
parameters by 1.95× (k=2: 1.41×). Against the dense model with the same **active**
parameters — the comparison the A2B label actually promises — it reaches 0.59× at
b64 prefill and 0.44× at b64 decode (k=2: 0.51× and 0.45×).

Those decode figures are 1.26–1.50× higher than the same weights with both arms on
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
costs nothing measurable — 0.96 under the batched path, 0.98 under the grouped one,
against a 0.80 arithmetic ideal and a 0.93 dense control. A 25.6% compute increase
cannot legitimately read 0.99; the saving is the removed kernel per layer plus the
removed second `[M, I]` materialisation, which dominated a decode step that is
already tiny per layer.

**The cost sits at large-batch prefill, and that cell is the least stable in the
study.** The shipped path reads 0.48 at b64 prefill in this run (repeat range
0.44–2.05); the previous run read 0.70 on the same shape. Both are below 1, so the
direction is not in doubt, but the magnitude is not measured tightly enough to quote a
point. b8 reads 1.01 and b1 0.93 in the same run. Row arithmetic predicts
1/1.256 = 0.80 for compute alone and 1/1.33 = 0.75 once the second expert's rows are
counted per group. Forcing that shape onto the batched path gives **0.02** instead — the
memory cliff §1f measures, not an arithmetic effect. §1d's
structure probe prices the remaining choice between one N=2I GEMM and two N=I ones;
those rows are measured by that probe directly, not through `orrg.dispatch`, so
`BATCHED_BELOW_ROWS` does not affect them.

Absolute: fused k=2 reaches 0.51× of dense-at-active at b64 prefill (k=1: 0.59×)
and 1.41× of dense-at-total (k=1: 1.95×). Fused ÷ python at k=2 is 9.2× / 7.4× /
3.1× on prefill b1/b8/b64 and 2.6× / 6.4× / 11.8× on decode — against 4.8× / 3.4× /
2.6× and 2.0× / 3.9× / 6.0× at k=1, so the implementation gap widens with k as well.

On this harness +25.6% active parameters cost nothing at b8 prefill (1.01), between
30% and 52% at b64 prefill (0.70 and 0.48 in the two runs) and nothing at b64 decode
(0.96). For the real 9B (+0.578B, 2.124 → 2.702B
active) the transferable part is the shape of that curve, not the magnitudes: the
real model's k=2 throughput is not measured.

The reference path still loses 40–49% (k2/k1 0.51 at prefill b1, 0.52 at decode
b64), because doubling the slots doubles its host syncs and per-segment launches.
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
decode step issues ~1,000 segment launches. Between `MIN_BATCHED_ROWS = 64` and
`BATCHED_BELOW_ROWS = 8_192` packed rows the dispatch runs `torch.bmm` twice instead:
shard segments are uniform, so they are one batched matmul over the stacked shard
weights, and residual segments are gathered into a fixed `[segments, capacity, K]`
block, batched, and scattered back with the padding lanes masked out. The probe below
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
| 1 | 128 | 1,536 | 128 | 56.7 | 37.7 | 0.66 | batched |
| 1 | 192 | 2,304 | 192 | 66.7 | 40.8 | 0.61 | batched |
| 1 | 256 | 3,072 | 256 | 66.8 | 41.6 | 0.62 | batched |
| 1 | 320 | 3,840 | 320 | 61.9 | 38.8 | 0.63 | batched |
| 1 | 384 | 4,608 | 384 | 61.0 | 39.2 | 0.64 | batched |
| 1 | 512 | 6,144 | 512 | 58.3 | 42.5 | 0.73 | batched |
| 1 | 576 | 6,912 | 576 | 59.6 | 44.4 | 0.75 | batched |
| 1 | 640 | 7,680 | 640 | 64.0 | 53.7 | 0.84 | batched |
| 1 | 704 | 8,448 | 704 | 57.5 | 51.1 | 0.89 | grouped |
| 1 | 768 | 9,216 | 768 | 73.0 | 59.2 | 0.81 | grouped |
| 1 | 896 | 10,752 | 896 | 61.6 | 64.2 | 1.04 | grouped |
| 2 | 128 | 2,048 | 256 | 56.9 | 41.3 | 0.73 | batched |
| 2 | 192 | 3,072 | 384 | 60.0 | 39.7 | 0.66 | batched |
| 2 | 256 | 4,096 | 512 | 64.9 | 44.8 | 0.69 | batched |
| 2 | 320 | 5,120 | 640 | 64.7 | 52.4 | 0.81 | batched |
| 2 | 384 | 6,144 | 768 | 59.4 | 56.2 | 0.95 | batched |
| 2 | 512 | 8,192 | 1,024 | 68.3 | 70.1 | 1.03 | batched |
| 2 | 576 | 9,216 | 1,152 | 62.9 | 77.3 | 1.23 | grouped |
| 2 | 640 | 10,240 | 1,280 | 70.2 | 86.7 | 1.23 | grouped |
| 2 | 704 | 11,264 | 1,408 | 75.5 | 91.3 | 1.21 | grouped |
| 2 | 768 | 12,288 | 1,536 | 70.6 | 100.5 | 1.42 | grouped |
| 2 | 896 | 14,336 | 1,792 | 69.8 | 116.8 | 1.67 | grouped |
<!-- END gate-band -->

The capacity must be `tokens * k`: one (group, variant) segment can hold every routed
row of its group, so sizing it from the mean drops rows exactly when the router is
peaked — the peaked-router test caught that. That bound is also the path's cost. The
block holds `segments / groups` = 13x the rows that exist, so the gather, the `[M, 2I]`
gate/up product and the `silu` temporary all scale with `segments * tokens * k` instead
of with the routing, and the last column of the table shows where that stops paying.

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

**The gate is set by that band, and the band moved it.** Every cell up to 8,448 rows is
a win or a tie: 0.61–0.81× at 2,000–6,144 rows, 0.73–0.95× above that, and the single
k=2 point at 8,192 rows reads 1.03× — inside the 0.06 floor, i.e. a tie. From 9,216
rows k=2 loses decisively (1.23×, capacity 1,536) and k=1 follows at 10,752 (1.04×).

The crossing is therefore a band, not a point, and it arrives at a different row count
for each k because the cost scales with the padded capacity `tokens × k` while the
useful work scales with rows: k=2 crosses between 6,144 and 9,216 rows, k=1 around
9–11k. `BATCHED_BELOW_ROWS = 8_192` sits inside that band at the last row count where
the measured k=2 value is still inside the floor; 9,216 is measured and contradicted, so
the bound stops below it.

The previous pass shipped 4,096 and called it “conservative rather than tight” only
because this band was unmeasured. Once measured, rows 4,608–7,680 turned out to win by
0.62–0.75×, so the old bound was declining 25–38% on every 3k–8k-row shape. The change
is a correction of a stale bound, not a re-tune in the other direction: at 32,768 rows,
what an even earlier pass used, prefill b8 lost 12% at k=1 and 45% at k=2 (kept as
`experiments/results/serving_throughput_threshold_32768.json`).

The lower bound is the weaker claim of the two: decode b1 packs 12 and 16 rows and
measures a tie (1.05× and 1.03×, inside the 0.06 floor), having read 0.86–0.89× in the
two sweeps before it. Nothing is won there either way, so `MIN_BATCHED_ROWS = 64` keeps
the cheaper path and avoids a padded block that only costs memory at that size. The padded block is what makes the
upper bound necessary: `experiments/probe_vs_model.py` measures peak allocation on the
full model at prefill b8 of 2.8 GB against 1.8 GB (k=1) and 5.5 GB against 4.0 GB
(k=2), and at prefill b64 of 7.0 GB against 2.3 GB (k=1) and 15.4 GB against 6.6 GB
(k=2), where the card has 16 GB and the batched path thrashes.

End to end the gain is larger than the layer-level ratio, because the decode step is
CPU-bound on segment launches. The per-shape table in §1b is that measurement on the
full harness model, with `fused_k1_batched` against `fused_k1_grouped` on identical
weights: **decode b8 gains 1.26× and b64 1.49× at k=1** (1.50× and 1.47× at k=2), and
**prefill b1 gains 1.35×** (1.60× at k=2), which is the whole batched region the gate
allows. Above it the same comparison turns against the path — prefill b8 reads 1.00 and
0.56, prefill b64 reads 0.22 and 0.01 — and the k=2 b64 cell, at 50× worse than k=1 at
the same shape, is the memory cliff rather than a GEMM effect.

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
