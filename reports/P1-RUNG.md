# P1 rung: ORRG on a real dense checkpoint

This is the first rung that is not a synthetic toy: a real pretrained checkpoint,
real text, and the three arms the ORRG proposal asks about (naive upcycling,
ORRG frozen, ORRG fully trained), plus a clustered-shard ablation and a
router-distillation ablation.

It is a **rung, not a training run**. 80 steps × 4 × 256 tokens = 82k tokens per
arm. Everything below is directional evidence about *conversion cost and the
first 80 steps*, not about final quality.

## Protocol

| item | value |
|---|---|
| seed model | `HuggingFaceTB/SmolLM2-360M` (361.8M params, 32 layers, hidden 960, intermediate 2560, vocab 49152) |
| corpus | WikiText-103 raw, cached parquet, 1.40M tokens; 95% train / 5% validation windows |
| batch | 4 sequences × 256 tokens |
| steps | 80 per arm, identical batches across arms within a seed |
| seeds | 2 (model init, batch order) |
| optimizer | Adafactor, OneCycle, lr 2e-4, bf16 autocast (fp32 master weights) |
| ORRG shape | `shared=2, groups=4, variants=3, k=1` → 676.7M total, **519.4M active** (76.8% of width) |
| shards | 6 skeleton shards tile the 2560 intermediate axis exactly; 8 residual experts |
| hardware | one 17.1 GB consumer GPU, offline HF cache |

`k=1` at `variants=3` is a capacity-growth configuration, not a compute-saving
one: 10 of 14 shards are active. The P1 point is "does conversion cost anything
and does training recover", not FLOPs.

Arms:

* `dense_continued` — the seed, still training (361.8M trainable).
* `naive_upcycle` — residuals are full copies of their group's base shard, all
  weights trainable (the standard sparse-upcycling control).
* `orrg_frozen` — lossless residuals, skeleton frozen; only residuals + routers
  train (314.9M trainable).
* `orrg_all` — lossless residuals, everything trainable (676.7M trainable).
* `orrg_clustered` — `orrg_all` with activation-clustered shard memberships
  (calibration pass over 2 × 4 × 256 tokens, balanced k-means).
* `orrg_distill` — `orrg_all` plus contribution-target router distillation
  (`coef=0.1`, 128 target tokens per layer per step).

## Results

Mean over 2 seeds; ± is half the seed range. "Δ vs dense" is the paired
difference in final validation loss within each seed (lower is better).

| arm | val before | val after | Δval | Δ vs dense | conv KL | trainable | s/step | imbalance | dead | diversity |
|---|---|---|---|---|---|---|---|---|---|---|
| dense continued | 3.0257 | 2.7473 ± 0.0007 | −0.2784 | – | – | 361.8M | 0.44 | – | – | – |
| naive upcycling | 3.4359 ± 0.0040 | 2.8490 ± 0.0007 | −0.5869 | **+0.1017 ± 0.0015** | 3.1e-01 | 550.8M | 2.27 | 0.212 | 0.0 | 0.0000 |
| ORRG frozen | 3.0258 | 2.7573 ± 0.0018 | −0.2685 | +0.0101 ± 0.0010 | 3.9e-10 | 314.9M | 1.67 | 0.457 | 0.5 | 0.0001 |
| ORRG all | 3.0258 | 2.7488 ± 0.0008 | −0.2770 | **+0.0016 ± 0.0015** | 3.9e-10 | 676.7M | 6.03 | 0.250 | 0.0 | 0.0000 |
| ORRG clustered | 3.0257 | 2.7488 ± 0.0004 | −0.2769 | +0.0015 ± 0.0003 | 3.3e-09 | 676.5M | 4.04 | 0.244 | 0.0 | 0.0000 |
| ORRG distill | 3.0258 | 2.7482 ± 0.0000 | −0.2776 | +0.0010 ± 0.0007 | 3.9e-10 | 676.7M | 4.96 | **0.177** | 0.0 | 0.0000 |

Clustered-shard diagnostics (mean over layers): cohesion 0.225 vs 0.065 for a
random partition of the same sizes, balance 0.998 (partition sizes differ by at
most one neuron). Conversion takes 29.5 s on CPU for the 32-layer model.

Raw rows: `experiments/results/p1_real.json`; table fragment:
`experiments/results/p1_real_table.md`.

## What the numbers say

**1. The lossless start survives real weights and real text.** Conversion KL is
3.9e-10 contiguous and 3.3e-09 clustered, and `val_before` is 3.0258 against
dense's 3.0257 — a 1e-4 difference that is float-order noise, not a quality
gap. Naive upcycling pays 0.31 KL and starts **+0.41 nats worse** (3.4359 vs
3.0257) on the same data.

**2. Naive upcycling never recovers that penalty here.** It recovers 0.59 of its
0.41 conversion loss but still ends **+0.1017 ± 0.0015** behind dense, 66×
the seed spread. `orrg_all` ends **+0.0016 ± 0.0015** behind dense — inside the
noise. On this rung, lossless conversion plus co-training reproduces dense
continued training, and copy-upcycling does not.

**3. Freezing the skeleton costs ~6× more than co-training.** `orrg_frozen` is
far better than naive (+0.0101 vs +0.1017) but worse than `orrg_all` (+0.0016)
while training 47% of the parameters. Consistent with the toy run, frozen is a
memory trade, not a quality win.

**4. Activation clustering is free at this scale.** Clustered shards are
functionally identical at conversion (KL 3.3e-09) and end at 2.7488 vs 2.7488
for contiguous — a paired difference of 0.0001 ± 0.0006. The clusters are real:
cohesion 0.225 against a 0.065 random baseline, with a balanced partition. What
clustering buys is the option to later drop or specialize whole *co-activated*
neuron sets; at equal steps it costs nothing measurable.

**5. Router distillation changes routing, not (yet) loss.** With the
contribution-target term the router's load imbalance drops from **0.250 ± 0.003
to 0.177 ± 0.006** (29% lower, ~10× the seed spread) — the supervision is
reaching the router and is not a no-op. Final loss moves from 2.7488 to 2.7482
(+0.0010 vs +0.0016 behind dense), i.e. **0.0006, smaller than the seed spread**.
At 80 steps, task-loss-only routing is not beaten on quality; contribution-target
supervision buys balance, not loss. It also costs ~2× wall clock per step when
computed every step (6.30 vs 2.37 s/step on seed 0).

**6. Diversity is structurally zero at this budget and this is not a failure
signal.** `expert_diversity` compares residual up-projections. A residual's
up-projection receives gradient through its down-projection, which is exactly
zero at conversion, so diversity is 0 by construction and only grows as the
down-projection fills in (toy scale: 0.028–0.056 at 300 steps). At 80 steps it
reads 0.0000–0.0001 in every ORRG arm. No dead experts appeared (one seed of the
frozen arm had a single dead expert).

## Timing and memory, and why the s/step column is noisy

The s/step column is **not** a clean benchmark: within one process, later arms
inherited a fragmented allocator (seed 1 `orrg_all` 9.68 s/step vs seed 0's 2.37
for the identical arm). Isolated measurements at this model size:

| measurement | value |
|---|---|
| dense 361.8M, b4×256 | 0.43 s/step |
| ORRG 676.7M, b4×256, AdamW fp32 states | 15.5 GB peak → 22–41 s/step (allocator spill) |
| ORRG 676.7M, b4×256, Adafactor | **12.9 GB peak, 2.66 s/step** |
| conversion, contiguous (32 layers) | ~14 s CPU |
| conversion, clustered + k-means | ~30 s CPU |

Two changes made the rung feasible on a 17.1 GB card: Adafactor (factored second
moments instead of 8 bytes/parameter of Adam state) and a dispatch rewrite that
sorts once per group instead of scanning a boolean mask per expert (768 GPU
syncs/step → 128). The first version of this experiment measured 22.4 s/step for
`orrg_all`; the current one measures 2.66 s/step in isolation.

## Threats to validity

* **82k tokens per arm.** This bounds conversion cost and early training
  behaviour; it says nothing about where the arms end up at 1B+ tokens.
* **`k=1` at `variants=3` is not sparse** (76.8% of width active). Compute
  savings are a separate, untested claim.
* **Two seeds, one model family, one corpus.** The ORRG-vs-dense gap (+0.0016)
  is within seed spread; the naive gap (+0.1017) is not, but it is still one
  model.
* **Distillation's quality effect is unmeasured at useful horizons.** It was
  evaluated on routing balance (where it wins) and 80-step loss (where it does
  not separate).
* **No downstream evaluations.** Validation loss on WikiText only.

## Reproduce

```bash
PYTHONPATH=. python experiments/p1_real.py --steps 80 --seeds 2 --batch 4 --seq 256 \
    --shared 2 --groups 4 --variants 3 --optimizer adafactor \
    --distill-coef 0.1 --distill-tokens 128 --cluster-tokens 512 \
    --out experiments/results/p1_real.json > experiments/results/p1_real.log 2>&1
PYTHONPATH=. python experiments/p1_report.py
```

Wall clock: ~50 min on the machine above (5 of 6 arms are MoE forwards plus
backward).

## Follow-up: longer rung (819k tokens per arm)

The 80-step numbers above are schedule-defined: OneCycle anneals the learning rate
to near zero by step 80, so the dense arm's 2.7480 is not comparable to a longer
run's endpoint. At 800 steps × 4 × 256 tokens the same arms give dense 2.8204,
ORRG-all 2.8289 and ORRG-distill 2.8275. Diversity is still 0.0004, residual
output share plateaus at ~4%, and the 80-step distillation balance advantage
reverses (imbalance 0.195 vs 0.183, 7 dead experts vs 0). Paired arm comparisons
at matched steps remain valid; absolute losses do not carry across rungs. See
`reports/P1-LONG.md`.
