# P1 long rung: does router distillation or residual diversity separate?

The 80-step rung (`reports/P1-RUNG.md`) showed a balance advantage for router
distillation (imbalance 0.250 → 0.177) but no loss difference, and left residual
diversity at ~0 with the caveat that 82k tokens is far too short to test it. This
rung multiplies the budget by ten and traces the mechanism every 100 steps.

## Protocol

| item | value |
|---|---|
| seed model | `HuggingFaceTB/SmolLM2-360M` (361.8M, 32 layers, hidden 960, intermediate 2560, vocab 49152) |
| corpus | WikiText-103 raw from the HF cache: 1.40M tokens, 1.33M train / 70k validation |
| batch | 4 sequences × 256 tokens |
| steps | **800 per arm = 819k tokens** (10× the 80-step rung) |
| seeds | 1 |
| optimizer | Adafactor, OneCycle over 800 steps, lr 2e-4, bf16 autocast |
| shape | `shared=2, groups=4, variants=3, k=1` → 676.7M total, 519.4M active |
| arms | `dense_continued`, `orrg_all` (task-loss-only routing), `orrg_distill` (`coef=0.1`, 128 target tokens/step) |
| isolation | one OS process per arm, so allocator fragmentation cannot skew a later arm |
| trace | every 100 steps: val loss, load imbalance, expert diversity, residual output share, distill CE |
| hardware | one 17.1 GB consumer GPU, 12.9–15.8 GB used, 3.42–3.45 s/step for the MoE arms |

The trace is the point of this run. "Load imbalance" is `max(util) - mean(util)`
over residuals; "diversity" is mean pairwise `1 - cos` between residual
up-projection weights inside a group (0 = copies); "residual share" is the sum of
residual output norms over the layer output norm, averaged over layers (exactly 0
at conversion, because the down-projections are zeroed).

## End of run

| arm | val before | val after | Δval | Δ vs dense | imbalance | diversity | residual share | distill CE | s/step |
|---|---|---|---|---|---|---|---|---|---|
| dense continued | 3.0257 | 2.8204 | −0.2053 | – | – | – | – | – | 0.45 |
| ORRG all | 3.0258 | 2.8289 | −0.1969 | **+0.0085** | 0.183 | 0.0004 | 4.18% | – | 3.42 |
| ORRG distill | 3.0258 | 2.8275 | −0.1984 | **+0.0071** | 0.195 | 0.0004 | 4.10% | 0.668 | 3.45 |

Imbalance and diversity are the step-800 trace values (last training forward).
The end-of-run collection, taken on the eval batches, reads 0.203 / 0.208 for
imbalance and counts **0 dead experts for `orrg_all` and 7 for `orrg_distill`**
(out of 256 residuals, summed over 32 layers).

## Trajectories

Validation loss:

| step | dense | ORRG all | ORRG distill |
|---|---|---|---|
| 0 | 3.0257 | 3.0258 | 3.0258 |
| 100 | 2.8296 | 2.8344 | 2.8350 |
| 200 | 2.8529 | 2.8596 | 2.8617 |
| 300 | 2.8614 | 2.8722 | 2.8730 |
| 400 | 2.8477 | 2.8590 | 2.8588 |
| 500 | 2.8332 | 2.8424 | 2.8421 |
| 600 | 2.8274 | 2.8366 | 2.8357 |
| 700 | 2.8199 | 2.8283 | 2.8267 |
| 800 | 2.8204 | 2.8289 | 2.8275 |

The two MoE arms track each other to within 0.002 at every point, and both run
0.007–0.012 behind dense throughout.

Load imbalance:

| step | ORRG all | ORRG distill |
|---|---|---|
| 0 | 0.2917 | 0.2830 |
| 100 | 0.1886 | 0.2027 |
| 200 | 0.1517 | 0.2028 |
| 300 | **0.1351** | 0.1891 |
| 400 | 0.1643 | 0.2041 |
| 500 | 0.1769 | 0.1969 |
| 600 | 0.1734 | 0.1935 |
| 700 | 0.1688 | 0.1945 |
| 800 | 0.1825 | 0.1954 |

Expert diversity and residual share (identical to four decimals between arms at
every trace point):

| step | diversity | residual share |
|---|---|---|
| 0 | 0.0000 | 0.0% |
| 100 | 0.0001 | 2.3% |
| 200 | 0.0002 | 3.2% |
| 300 | 0.0003 | 3.7% |
| 400 | 0.0004 | 4.0% |
| 500 | 0.0004 | 4.1% |
| 600 | 0.0004 | 4.1% |
| 700 | 0.0004 | 4.2% |
| 800 | 0.0004 | 4.2% |

Distillation CE (ceiling ln 2 = 0.6931):

| step | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 |
|---|---|---|---|---|---|---|---|---|
| CE | 0.6830 | 0.6773 | 0.6732 | 0.6704 | 0.6673 | 0.6667 | 0.6679 | 0.6678 |

## What this says

**1. Neither mechanism separates from task-loss-only routing over 10× the
budget.** Distill ends 0.0014 nats better than `orrg_all` (2.8275 vs 2.8289) —
smaller than the 0.007–0.009 gap both arms have to dense, and measured with a
single seed. Diversity is identical to four decimals at every trace point, and
the residual share differs by 0.08 percentage points.

**2. The 80-step balance advantage reverses.** At 80 steps distillation cut
imbalance from 0.250 to 0.177 and looked like a routing win. At 800 steps it is
slightly *worse* (0.195 vs 0.183 on the trace; 0.208 vs 0.203 at end of run) and
leaves 7 dead experts where `orrg_all` has none. The early advantage was a
transient of the short schedule, not a property of the mechanism.

**3. The contribution target is too flat to teach the router much.** Its CE moves
from 0.683 to 0.668 against a ceiling of 0.6931 — about 3.6% of the available
range in 819k tokens. The target is computed from residual output norms, and
those norms stay nearly identical across the two residuals in a group, so the
softmax over them stays near-uniform and the router gets a weak, nearly
symmetric signal.

**4. The extra capacity is mostly inert.** Residual share rises to ~4% of the
layer output and then saturates (4.14% at step 500 → 4.18% at 800). Up to that
point the residuals are small additive corrections; after it they stop growing.
Diversity correspondingly plateaus at 0.0004 — the two residuals per group remain
near-copies of each other and of their base. Under the metric used here, ORRG at
this scale is behaving like dense plus a small residual adapter, not like a
mixture that specializes.

**5. The 80-step absolute numbers were schedule-defined.** Dense reaches 2.7480
after 80 steps but 2.8204 after 800. The 80-step run anneals lr to ~0 inside 80
steps; the long run spends far longer at high lr, and 819k tokens on a 1.33M-token
train split is enough to start fitting it (train loss ~2.4 against val 2.82).
Paired arm comparisons at matched steps remain valid — all arms share the
schedule — but absolute losses should not be quoted across rungs.

**6. Distillation is now cheap.** In isolated processes it costs 3.45 s/step
against 3.42 for the same arm without it — about 1% overhead at
`distill_tokens=128`. The 2× overhead seen in the 80-step run was allocator
fragmentation, not the target pass.

## Interpretation

The zeroed down-projection is the bottleneck. A residual's output starts at
exactly zero, and its up-projection can only move through the down-projection
that is growing from zero. Output share therefore grows quickly to a few percent
and then saturates, which has two consequences at once: the residuals stay too
similar for diversity to develop, and the contribution target stays too uniform
for distillation to be a strong teacher. Both mechanisms need residuals to become
*different*, and nothing in this recipe supplies a pressure to differentiate —
they are initialized identical, receive the same task gradient, and the router's
choice is the only asymmetry.

Three plausible ways out, none tested here: a longer horizon (though diversity is
flat between step 400 and 800, so the trend is not encouraging), wider residuals
so a few percent of share is a bigger absolute signal, or an explicit
diversification/orthogonality term on residuals. A sharper contribution target
(target temperature < 1) would at least give distillation a stronger signal, at
the risk of chasing noise in near-identical norms.

## Threats to validity

* **One seed.** The 0.0014-nat distill difference and the 0.0085 / 0.0071 gaps to
  dense are not statistically resolved. The diversity and residual-share results
  are structural and reproduce across arms, so they are the more robust part.
* **One horizon, one model, one corpus.** 819k tokens ≈ 0.6 epoch of the train
  split; validation loss bottoms around step 300 and then rises, so this is a
  mildly overfitting regime, not a scaling regime.
* **Absolute losses are schedule-specific** (finding 5). Only matched-step
  comparisons are meaningful.
* **Diversity is one particular definition** (cosine between up-projection
  weights). Function-level or activation-level specialization could look
  different even when weight cosine is ~0.
* **`k=1` of two residuals per group** with 76.8% of width active: this is a
  capacity-growth configuration, not a sparse or compute-saving one.
* **Dead experts** are counted on one eval forward, so 7 vs 0 is indicative, not
  a stable measurement.
* End-of-run imbalance (eval batches) and trace imbalance (training forward)
  differ by ~0.02; both are reported rather than picking the flattering one.

## Reproduce

```bash
for arm in dense_continued orrg_all orrg_distill; do
  PYTHONPATH=. python experiments/p1_real.py --steps 800 --seeds 1 --batch 4 --seq 256 \
      --shared 2 --groups 4 --variants 3 --optimizer adafactor \
      --distill-coef 0.1 --distill-tokens 128 --trace-every 100 --log-every 50 \
      --arms "$arm" --out "experiments/results/p1_long_${arm}.json"
done
PYTHONPATH=. python experiments/p1_long_report.py
```

Wall clock: 6 min (dense), 46 min (`orrg_all`), 47 min (`orrg_distill`).
