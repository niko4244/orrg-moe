# ORRG: lossless, model-agnostic dense → MoE conversion

ORRG turns a dense decoder-style transformer into a sparse mixture of experts
**without changing the model's function at conversion time**. It works on any
architecture whose FFN is a self-contained module, because it never
re-implements the FFN: it discovers blocks from parameter shapes and slices them
along the intermediate axis.

```python
import torch
from transformers import AutoModelForCausalLM
from orrg import MoEConfig, convert_model

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
probe = {"input_ids": torch.randint(0, 128, (1, 16))}

cfg = MoEConfig(shared=2, groups=6, variants=8, top_residuals=1)
model, report = convert_model(model, cfg, probe=probe)
print(report.summary())
# dense params    :    1.236 B
# converted total :    7.599 B
# active (k=1)    :    2.096 B
# carve error     : 3.1e-07 (relative, per layer)
# logit max |diff|: 5.9e-08
```

Plan a 6.4B → 27B-A6B shape before spending GPU time:

```bash
python -m orrg scale --hidden 4096 --inter 10240 --layers 32 --vocab 128000 \
    --heads 32 --kv-heads 8 --shared 2 --groups 6 --variants 8
```

## Why the conversion is lossless

A dense FFN is additive over its intermediate neurons:

```
FFN(x) = act(x W_up^T + b_up) W_down^T + b_down
```

ORRG splits that axis into `shared + groups` **skeleton** shards that tile it
exactly and are always active. Their sum is the original FFN, up to
floating-point order (measured: 2.5e-07 – 4.2e-07 relative error across Llama,
GPT-2, Phi, Mistral and Qwen2). Each group also owns `variants - 1` **residual**
experts: narrow copies of the group's own range whose down-projection is zeroed.
They output exactly zero at t=0, so the converted model is bit-for-bit the seed,
but each residual is free to grow in any direction instead of being a copy of a
full FFN. This removes the trade-off that defeats sparse upcycling (lossless but
identical experts) and drop-upcycling (diverse but lossy at conversion).

## Design

| Piece | What it does |
|---|---|
| `discover_ffn_modules` | Finds FFN blocks from parameter roles (`intermediate > hidden`, one projection out, one in). No per-architecture registry. Rejects attention (including GPT-2's fused qkv) and split-FFN blocks. |
| `shard.narrow_module` | Materializes a narrow copy of a block with sliced up/down weights and biases. Works for `nn.Linear` and transformers `Conv1D`. |
| `MoELayer` | Skeleton + groups + residuals; dropless token dispatch; per-group structural balance. |
| `TokenRouter` | The model-agnostic "tokenizer": maps each token's hidden state to experts, per group, in fp32. Supports grouped or flat routing, jitter, and aux-loss-free bias updates. |
| `distill` | Output KD, on-policy KD, and *router distillation* from per-expert contribution targets. |
| `metrics` | Conversion KL, utilization, load imbalance, dead experts, expert diversity, router entropy. |
| `scale` | Total vs active parameters, FLOPs/token, optimizer memory, GPU-hours. |

### Growth mode (default)

```
out = Σ shared shards + Σ group base shards + Σ selected residuals
```

The skeleton reproduces the seed. Residuals add capacity; `top_residuals` is an
**elastic serving knob**: 0 gives the exact seed, 1..V-1 trade FLOPs for
capacity, all from one checkpoint.

### Partition mode

All skeleton shards are pooled and `top_k_partition` are selected. This is the
compute-reduction (MoEfication-style) regime. It is exact only when every shard
is active; the tests assert both the exact case and that sparse selection is
approximate.

### Routing

Routing is grouped: group *g* owns one base shard and `V-1` residuals, and
exactly `top_residuals` are selected per group per token. Every group
contributes, so load balance is structural rather than a penalty that fights
specialization. `bias_update_rate > 0` enables DeepSeek-style aux-loss-free
balancing within groups.

### Router distillation

Residuals output zero at t=0, so the task loss gives the router no gradient on
the first step (the expert down-projections do get gradient and grow
immediately). `contribution_targets` computes, per group, the norm of every
residual expert's output and turns it into a target distribution;
`router_distill_loss` is the cross-entropy between the router and that target.
This is the router-level supervision the original ORRG proposal asked for,
defined in a way that is measurable.

## Verified in this repository

`python -m pytest tests/test_orrg.py -q` → **34 passed**

* Lossless conversion across 5 architectures (`carve error < 1e-5`, logits
  `< 1e-5`).
* Partition mode exact at full activation; provably not exact when sparse.
* Residuals start at exactly zero, receive gradient on step 1, and leave zero
  after one optimizer step.
* Grouped routing: per-group token counts are exactly `top_residuals × N`.
* Checkpoint round-trip is bit-identical.
* Scale accounting reproduces the 6.4B → 27B/5.9B-active plan.

## Supported / not supported

Supported: Llama, Qwen2/3, Mistral, Gemma, Phi, GPT-2, Bloom, and any model whose
FFN is one module with a widened intermediate axis.

Not supported (fails loudly): OPT-style models where `fc1`/`fc2` sit directly on
the decoder layer, so the only module containing both also owns attention.
Attention is left dense on purpose: shrinking it is a separate teacher–student
problem, and the ORRG analysis showed it cannot be carved away for free.

## Repository layout

```
orrg/            library (config, discovery, sharding, router, MoE layer, conversion,
                 distillation, metrics, scale, CLI)
tests/           pytest suite
experiments/     smoke.py (numerical checks), toy_scale.py (head-to-head run),
                 p1_real.py + p1_report.py (real-checkpoint rung),
                 p1_long_report.py (long-rung trajectories)
reports/         P1-RUNG.md, P1-LONG.md, P1-ORTHO.md, P1-MAGNITUDE.md, P1-INTERACTION.md
spec/            ORRG-SPEC-v3.md — scale-up protocol and budgets
```

## Measured head-to-head (toy scale, 2 seeds)

A 0.67M-parameter byte-level Llama seed, three synthetic domains, 300 steps per
arm after conversion:

| arm | val loss | conversion KL | expert diversity |
|---|---|---|---|
| dense continued | 1.0142 | – | – |
| naive upcycling | 1.1625 | 1.41e-01 | 0.028 |
| ORRG, frozen skeleton | 1.1668 | 0.00 | 0.056 |
| ORRG, all trainable | **1.0098** | 0.00 | 0.041 |

The lossless start is exact and free, residuals diverge faster than copied
experts, and the gain shows up under co-training. Freezing the skeleton did
**not** beat naive upcycling here — that negative result is in the spec, not
hidden. Raw numbers: `experiments/results/toy_scale.json`.

## Measured on a real checkpoint (P1 rung)

`HuggingFaceTB/SmolLM2-360M` on WikiText-103, 80 steps per arm, 2 seeds,
`shared=2, groups=4, variants=3, k=1` (676.7M total, 519.4M active):

| arm | val loss | Δ vs dense | conversion KL | trainable |
|---|---|---|---|---|
| dense continued | 2.7473 | – | – | 361.8M |
| naive upcycling | 2.8490 | +0.1017 | 3.1e-01 | 550.8M |
| ORRG frozen skeleton | 2.7573 | +0.0101 | 3.9e-10 | 314.9M |
| ORRG all trainable | 2.7488 | +0.0016 | 3.9e-10 | 676.7M |
| ORRG + clustered shards | 2.7488 | +0.0015 | 3.3e-09 | 676.5M |
| ORRG + router distillation | 2.7482 | +0.0010 | 3.9e-10 | 676.7M |

Lossless conversion holds on real weights, copy-upcycling's 0.41-nat conversion
penalty is never recovered within the rung, and co-training matches dense.
Activation clustering costs nothing measurable (cohesion 0.225 vs 0.065 random).
This is a rung, not a training run: 82k tokens per arm. Details and threats to
validity: `reports/P1-RUNG.md`.

### Longer rung (819k tokens per arm)

800 steps × 4 × 256 tokens, one seed, one process per arm, traced every 100 steps:

| arm | val after | Δ vs dense | imbalance | diversity | residual share |
|---|---|---|---|---|---|
| dense continued | 2.8204 | – | – | – | – |
| ORRG all | 2.8289 | +0.0085 | 0.183 | 0.0004 | 4.2% |
| ORRG distill | 2.8275 | +0.0071 | 0.195 | 0.0004 | 4.1% |

Neither router distillation nor residual diversity separated from task-loss-only
routing at 10× the budget: the arms end 0.0014 nats apart, diversity is identical
at every trace point, and the 80-step balance advantage of distillation reverses
(0.195 vs 0.183, with 7 dead experts vs 0). Residuals plateau at ~4% of each
layer's output and their weights stay near-copies, so the extra capacity is
largely inert here. Details: `reports/P1-LONG.md`.

### Differentiation and sharp-target rung

Two interventions against the same controls, 819k tokens per arm:

| arm | val after | diversity | residual share | imbalance | dead experts |
|---|---|---|---|---|---|
| ORRG all | 2.8289 | 0.0004 | 4.2% | 0.183 | 0 |
| ORRG distill (τ=1) | 2.8275 | 0.0004 | 4.1% | 0.195 | 7 |
| ORRG orth | 2.8277 | **0.0486** | 4.2% | 0.177 | 0 |
| ORRG sharp (τ=0.25) | 2.8366 | 0.0004 | 3.5% | 0.460 | 34 |

A symmetry-breaking init plus an explicit normalised push breaks the diversity
plateau (121×) at no loss cost, but residual output share does not move — the
extra capacity is still inert. Sharpening the contribution target makes it a real
signal (CE 0.668 → 0.536) yet destabilises routing and costs 0.009 nats. The
bottleneck is residual magnitude, not similarity or target sharpness. Details:
`reports/P1-ORTHO.md`.

### Residual-magnitude rung

| arm | val after | residual share | diversity | dead experts | conversion KL |
|---|---|---|---|---|---|
| ORRG all | 2.8289 | 4.18% | 0.0004 | 0 | 3.9e-10 |
| ORRG mag | 2.8292 | **10.88%** | 0.0004 | 0 | 1.1e-3 |

A small nonzero down-projection init (5% of the shard's function) lifts residual
output share from the 4.18% plateau to 10.88% — 2.6× — with no loss change and no
dead experts, at a conversion KL of 1.1e-3. Nearly all of the lift is present at
step 0 (9.98%) and training adds 0.9 pp, so the old 4% was the distance covered
from zero, not a ceiling. A gate or a scale alone cannot do this: with a zeroed
down-projection both are multiplied by zero. Details: `reports/P1-MAGNITUDE.md`.

### Interaction rung

| arm | val after | diversity | residual share | dead experts |
|---|---|---|---|---|
| ORRG all | 2.8289 | 0.0004 | 4.18% | 0 |
| ORRG orth | 2.8277 | **0.0486** | 4.20% | 0 |
| ORRG mag | 2.8292 | 0.0004 | **10.88%** | 0 |
| ORRG orth+mag | 2.8310 | **0.0484** | **10.91%** | 0 |

Running both levers together composes additively: diversity lands where the orth
arm put it (99.6%) and residual share where the magnitude arm put it (100.3%),
with no dead experts and a 0.002-nat loss cost. Direction and magnitude are
independent controls, and neither — nor both — improves loss at this budget.
Details: `reports/P1-INTERACTION.md`.

## Honest limitations

The toy-scale experiment is a method smoke test, not evidence about 9B models.
No 9B or 27B checkpoint was trained here; the spec gives the gated protocol,
budgets and kill criteria for doing that. On the real 360M checkpoint, two rungs
(82k and 819k tokens per arm, one seed at the long rung) bound conversion cost and
early training behaviour, not final quality — and neither showed router
distillation or residual specialization separating from task-loss-only routing.
A follow-up rung forced diversity to 0.0486 with an explicit pressure (share
unchanged), and a further rung raised residual share from 4.18% to 10.88% with a
scaled init (loss unchanged); running both together composes additively (0.0484
diversity, 10.91% share) and still leaves loss unchanged. Neither capacity axis
is binding at this budget.
