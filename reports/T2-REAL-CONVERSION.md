# T2 — Real-checkpoint conversion: Qwen2.5-1.5B → 9.06B

The 9B geometry in `spec/ORRG-SPEC-v4.md` §2.7 was arithmetic from `orrg scale`.
This report runs it through the actual cached checkpoint and records what
conversion costs.

```
seed        Qwen/Qwen2.5-1.5B-Instruct (1,543,714,304 params, bf16, CPU)
geometry    shared=4 groups=4 variants=14, additive, f=1, ratio=1
            contiguous shards, router warm start = cluster
host        64 GB RAM, RTX 5060 Ti 16 GB, Win 11; 18.1 GB converted model does
            not fit on the card, so the forward pass is CPU bfloat16
```

## Result

| quantity | measured | arithmetic prediction |
|---|---|---|
| converted total | **9,060,309,936** (9.060B) | 9.06B |
| active at k=0 | 1.546B | 1.54B |
| active at k=1 | **2.124B** | 2.12B |
| active at k=2 | 2.702B | 2.70B |
| expansion | 5.87× | 5.87× |
| routers (always on) | 2,238,720 | — |
| shard width | 1120 neurons (8 shards over 8960) | 1120 |
| `lossless` label | true | true |

Conversion cost, probe = 192 tokens of real text:

| metric | value |
|---|---|
| conversion KL (converted vs dense logits) | **1.324e-3** nats |
| logit max abs diff | 1.062 (bf16) |
| block carve error, layer 0 | 5.43e-3 |
| block carve error, layer 27 (max) | **4.99e-2** |

Router warm start (k-means, 14 centroids, 512 calibration tokens, per layer):

| | value |
|---|---|
| agreement mean over 28 layers | **0.988** |
| min / max | 0.967 / 0.998 |

Wall clock: 79 s convert+verify (155 s on the first, unoptimised run), 1.1 s to
load the seed.

## Is "lossless" true?

The construction is exact; the measured error at bf16 is dtype rounding. The
control in `experiments/check_split_numerics.py` splits one real Qwen2.5 MLP into
the same 8 contiguous shards *without* ORRG and compares against the dense block:

| computation | relative max error |
|---|---|
| 8 shards summed, bf16 | 2.72e-3 |
| 8 shards summed, fp32 | **3.31e-7** |
| bf16 dense block vs fp32 dense block | 2.66e-3 |

Splitting costs exactly what bf16 representation costs (2.72e-3 vs 2.66e-3), and
the split error disappears in fp32 (3.31e-7). So `A(k)=N0+k·r·F` and
"variant 0 reproduces the dense block" hold; what the 9.06B model reports is the
price of doing 8 GEMMs in bf16 instead of 1.

The carve error is **not** a per-layer defect. `_capture_outputs` compares the
dense run against the converted run, so layer *i* is evaluated on inputs that
have already drifted through *i-1* converted layers. That produces a plateau, not
a monotone climb:

```
layer  0-5   2.8e-3 .. 7.4e-3      dtype floor, no accumulated drift yet
layer  6-23  ~1.2e-2 .. 3.0e-2     plateau; per-block cost is 2.7e-3, so this is
                                   dense-vs-converted forward divergence
layer 24-26  9.7e-3, 1.2e-2, 1.2e-3  noisy: the metric divides by max|ref| for
                                   that layer's output
layer 27     5.0e-2                worst; final block before the LM head
```

Layer 0 sits at the control's 2.7e-3, and every block's own contribution is the
2.7e-3 measured above. The plateau is the cost of comparing two different forward
passes, not 28 independent conversion errors.

## Two defects this run found

1. **bf16 crashed outright.** `contribution.index_add_` rejected float32 router
   weights against a bf16 residual sum. Every conversion test used fp32, so the
   suite could not see it. The router's probabilities are now cast to the block
   dtype; `test_conversion_works_in_bfloat16` covers additive and replace.
2. **Shard sums were accumulated in bf16.** Eight partial sums, each rounded to
   bf16 before being added, cost ~1e-2 relative per block. The accumulator is now
   float32 and the block is cast back once at the end: carve error fell
   7.90e-2 → 4.99e-2 and KL 1.438e-3 → 1.324e-3. The remaining error is the
   per-shard down-projection rounding, which no accumulator can remove without
   running the down projection itself in fp32.

## Caveats

* One 192-token probe, one text: the KL is a point estimate, not a distribution.
* CPU bf16 only; the GPU cannot hold an 18 GB model, so no CUDA or
  `torch.compile` path is exercised.
* Warm-start "agreement" is agreement with the k-means labels **on the same
  calibration tokens** the centroids were fit on. It is a sanity check that the
  nearest-centroid init took, not held-out routing accuracy.
* Contiguous shards, no neuron clustering: experts are copies, so this measures
  fidelity, not usefulness.
* Residuals are zero-init and untrained. This is exactly the state where the
  lossless guarantee applies; after any training step the guarantee is about the
  skeleton, and the residual contribution is learned.

## Re-check at k=2

The first run above reports k=1. `top_residuals` never reached the router, so any
stored "k=2" row was routing one expert. `MoEConfig` now makes the router follow
`top_residuals` (a conflicting `RouterConfig` is corrected at construction), and
the script takes `--k` and `--residual-init`. Same checkpoint, same probe, CPU
bfloat16, both runs after the fix:

| residual_init | k | router k | logit KL | max abs diff | max carve error | active |
|---|---|---|---|---|---|---|
| zero | 1 | 1 | 1.3237e-3 | 1.0625 | 4.99e-2 | 2.124 B |
| zero | 2 | 2 | 1.3237e-3 | 1.0625 | 4.99e-2 | 2.702 B |
| copy | 1 | 1 | 7.196e-1 | 11.53 | 1.58 | 2.124 B |
| copy | 2 | 2 | 1.164 | 14.03 | 1.47 | 2.702 B |

**With zero-init residuals the fidelity is invariant to k.** Routing one or two
experts both add exactly zero, so the conversion is exact for any k and the KL
above stands as a k=2 number too; only the active count moves, 2.124 B → 2.702 B
(+0.578 B). The earlier k=2 label was wrong about *routing and cost*, not about
fidelity.

**With copy-init residuals the k=2 run is no longer vacuous.** KL rises from 0.720
to 1.164 and max|diff| from 11.5 to 14.0 when the second expert is actually
evaluated, which is what proves the router now selects two experts per group
rather than one. `lossless` is False on both copy rows, as it should be.

The serving numbers in `reports/T3-FOUR-PROBES.md` are k=1 measurements
(`active_1`), matching the dense baseline they are compared against; they are not
relabelled here.

## Reproduce

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=. python experiments/verify_9b_qwen.py --k 1
HF_HUB_OFFLINE=1 PYTHONPATH=. python experiments/verify_9b_qwen.py --k 2
HF_HUB_OFFLINE=1 PYTHONPATH=. python experiments/verify_9b_qwen.py --k 2 --residual-init copy
HF_HUB_OFFLINE=1 PYTHONPATH=. python experiments/check_split_numerics.py
```

Outputs: `experiments/results/verify_9b_qwen.json`,
`experiments/results/verify_9b_qwen_k2.json`,
`experiments/results/verify_9b_qwen_copy.json`,
`experiments/results/verify_9b_qwen_copy_k2.json`,
`experiments/results/check_split_numerics.json`, `logs/verify_9b_qwen.log`,
`logs/check_split_numerics.log`.
