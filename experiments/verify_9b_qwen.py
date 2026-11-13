"""Convert the real Qwen2.5-1.5B seed to the planned 9.06B geometry and measure it.

Every 9B number in ``spec/ORRG-SPEC-v4.md`` so far comes from ``orrg scale``, which
is arithmetic. This script runs the same geometry through the actual checkpoint in
the local HF cache and records what conversion really costs:

* ``logit_kl`` / ``logit_max_abs`` -- converted vs dense logits on real text
* ``max_carve_error`` -- worst relative block-output error across 28 layers
* ``cluster_agreement`` -- how often the warmed router's argmax matches the
  k-means label on the seed's own hidden states

Run on CPU in bfloat16 (the model does not fit on a 16 GB card):

    HF_HUB_OFFLINE=1 PYTHONPATH=. python experiments/verify_9b_qwen.py --k 1
    HF_HUB_OFFLINE=1 PYTHONPATH=. python experiments/verify_9b_qwen.py --k 2

``--k`` is the number of residual experts routed per group per token. Before
``MoEConfig`` enforced the invariant, the router stayed at k=1 whatever was asked
for, so any stored "k=2" number is really a k=1 measurement.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from orrg import MoEConfig, convert_model

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SHARED, GROUPS, VARIANTS = 4, 4, 14

TEXT = """
A mixture of experts layer replaces a single feed-forward network with a set of
smaller ones plus a router that decides which of them each token uses. The appeal
is straightforward. A dense model spends the same compute on every token, while a
sparse model can hold far more parameters than it evaluates, so capacity grows
without a matching growth in cost per token. The difficulty is that capacity is
only useful if the router learns to send tokens to the experts that handle them
well, and that is a chicken and egg problem: good routing needs trained experts,
and trained experts need good routing.

The usual answer is to train the whole thing from scratch with a load balancing
loss that keeps the experts busy. It works, but it means the dense model you
already have cannot simply be widened into a sparse one, because the experts have
no reason to specialise in anything until they have been trained, and by then you
have paid for the full run. What we want instead is a conversion that is exact
before any training happens, so that the sparse model starts life as a faithful
copy of the dense one and every later gain is measured against that copy.

A construction that achieves this splits the intermediate axis into contiguous
shards. One shard from each group is always active, which reproduces the dense
function exactly when the residual experts are initialised to zero. The remaining
shards become experts that add their contribution on top of the base. Because the
skeleton alone is already the dense model, the active parameter count at zero
experts equals the seed, and the total parameter count is set by how many expert
copies are present. That gives a clean two number description of the model: what
it costs to store and what it costs to run.

Nothing in that construction is specific to text, but the tests here use language
models, so the numbers depend on how the intermediate neurons of a particular
checkpoint are organised. If neurons that fire together are scattered across the
axis, contiguous shards still reproduce the dense function, but the experts are
redundant copies rather than specialists. Reordering the axis so that correlated
neurons land in the same shard is a separate step, and it is the step that decides
whether the conversion is merely correct or also useful. Correctness comes first,
because a conversion that is approximate cannot be distinguished from a training
failure later.

The router deserves a little attention too. At conversion time its weights are
random, so every token picks an arbitrary variant and the model's outputs differ
from the dense one even though each individual expert is a faithful copy of its
shard. That is not an error in the conversion; it is a statement about the router.
Warming the router up from the seed's own hidden states gives the variants a
meaning before training starts, so the first gradient steps adjust a routing that
is already sensible rather than inventing one from noise.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1, help="residual experts routed per group")
    parser.add_argument(
        "--residual-init",
        choices=["zero", "copy"],
        default="zero",
        help="zero keeps the conversion exact; copy makes routing observable",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    k = args.k

    torch.manual_seed(0)
    started = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    model.eval()
    load_s = time.time() - started

    # The probe is what the KL is measured on; keep it short enough that a CPU
    # forward of the converted model is tolerable. The calibration batch is the
    # signal the router is warmed from, so give it more tokens.
    probe = tokenizer(TEXT, return_tensors="pt", truncation=True, max_length=192)
    calibration = [tokenizer(TEXT, return_tensors="pt", truncation=True, max_length=512)]
    print(f"probe tokens      : {probe['input_ids'].shape[1]}")
    print(f"calibration tokens: {calibration[0]['input_ids'].shape[1]}")

    cfg = MoEConfig(
        shared=SHARED,
        groups=GROUPS,
        variants=VARIANTS,
        top_residuals=k,
        residual_init=args.residual_init,
        router_warm_start="cluster",
        cluster_tokens=1024,
        cluster_iters=10,
    )
    print(f"routed k          : {cfg.top_residuals} (router group_top_k={cfg.router.group_top_k})")
    t0 = time.time()
    converted, report = convert_model(
        model, cfg, verify=True, probe=probe, calibration=calibration
    )
    convert_s = time.time() - t0

    print(report.summary())
    print(f"load seconds      : {load_s:.1f}")
    print(f"convert+verify s  : {convert_s:.1f}")

    agreements = [w["cluster_agreement"] for w in report.warm_start]
    carve = [(entry["name"], entry.get("carve_error")) for entry in report.per_layer]
    worst = max(carve, key=lambda item: item[1] or 0.0)

    result = {
        "model": MODEL,
        "dtype": "bfloat16",
        "device": "cpu",
        "geometry": {
            "shared": SHARED,
            "groups": GROUPS,
            "variants": VARIANTS,
            "skeleton_scale": cfg.skeleton_scale,
            "expert_width_ratio": cfg.expert_width_ratio,
            "expert_mode": cfg.expert_mode,
            "router_warm_start": cfg.router_warm_start,
            "top_residuals": cfg.top_residuals,
            "router_group_top_k": cfg.router.group_top_k,
            "residual_init": cfg.residual_init,
        },
        "params": {
            "dense": report.dense_params,
            "total": report.total_params,
            "active_k0": report.active_params(0),
            "active_k1": report.active_params(1),
            "active_k2": report.active_params(2),
            "expansion": report.total_params / report.dense_params,
            "router": report.router_params,
            "skeleton": report.skeleton_params,
            "residual": report.residual_params,
            "shard_width": report.shard_width,
            "expert_width": report.expert_width,
        },
        "conversion": {
            "lossless_label": report.lossless,
            "logit_kl": report.logit_kl,
            "logit_max_abs": report.logit_max_abs,
            "max_carve_error": report.max_carve_error,
            "worst_layer": worst[0],
            "first_layer_error": carve[0][1],
            "last_layer_error": carve[-1][1],
            "per_layer_carve": [
                {"layer": name.split(".")[-2], "error": error} for name, error in carve
            ],
            "layers": len(report.layer_names),
        },
        "warm_start": {
            "agreement_mean": sum(agreements) / len(agreements),
            "agreement_min": min(agreements),
            "agreement_max": max(agreements),
            "layers": len(agreements),
        },
        "timing": {"load_s": load_s, "convert_s": convert_s},
    }

    stem = "experiments/results/verify_9b_qwen"
    if args.residual_init != "zero":
        stem += f"_{args.residual_init}"
    if k != 1:
        stem += f"_k{k}"
    out = Path(args.out or (stem + ".json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote             : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
