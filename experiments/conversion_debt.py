"""Conversion-debt curve: what does shrinking the skeleton cost on a real model?

Runs the existing verification harness (dense vs converted logits, per-layer carve
error) at ``skeleton_scale`` 1.0 / 0.9 / 0.75 / 0.5 on the cached Qwen2.5-1.5B at
the 9B geometry, so the 27B-A6B KD budget has a measured debt curve instead of an
assumption.

    HF_HUB_OFFLINE=1 PYTHONPATH=. python experiments/conversion_debt.py
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from orrg import MoEConfig, convert_model

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SHARED, GROUPS, VARIANTS = 4, 4, 14
SCALES = (1.0, 0.9, 0.75, 0.5)

TEXT = """
A mixture of experts layer replaces a single feed-forward network with a set of
smaller ones plus a router that decides which of them each token uses. The appeal
is straightforward: a dense model spends the same compute on every token, while a
sparse model can hold far more parameters than it evaluates. The difficulty is
that capacity is only useful if the router learns to send tokens to the experts
that handle them well, and that is a chicken and egg problem: good routing needs
trained experts, and trained experts need good routing. The usual answer is to
train from scratch with a load balancing loss, which means the dense model you
already own cannot simply be widened into a sparse one.
"""


def run(scale: float, tokenizer, probe) -> dict:
    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    model.eval()
    cfg = MoEConfig(shared=SHARED, groups=GROUPS, variants=VARIANTS, skeleton_scale=scale)
    _, report = convert_model(model, cfg, verify=True, probe=probe)
    carve = [entry.get("carve_error") for entry in report.per_layer]
    out = {
        "skeleton_scale": scale,
        "total_params": report.total_params,
        "active_k1": report.active_params(1),
        "lossless_label": report.lossless,
        "logit_kl": report.logit_kl,
        "logit_max_abs": report.logit_max_abs,
        "max_carve_error": report.max_carve_error,
        "first_layer_carve": carve[0],
        "last_layer_carve": carve[-1],
        "median_carve": sorted(carve)[len(carve) // 2],
        "seconds": time.time() - started,
    }
    print(
        f"f={scale:<4} total={report.total_params/1e9:5.2f}B "
        f"kl={report.logit_kl:.3e} logitmax={report.logit_max_abs:.3f} "
        f"carve med={out['median_carve']:.2e} first={out['first_layer_carve']:.2e} "
        f"last={out['last_layer_carve']:.2e} ({out['seconds']:.0f}s)",
        flush=True,
    )
    del model, report
    gc.collect()
    return out


def main() -> int:
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    probe = tokenizer(TEXT, return_tensors="pt", truncation=True, max_length=192)
    rows = [run(scale, tokenizer, probe) for scale in SCALES]
    out = Path("experiments/results/conversion_debt.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": MODEL, "geometry": {
        "shared": SHARED, "groups": GROUPS, "variants": VARIANTS,
        "expert_mode": "additive", "expert_width_ratio": 1.0,
    }, "rows": rows}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
