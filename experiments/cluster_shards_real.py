"""Clustered vs contiguous shards on the real Qwen2.5-1.5B.

Three questions, one run per arm:

* does activation clustering find structure -- cohesion vs the random baseline?
* does it cost conversion fidelity -- KL and carve error vs the contiguous arm?
* do the groups' outputs (the residuals' KD targets) differ enough that an expert
  has anything to specialise into?

Residual targets are the group base's output, so with `g` groups there are only
`g` distinct targets per layer no matter how many experts exist. "Diversity" here
is therefore measured across groups: mean pairwise cosine between the group base
outputs on real tokens. Low cosine = the router has a real choice; cosine ~1 =
every expert is chasing the same function.

    HF_HUB_OFFLINE=1 PYTHONPATH=. python experiments/cluster_shards_real.py
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from orrg import MoEConfig, convert_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SHARED, GROUPS, VARIANTS = 4, 4, 14

PROBE = """
A mixture of experts layer replaces a single feed-forward network with a set of
smaller ones plus a router that decides which of them each token uses. Capacity is
only useful if the router learns to send tokens to the experts that handle them
well, and that is a chicken and egg problem.
"""

CALIB = """
The quick brown fox jumps over the lazy dog. In 1969 the first message was sent
across the ARPANET, and by 1983 the network had adopted TCP/IP, which allowed
different kinds of computers to exchange packets. Email followed quickly, then
the web, then search, then social networks, each layer built on the last.
"""

EVAL = """
The committee reviewed the proposal on a cold morning in November. Nobody
objected to the budget, but several members questioned the timeline, and the
chair asked for a revised schedule by the end of the month. The minutes record
the vote as unanimous, with one abstention noted in the margin.
"""

CODE = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

for i in range(10):
    print(i, fibonacci(i))
"""


def load():
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    model.eval()
    return model


def routing_probe(model, fit_batch, eval_batch, ridge: float = 1.0) -> dict:
    """Can a linear probe predict which group's neurons matter for a token?

    Label = argmax_g ||base_g(x)||, i.e. the group whose contribution is largest.
    Fit a ridge classifier on one document's tokens and score another document's,
    so the number is held-out. Chance is 1/G; the majority baseline is reported
    because contribution labels can be imbalanced.
    """

    layers = [m for m in model.modules() if getattr(m, "is_orrg_moe", False)]

    def capture(batch):
        captured: dict[int, torch.Tensor] = {}
        handles = [
            layer.register_forward_pre_hook(
                lambda _m, args, i=i: captured.__setitem__(i, args[0].detach())
            )
            for i, layer in enumerate(layers)
        ]
        try:
            with torch.no_grad():
                model(**batch)
        finally:
            for handle in handles:
                handle.remove()
        return captured

    fit, ev = capture(fit_batch), capture(eval_batch)
    accuracies, majorities = [], []
    for i, layer in enumerate(layers):
        groups = len(layer.groups)

        def labels(x):
            with torch.no_grad():
                norms = torch.stack(
                    [layer.groups[g].base(x).float().norm(dim=-1) for g in range(groups)], dim=-1
                )
            return norms.argmax(-1)

        # Labels are per token: flatten [B, T, H] before taking the group argmax,
        # or argmax(-1) returns one label per batch row instead of per token.
        raw_fit = fit[i].reshape(-1, fit[i].shape[-1])
        raw_ev = ev[i].reshape(-1, ev[i].shape[-1])
        y_fit, y_ev = labels(raw_fit), labels(raw_ev)
        X_fit = raw_fit.float()
        X_ev = raw_ev.float()
        mean = X_fit.mean(0, keepdim=True)
        std = X_fit.std(0, keepdim=True).clamp_min(1e-6)
        X_fit = (X_fit - mean) / std
        X_ev = (X_ev - mean) / std
        one_hot = F.one_hot(y_fit, groups).float()
        gram = X_fit.t() @ X_fit + ridge * torch.eye(X_fit.shape[1], device=X_fit.device)
        weights = torch.linalg.solve(gram, X_fit.t() @ one_hot)
        pred = (X_ev @ weights).argmax(-1)
        accuracies.append(float((pred == y_ev).float().mean()))
        counts = torch.bincount(y_ev, minlength=groups).float()
        majorities.append(float(counts.max() / counts.sum()))

    return {
        "probe_accuracy": sum(accuracies) / len(accuracies),
        "chance": 1 / len(layers[0].groups),
        "majority_baseline": sum(majorities) / len(majorities),
        "per_layer": accuracies,
    }


def run_arm(clustered: bool, tokenizer, probe, calibration, eval_batch) -> dict:
    started = time.time()
    model = load()
    cfg = MoEConfig(
        shared=SHARED,
        groups=GROUPS,
        variants=VARIANTS,
        cluster_shards=clustered,
        cluster_tokens=512,
        cluster_iters=8,
    )
    converted, report = convert_model(
        model, cfg, verify=True, probe=probe, calibration=calibration if clustered else None
    )
    carve = [entry.get("carve_error") for entry in report.per_layer]
    cohesion = [c.cohesion for c in report.clustering]
    baseline = [c.baseline for c in report.clustering]
    routing = routing_probe(converted, calibration[0], eval_batch)
    out = {
        "arm": "clustered" if clustered else "contiguous",
        "total_params": report.total_params,
        "active_k1": report.active_params(1),
        "logit_kl": report.logit_kl,
        "logit_max_abs": report.logit_max_abs,
        "max_carve_error": report.max_carve_error,
        "median_carve": sorted(carve)[len(carve) // 2],
        "cohesion_mean": sum(cohesion) / len(cohesion) if cohesion else None,
        "baseline_mean": sum(baseline) / len(baseline) if baseline else None,
        "routing": routing,
        "seconds": time.time() - started,
    }
    print(
        f"{out['arm']:10s} kl={report.logit_kl:.3e} carve={report.max_carve_error:.2e} "
        f"cohesion={out['cohesion_mean']} baseline={out['baseline_mean']} "
        f"probe acc={routing['probe_accuracy']:.3f} (chance {routing['chance']:.2f}, "
        f"majority {routing['majority_baseline']:.2f}) ({out['seconds']:.0f}s)",
        flush=True,
    )
    del model, converted, report
    gc.collect()
    return out


def main() -> int:
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    probe = tokenizer(PROBE, return_tensors="pt", truncation=True, max_length=192)
    calibration = [
        tokenizer(CALIB, return_tensors="pt", truncation=True, max_length=384),
        tokenizer(CODE, return_tensors="pt", truncation=True, max_length=128),
    ]
    eval_batch = tokenizer(EVAL, return_tensors="pt", truncation=True, max_length=192)
    rows = [
        run_arm(False, tokenizer, probe, calibration, eval_batch),
        run_arm(True, tokenizer, probe, calibration, eval_batch),
    ]
    out = Path("experiments/results/cluster_shards_real.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": MODEL, "rows": rows}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
