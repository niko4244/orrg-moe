"""Routing and conversion metrics used as the gates in the scale-up protocol."""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F

from .moe import MoELayer


def moe_layers(model: torch.nn.Module) -> list[MoELayer]:
    return [m for m in model.modules() if isinstance(m, MoELayer)]


def load_imbalance(model: torch.nn.Module) -> float:
    """max(util) - mean(util). 0 is perfectly balanced; 1 is total collapse."""

    values = []
    for layer in moe_layers(model):
        stats = layer.last_stats
        if stats is None:
            continue
        util = stats["utilization"].float()
        if util.numel() > 1:
            values.append(float(util.max() - util.mean()))
    return sum(values) / len(values) if values else 0.0


def dead_experts(model: torch.nn.Module) -> int:
    """Experts that received no tokens in the last recorded forward pass."""

    total = 0
    for layer in moe_layers(model):
        stats = layer.last_stats
        if stats is None:
            continue
        total += int((stats["counts"] == 0).sum().item())
    return total


def router_entropy(model: torch.nn.Module) -> float:
    values = [layer.last_stats["entropy"] for layer in moe_layers(model) if layer.last_stats]
    return sum(values) / len(values) if values else 0.0


def expert_utilization(model: torch.nn.Module) -> torch.Tensor:
    chunks = [
        layer.last_stats["utilization"].float()
        for layer in moe_layers(model)
        if layer.last_stats
    ]
    return torch.cat(chunks) if chunks else torch.zeros(0)


def expert_diversity(model: torch.nn.Module) -> float:
    """Mean pairwise 1 - cosine between residual experts inside the same group.

    Near 0 means the experts are still copies; the toy-scale experiments showed
    ~0.025 at the start of joint training and ~0.08 with stronger perturbation.
    """

    sims = []
    for layer in moe_layers(model):
        for group in layer.groups:
            residuals = list(group.residuals)
            if len(residuals) < 2:
                continue
            vectors = []
            for expert in residuals:
                weight = max(expert.parameters(), key=lambda p: p.numel())
                vectors.append(weight.detach().float().reshape(-1))
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    a, b = vectors[i], vectors[j]
                    n = min(a.numel(), b.numel())  # shard widths can differ by one neuron
                    cos = F.cosine_similarity(a[:n], b[:n], dim=0)
                    sims.append(float(1.0 - cos))
    return max(0.0, sum(sims) / len(sims)) if sims else 0.0


def conversion_error(reference_logits: torch.Tensor, candidate_logits: torch.Tensor) -> dict:
    """Compare a converted model to its dense seed on the same batch."""

    ref = reference_logits.float()
    got = candidate_logits.float()
    log_q = F.log_softmax(ref, dim=-1)
    log_p = F.log_softmax(got, dim=-1)
    kl = max(0.0, float((log_q.exp() * (log_q - log_p)).sum(-1).mean()))
    max_abs = float((ref - got).abs().max())
    denom = float(ref.abs().max()) + 1e-9
    top1 = float((ref.argmax(-1) == got.argmax(-1)).float().mean())
    return {
        "kl": kl,
        "max_abs": max_abs,
        "relative_max_abs": max_abs / denom,
        "top1_agreement": top1,
    }


@torch.no_grad()
def residual_share(model: torch.nn.Module, batch: Mapping) -> float:
    """Sum of residual-expert output norms over the layer output norm, mean over layers.

    This is the direct test of whether residuals are alive: it is exactly 0 at
    conversion (the down-projection is zeroed) and grows only as residuals start
    contributing to the layer output. Weights can diverge while this stays ~0,
    in which case the extra capacity is still inert.
    """

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    layers = moe_layers(model)
    totals: dict[int, float] = {}
    residual: dict[int, float] = {i: 0.0 for i in range(len(layers))}
    handles = []

    def layer_hook(module, inputs, output, li):
        totals[li] = float(output.detach().float().norm())

    def expert_hook(module, inputs, output, li):
        residual[li] += float(output.detach().float().norm())

    for li, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(lambda *a, li=li: layer_hook(*a, li)))
        for group in layer.groups:
            for expert in group.residuals:
                handles.append(
                    expert.register_forward_hook(lambda *a, li=li: expert_hook(*a, li))
                )

    try:
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    values = [residual[li] / totals[li] for li in totals if totals[li] > 0]
    return sum(values) / len(values) if values else 0.0


def collect_metrics(model: torch.nn.Module) -> dict:
    return {
        "load_imbalance": load_imbalance(model),
        "dead_experts": dead_experts(model),
        "router_entropy": router_entropy(model),
        "expert_diversity": expert_diversity(model),
        "layers": len(moe_layers(model)),
    }


def perplexity(loss: float) -> float:
    return math.exp(loss) if loss < 50 else float("inf")
