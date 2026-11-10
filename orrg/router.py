"""The model-agnostic token router (the "tokenizer" for experts).

The router sees only the hidden state that enters an FFN block, so it works
unchanged for every architecture that passes ``[..., hidden]`` through an MLP.
Routing is grouped: each group owns one always-on base shard plus a set of
residual experts, and exactly ``group_top_k`` residuals are selected per group
per token. That makes load balance structural instead of a penalty term that
fights specialization.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import RouterConfig


@dataclass
class RouteResult:
    """Hard assignment plus the probabilities a router-distillation loss needs."""

    index: torch.Tensor  # [N, G, k] expert ids within each group (or [N, k] when flat)
    weight: torch.Tensor  # [N, G, k]
    probs: torch.Tensor  # [N, G, V] full distribution per group
    logits: torch.Tensor  # [N, G, V] pre-softmax scores

    def __iter__(self):  # allow ``idx, w, probs = layer(...)``-style unpacking
        return iter((self.index, self.weight, self.probs))


class TokenRouter(nn.Module):
    """Per-token, per-group expert selection."""

    def __init__(self, hidden: int, n_groups: int, n_variants: int, cfg: RouterConfig | None = None):
        super().__init__()
        if n_groups < 1 or n_variants < 1:
            raise ValueError("router needs at least one group and one variant")
        self.hidden = hidden
        self.n_groups = n_groups
        self.n_variants = n_variants
        self.cfg = cfg or RouterConfig()
        self.weight = nn.Parameter(torch.empty(n_groups * n_variants, hidden))
        self.bias = nn.Parameter(torch.zeros(n_groups * n_variants))
        nn.init.normal_(self.weight, std=0.02)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(torch.float32)
        b = self.bias.to(torch.float32)
        scores = F.linear(x.to(torch.float32), w, b)
        return scores.view(-1, self.n_groups, self.n_variants)

    def forward(self, x: torch.Tensor) -> RouteResult:
        logits = self.logits(x)
        if self.training and self.cfg.jitter > 0:
            logits = logits + torch.randn_like(logits) * self.cfg.jitter
        probs = logits.softmax(dim=-1)

        if self.cfg.routing == "grouped":
            k = min(self.cfg.group_top_k, self.n_variants)
            weight, index = probs.topk(k, dim=-1)
        else:
            k = min(self.cfg.flat_top_k, self.n_groups * self.n_variants)
            flat = probs.reshape(probs.shape[0], -1)
            weight, index = flat.topk(k, dim=-1)
            probs = probs  # full distribution kept for aux losses

        if self.cfg.normalize_top_k and weight.shape[-1] > 1:
            weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return RouteResult(index=index, weight=weight, probs=probs, logits=logits)

    @torch.no_grad()
    def assign(self, x: torch.Tensor) -> torch.Tensor:
        """Hard assignment without building an autograd graph (analysis / export)."""

        was_training = self.training
        self.eval()
        try:
            return self.forward(x).index
        finally:
            self.train(was_training)

    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden}, groups={self.n_groups}, variants={self.n_variants}, "
            f"routing={self.cfg.routing}, top_k={self.cfg.group_top_k if self.cfg.routing == 'grouped' else self.cfg.flat_top_k}"
        )


@torch.no_grad()
def update_router_bias(bias: torch.Tensor, counts: torch.Tensor, rate: float) -> None:
    """Aux-loss-free balancing: nudge each expert's bias by the sign of its load.

    ``counts`` is a flat vector of per-expert token counts from the last
    optimizer step. Overloaded experts get a negative bias, starved ones a
    positive bias. This decouples balance from the task loss entirely, so the
    router is not asked to trade quality for uniformity.
    """

    if rate <= 0 or counts.numel() == 0:
        return
    total = counts.sum().clamp_min(1)
    target = total / counts.numel()
    bias.add_(rate * torch.sign(target - counts.float()).to(bias.dtype))
    bias.sub_(bias.mean())


def router_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Mean entropy (nats) of the per-group distribution, normalized by log V."""

    p = probs.clamp_min(1e-9)
    ent = -(p * p.log()).sum(dim=-1)
    return ent.mean() / torch.log(torch.tensor(float(probs.shape[-1]), device=probs.device)).clamp_min(1e-9)
