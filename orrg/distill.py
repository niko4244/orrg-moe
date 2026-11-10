"""Distillation utilities: output KD and router-level distillation.

The router signal that works at conversion time cannot come from the task loss
alone (residual experts output zero at t=0, so the router's gradient is zero).
Instead we define a *contribution target*: for a token, which residual expert
would reduce the layer's residual error the most, measured by the norm of its
output. Training the router toward that target is router distillation, and it
can run with every weight frozen.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .moe import MoELayer


def _subsample(flat: torch.Tensor, max_tokens: int) -> torch.Tensor:
    """Evenly strided token subset. The target is a per-token distribution, so a
    stride keeps it unbiased while cutting the extra expert passes."""

    if max_tokens and flat.shape[0] > max_tokens:
        stride = max(1, flat.shape[0] // max_tokens)
        return flat[::stride][:max_tokens]
    return flat


def kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    *,
    reverse: bool = False,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Token-level KL between two next-token distributions.

    ``reverse=True`` gives reverse KL (mode seeking), which is what on-policy
    distillation of sampled student sequences wants.
    """

    t = max(temperature, 1e-6)
    log_s = F.log_softmax(student_logits / t, dim=-1)
    log_t = F.log_softmax(teacher_logits / t, dim=-1)
    if reverse:
        loss = (log_s.exp() * (log_s - log_t)).sum(-1)
    else:
        loss = (log_t.exp() * (log_t - log_s)).sum(-1)
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp_min(1)
    return loss.mean() * t * t


@torch.no_grad()
def contribution_targets(
    layer: MoELayer, x: torch.Tensor, temperature: float = 1.0, max_tokens: int = 0
) -> list[torch.Tensor]:
    """Per-group target distribution over residual experts.

    Runs every residual expert (no sparsity) so the target reflects the whole
    group, then normalizes the output norms. With zero-output residuals the
    target starts uniform, which is the correct prior: no expert is better yet.
    """

    flat = _subsample(x.reshape(-1, x.shape[-1]), max_tokens)
    targets: list[torch.Tensor] = []
    for group in layer.groups:
        if len(group.residuals) == 0:
            continue
        norms = torch.stack(
            [expert(flat).float().norm(dim=-1) for expert in group.residuals], dim=-1
        )
        scaled = norms / max(temperature, 1e-6)
        targets.append(torch.softmax(scaled, dim=-1))
    return targets


def router_distill_loss(
    layer: MoELayer,
    x: torch.Tensor,
    *,
    temperature: float = 1.0,
    target_temperature: float = 1.0,
    max_tokens: int = 0,
) -> torch.Tensor:
    """CE between the router's group distribution and the contribution target."""

    flat = _subsample(x.reshape(-1, x.shape[-1]), max_tokens)
    route = layer.router(flat)
    targets = contribution_targets(layer, flat, temperature=target_temperature)
    if not targets:
        return flat.new_zeros(())
    probs = route.probs
    if probs.shape[1] == 1 and probs.shape[-1] == targets[0].shape[-1]:  # flat router
        probs = probs.reshape(probs.shape[0], 1, -1)
    loss = flat.new_zeros(())
    for gi, target in enumerate(targets):
        if gi >= probs.shape[1]:
            break
        loss = loss - (target * torch.log(probs[:, gi].clamp_min(1e-9))).sum(-1).mean()
    return loss / len(targets)


def on_policy_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    prompt_len: int,
    temperature: float = 1.0,
    reverse: bool = True,
) -> torch.Tensor:
    """Distill only the continuation of sequences (prompt tokens masked out)."""

    mask = torch.zeros(student_logits.shape[:2], device=student_logits.device)
    mask[:, prompt_len:] = 1.0
    return kl_divergence(
        student_logits[:, :-1],
        teacher_logits[:, :-1],
        temperature=temperature,
        reverse=reverse,
        mask=mask[:, :-1],
    )


def contribution_entropy(targets: list[torch.Tensor]) -> float:
    """How concentrated the contribution targets are (0 = uniform, 1 = one-hot)."""

    if not targets:
        return 0.0
    values = []
    for target in targets:
        p = target.clamp_min(1e-9)
        ent = -(p * p.log()).sum(-1).mean()
        values.append(1.0 - float(ent) / torch.log(torch.tensor(float(target.shape[-1]))).item())
    return sum(values) / len(values)
