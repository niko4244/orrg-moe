"""Training loop with task loss, router distillation and aux-loss-free balancing.

Used by the P1 experiment and intended as the reference recipe:

    loss = CE(logits, labels) + coef * Σ_layers router_distill_loss(layer)

The distillation term gives the router a gradient at step 0, when every residual
expert still outputs zero and the task loss alone cannot move it.
"""

from __future__ import annotations

import itertools
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_frac: float = 0.03
    bias_update_rate: float | None = None
    router_distill_coef: float = 0.0
    orth_step: float = 0.0  # explicit per-step push apart, RMS per weight component
    orth_margin: float = 0.5  # hinge: only push residuals whose cosine is above this
    distill_every: int = 1
    bf16: bool = True
    optimizer: str = "adamw"  # "adamw" (fp32 states, 8 bytes/param) | "adafactor"
    log_every: int = 50
    trace_every: int = 0  # call the trace hook every N steps (0 = never)
    checkpoint_path: str | None = None
    checkpoint_every: int = 0  # save every N steps (0 = never)
    resume: bool = False  # continue from checkpoint_path when it exists


def moe_layers(model: nn.Module) -> list[nn.Module]:
    return [m for m in model.modules() if getattr(m, "is_orrg_moe", False)]


def set_router_distill(model: nn.Module, coef: float) -> None:
    """Toggle router distillation on every MoE layer without rebuilding modules."""

    for layer in moe_layers(model):
        layer.cfg.router_distill_coef = float(coef)
        if coef <= 0:
            layer.last_distill_loss = None


def router_distill_total(model: nn.Module) -> torch.Tensor | None:
    terms = [layer.last_distill_loss for layer in moe_layers(model) if layer.last_distill_loss is not None]
    if not terms:
        return None
    return torch.stack([term.reshape(()) for term in terms]).sum()


def orthogonality_loss(model: nn.Module, margin: float = 0.5) -> torch.Tensor | None:
    """Hinge on the squared cosine between residual up-projections in a group.

    The task loss reaches a residual's up-projection only through its
    down-projection, which starts at zero. This term acts on the weights
directly, so it can differentiate residuals before they contribute anything,
    and it switches off once a pair is below ``margin``.
    """

    terms = []
    for layer in moe_layers(model):
        for group in layer.groups:
            vectors = [
                max(expert.parameters(), key=lambda p: p.numel()).reshape(-1)
                for expert in group.residuals
            ]
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    a, b = vectors[i], vectors[j]
                    n = min(a.numel(), b.numel())  # shard widths can differ by one neuron
                    cos = F.cosine_similarity(a[:n], b[:n], dim=0)
                    terms.append(torch.relu(cos - margin).pow(2))
    if not terms:
        return None
    return torch.stack(terms).mean()


def orthogonality_push(
    model: nn.Module, rate: float, margin: float = 0.5
) -> float | None:
    """One normalised descent step away from residual similarity.

    The hinge on cosine is stationary at exact copies, and optimizer scaling can
    make a penalty in the loss arbitrarily weak (Adafactor normalises per
    parameter). So this applies the gradient of the hinge directly, normalised to
    unit RMS per component, with an explicit step size. Residual up-projections
    carry no function while their down-projection is zero, which is why this can
    act from step 0 without disturbing the lossless conversion.
    """

    params = []
    for layer in moe_layers(model):
        for group in layer.groups:
            for expert in group.residuals:
                params.append(max(expert.parameters(), key=lambda p: p.numel()))
    if len(params) < 2:
        return None
    with torch.enable_grad():
        term = orthogonality_loss(model, margin)
    if term is None:
        return None
    grads = torch.autograd.grad(term, params, allow_unused=True)
    with torch.no_grad():
        for param, grad in zip(params, grads):
            if grad is None or not torch.isfinite(grad).all():
                continue
            norm = grad.norm()
            if float(norm) == 0.0:
                continue
            param.add_(grad, alpha=-rate * (param.numel() ** 0.5) / norm)
    return float(term.detach())


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("model has no trainable parameters")
    if cfg.optimizer == "adafactor":
        # Factored second moments: ~O(sqrt(n)) state instead of 8 bytes/parameter,
        # which is what makes a 600M-parameter arm fit on a 16GB card.
        from transformers.optimization import Adafactor

        return Adafactor(
            params,
            lr=cfg.lr,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=cfg.weight_decay,
        )
    if cfg.optimizer != "adamw":
        raise ValueError(f"unknown optimizer {cfg.optimizer!r}")
    return torch.optim.AdamW(
        params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(cfg.beta1, cfg.beta2)
    )


def _infinite(batches: Iterable) -> Iterable:
    if isinstance(batches, (list, tuple)):
        return itertools.cycle(batches)
    return batches


def _forward_loss(model: nn.Module, batch: Mapping, *, bf16: bool, device: torch.device):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
        outputs = model(**batch)
        logits = outputs.logits
    loss = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        batch["labels"].reshape(-1),
    )
    return loss


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    batches: Iterable[Mapping],
    *,
    iters: int | None = None,
    device: torch.device | None = None,
    bf16: bool = True,
) -> float:
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for batch in batches:
        if iters is not None and count >= iters:
            break
        loss = _forward_loss(model, batch, bf16=bf16, device=device)
        total += float(loss)
        count += 1
    model.train(was_training)
    return total / max(count, 1)


def _save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
) -> None:
    """Atomically save trainable weights, optimizer/scheduler state and RNG.

    Only trainable parameters are stored: frozen skeleton blocks are rebuilt
    bit-exactly from the seed at conversion, so they do not need to be copied.
    """

    payload = {
        "step": step,
        "model": {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad},
        "optim": optimizer.state_dict(),
        "sched": scheduler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def train(
    model: nn.Module,
    batches: Iterable[Mapping],
    steps: int,
    cfg: TrainConfig | None = None,
    *,
    tag: str = "",
    log: Callable[[str], None] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    trace: Callable[[int], None] | None = None,
) -> dict:
    """Run ``steps`` optimization steps. ``batches`` may be a list or a generator."""

    cfg = cfg or TrainConfig()
    device = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer or build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        total_steps=max(steps, 1),
        pct_start=max(cfg.warmup_frac, 0.02),
        # Adafactor has no momentum/beta1 to cycle.
        cycle_momentum=cfg.optimizer == "adamw",
    )
    start_step = 0
    if cfg.resume and cfg.checkpoint_path and os.path.exists(cfg.checkpoint_path):
        payload = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
        expected = {name for name, p in model.named_parameters() if p.requires_grad}
        if set(payload["model"]) != expected:
            raise RuntimeError(
                f"checkpoint {cfg.checkpoint_path} does not match this model's trainable parameters"
            )
        model.load_state_dict(payload["model"], strict=False)
        optimizer.load_state_dict(payload["optim"])
        scheduler.load_state_dict(payload["sched"])
        torch.set_rng_state(payload["torch_rng"])
        if payload.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        start_step = int(payload["step"])
        if start_step >= steps:
            raise ValueError(
                f"checkpoint is at step {start_step}, which is not before the budget {steps}"
            )
        if log:
            log(f"    {tag} resuming from step {start_step}/{steps}")

    model.train()
    losses: list[float] = []
    distill_values: list[float] = []
    orth_values: list[float] = []
    started = time.time()

    # Skip the data stream to the saved step: generation is deterministic, so a
    # resumed run sees exactly the batches the uninterrupted run would have seen.
    stream = itertools.islice(_infinite(batches), start_step, steps)
    for offset, batch in enumerate(stream):
        step = start_step + offset
        if cfg.router_distill_coef > 0:
            active = cfg.distill_every <= 1 or step % cfg.distill_every == 0
            set_router_distill(model, cfg.router_distill_coef if active else 0.0)
        else:
            set_router_distill(model, 0.0)

        loss = _forward_loss(model, batch, bf16=cfg.bf16, device=device)
        distill = router_distill_total(model)
        total = loss
        if distill is not None:
            total = loss + cfg.router_distill_coef * distill
            distill_values.append(float(distill.detach()))


        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if total.requires_grad:
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optimizer.step()
            scheduler.step()

        for layer in moe_layers(model):
            layer.apply_bias_update(cfg.bias_update_rate)

        if cfg.orth_step > 0:
            pressed = orthogonality_push(model, cfg.orth_step, cfg.orth_margin)
            if pressed is not None:
                orth_values.append(pressed)

        losses.append(float(loss.detach()))
        if cfg.checkpoint_path and cfg.checkpoint_every and (step + 1) % cfg.checkpoint_every == 0:
            _save_checkpoint(cfg.checkpoint_path, model, optimizer, scheduler, step + 1)
        if trace is not None and cfg.trace_every and (step + 1) % cfg.trace_every == 0:
            trace(step + 1)
        if log and cfg.log_every and (step + 1) % cfg.log_every == 0:
            log(
                f"    {tag} step {step + 1}/{steps} "
                f"loss={sum(losses[-cfg.log_every:]) / len(losses[-cfg.log_every:]):.4f}"
            )

    elapsed = time.time() - started
    return {
        "steps": steps,
        "loss_mean": sum(losses) / max(len(losses), 1),
        "loss_last": sum(losses[-50:]) / max(len(losses[-50:]), 1),
        "distill_mean": sum(distill_values) / len(distill_values) if distill_values else None,
        "orth_mean": sum(orth_values) / len(orth_values) if orth_values else None,
        "seconds": elapsed,
        "seconds_per_step": elapsed / max(len(losses), 1),
        "trainable_params": sum(p.numel() for p in params),
    }
