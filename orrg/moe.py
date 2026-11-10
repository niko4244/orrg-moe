"""Sparse MoE layer built from neuron shards of a dense FFN.

Growth mode
-----------
``out = sum(shared shards) + sum(group base shards) + sum(selected residuals)``

The skeleton (shared + bases) tiles the dense intermediate dimension, so it is
exactly the seed FFN. Residuals are narrow copies with a zeroed down-projection:
they contribute exactly zero at t=0 and are then trained. The converted model
is therefore lossless at conversion *and* every extra expert is free to diverge.

Partition mode
--------------
All shards are pooled and the router selects ``top_k_partition`` of them. This
is the compute-reduction regime (MoEfication-style): exact only when every
shard is active.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn

from .config import MoEConfig
from .ffn import FFNSpec
from .router import TokenRouter, RouteResult, router_entropy, update_router_bias
from .shard import narrow_module, shard_bounds


@dataclass
class LayerStats:
    utilization: torch.Tensor  # [n_experts] fraction of tokens per expert
    entropy: float  # normalized router entropy (1 = uniform)
    aux_loss: torch.Tensor  # switch-style load-balance term (analysis only)
    z_loss: torch.Tensor  # router logit magnitude regularizer
    tokens: int
    counts: torch.Tensor  # raw counts, used for bias updates


def _contiguous(index_sets: list[torch.Tensor]) -> bool:
    for indices in index_sets:
        if indices.numel() and int(indices[-1] - indices[0]) != indices.numel() - 1:
            return False
    return True


class MoEGroup(nn.Module):
    """One base shard plus its additive residual experts."""

    def __init__(self, base: nn.Module, residuals: list[nn.Module]):
        super().__init__()
        self.base = base
        self.residuals = nn.ModuleList(residuals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # analysis helper
        out = self.base(x)
        for expert in self.residuals:
            out = out + expert(x)
        return out


class MoELayer(nn.Module):
    """Drop-in replacement for a dense FFN block."""

    is_orrg_moe = True

    def __init__(
        self,
        spec: FFNSpec,
        cfg: MoEConfig,
        layer_name: str = "",
        index_sets: list[torch.Tensor] | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.layer_name = layer_name or spec.name
        self.hidden = spec.hidden
        self.intermediate = spec.intermediate
        # The skeleton (shared + one base per group) partitions the whole
        # intermediate axis, which is what makes the conversion exactly function
        # preserving. Residual variants are independent copies of their group's
        # neurons, not extra slices of the axis.
        if index_sets is None:
            bounds = shard_bounds(spec.intermediate, cfg.shared + cfg.groups)
            index_sets = [torch.arange(start, end) for start, end in bounds]
        if len(index_sets) != cfg.shared + cfg.groups:
            raise ValueError(
                f"expected {cfg.shared + cfg.groups} index sets, got {len(index_sets)}"
            )
        union = torch.cat([idx.long() for idx in index_sets])
        if union.numel() != spec.intermediate or union.unique().numel() != spec.intermediate:
            raise ValueError("index sets must partition the intermediate axis exactly")
        self.index_sets = [idx.long() for idx in index_sets]
        self.widths = [int(idx.numel()) for idx in self.index_sets]
        self.mode = cfg.mode

        modules: list[nn.Module] = []
        for i in range(cfg.shared):
            modules.append(
                narrow_module(
                    spec.module, spec, indices=self.index_sets[i], keep_out_bias=(i == 0)
                )
            )
        self.shared = nn.ModuleList(modules[: cfg.shared])

        self.groups = nn.ModuleList()
        for g in range(cfg.groups):
            indices = self.index_sets[cfg.shared + g]
            base = narrow_module(
                spec.module,
                spec,
                indices=indices,
                keep_out_bias=(cfg.shared == 0 and g == 0),
            )
            residuals = [
                narrow_module(
                    spec.module,
                    spec,
                    indices=indices,
                    keep_out_bias=False,
                    zero_down=(cfg.residual_init == "zero"),
                )
                for _ in range(cfg.variants - 1)
            ]
            if cfg.residual_init == "scaled":
                # Magnitude lever: give each residual a small nonzero function.
                # A gate or scale on a zeroed down-projection is inert (its
                # gradient is zero too), so this is what lets the up-projections
                # receive gradient from step 0. It trades exact conversion for
                # immediate capacity; the caller can read the cost from
                # ConversionReport.logit_kl.
                for expert in residuals:
                    for slice_spec in spec.down:
                        expert.get_parameter(slice_spec.name).data.mul_(cfg.down_init_scale)
                    for bias_name in spec.out_bias:
                        expert.get_parameter(bias_name).data.mul_(cfg.down_init_scale)
            if cfg.residual_noise > 0:
                # Residuals are exact copies, and a cosine penalty has zero
                # gradient at exact alignment. A small jitter is what lets any
                # differentiation pressure act at all. The down-projection stays
                # zero, so the conversion remains lossless.
                for expert in residuals:
                    up = max(expert.parameters(), key=lambda p: p.numel())
                    up.data.add_(torch.randn_like(up) * up.std() * cfg.residual_noise)
            self.groups.append(MoEGroup(base, residuals))

        n_experts = cfg.n_residuals if self.mode == "growth" else cfg.shared + cfg.groups
        router_cfg = cfg.router
        if self.mode == "partition":
            router_cfg = replace(cfg.router, routing="flat", flat_top_k=cfg.top_k_partition)
        self.router = TokenRouter(
            hidden=spec.hidden,
            n_groups=cfg.groups if self.mode == "growth" else 1,
            n_variants=max(1, cfg.variants - 1)
            if self.mode == "growth"
            else cfg.shared + cfg.groups,
            cfg=router_cfg,
        )

        self._pool: list[nn.Module] = []
        if self.mode == "partition":
            # Partition mode routes over the disjoint skeleton only; residual
            # variants are a capacity-growth device and are disabled here.
            for module in self.shared:
                self._pool.append(module)
            for group in self.groups:
                self._pool.append(group.base)
            # Uniform start is required for exactness at full activation, and
            # with every shard active the router still gets gradient.
            nn.init.zeros_(self.router.weight)
            nn.init.zeros_(self.router.bias)

        device = next((p.device for p in spec.module.parameters()), torch.device("cpu"))
        self.router.to(device=device)
        self.register_buffer("_counts", torch.zeros(max(n_experts, 1), device=device), persistent=False)
        self.register_buffer("_probs", torch.zeros(max(n_experts, 1), device=device), persistent=False)
        self.register_buffer("_tokens", torch.zeros(1, device=device), persistent=False)
        self.last_stats: dict | None = None
        self.last_distill_loss: torch.Tensor | None = None

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])

        if self.mode == "partition":
            return self._forward_partition(flat).reshape(shape)

        out = None
        for module in self.shared:
            out = module(flat) if out is None else out + module(flat)
        for group in self.groups:
            out = group.base(flat) if out is None else out + group.base(flat)
        if out is None:  # cfg with zero shards is rejected in MoEConfig
            out = torch.zeros_like(flat)

        if self.cfg.n_residuals > 0 and self.cfg.top_residuals > 0:
            route = self.router(flat)
            index, weight = route.index, route.weight  # [N, G, k]
            contribution = flat.new_zeros(flat.shape)
            for gi, group in enumerate(self.groups):
                if not group.residuals:
                    continue
                for slot in range(index.shape[-1]):
                    selected = index[:, gi, slot]
                    selected_w = weight[:, gi, slot]
                    # One sort + one host-side sync per group beats scanning a
                    # boolean mask per expert (which syncs twice per expert).
                    order = torch.argsort(selected)
                    counts = torch.bincount(selected, minlength=len(group.residuals))
                    ends = counts.cumsum(0).tolist()
                    start = 0
                    for ei, expert in enumerate(group.residuals):
                        end = ends[ei]
                        if end > start:
                            tokens = order[start:end]
                            contribution.index_add_(
                                0, tokens, expert(flat[tokens]) * selected_w[tokens].unsqueeze(-1)
                            )
                        start = end
            out = out + contribution
            self._record(route, flat.shape[0], groups=True)

        if (
            self.training
            and self.cfg.router_distill_coef > 0
            and self.router.n_variants > 1  # with one residual the target is trivially 1.0
        ):
            # Router-level supervision: the loss depends on the router directly,
            # so it produces a gradient even while every residual is still zero.
            from .distill import router_distill_loss

            self.last_distill_loss = router_distill_loss(
                self,
                flat,
                max_tokens=self.cfg.distill_tokens,
                target_temperature=self.cfg.distill_target_temp,
            )
        else:
            self.last_distill_loss = None

        return out.reshape(shape)

    def _forward_partition(self, flat: torch.Tensor) -> torch.Tensor:
        route = self.router(flat)
        out = torch.zeros_like(flat)
        index = route.index.reshape(flat.shape[0], -1)
        # Partitioned neuron shards recombine by *sum*, not by a convex
        # combination, so the softmax weights are scaled back up by n_shards.
        weight = route.weight.reshape(flat.shape[0], -1) * len(self._pool)
        for slot in range(index.shape[-1]):
            selected = index[:, slot]
            selected_w = weight[:, slot]
            for ei, expert in enumerate(self._pool):
                mask = selected == ei
                if not bool(mask.any()):
                    continue
                tokens = mask.nonzero(as_tuple=True)[0]
                weighted = expert(flat[tokens]) * selected_w[tokens].unsqueeze(-1)
                out = out.index_add(0, tokens, weighted)
        self._record(route, flat.shape[0], groups=False)
        return out

    # ------------------------------------------------------------------- stats
    @torch.no_grad()
    def _record(self, route: RouteResult, n_tokens: int, *, groups: bool) -> None:
        if groups:
            index = route.index  # [N, G, k]
            counts = torch.zeros(self._counts.numel(), device=index.device)
            per_group = self.router.n_variants
            for gi in range(index.shape[1]):
                for slot in range(index.shape[2]):
                    ids = index[:, gi, slot] + gi * per_group
                    counts.index_add_(0, ids, torch.ones_like(ids, dtype=counts.dtype))
            probs = route.probs.reshape(route.probs.shape[0], -1).mean(0)
            self._probs.mul_(0.9).add_(0.1 * probs.to(self._probs.dtype))
        else:
            index = route.index.reshape(-1)
            counts = torch.zeros(self._counts.numel(), device=index.device)
            counts.index_add_(0, index, torch.ones_like(index, dtype=counts.dtype))
            self._probs.mul_(0.9).add_(0.1 * route.probs.mean(0).reshape(-1).to(self._probs.dtype))

        self._counts.add_(counts)
        self._tokens.add_(n_tokens)

        util = counts / max(n_tokens, 1)
        probs_mean = self._probs / self._probs.sum().clamp_min(1e-9)
        aux = (util * probs_mean).sum() * counts.numel()
        z = torch.logsumexp(route.logits.float(), dim=-1).pow(2).mean()
        self.last_stats = {
            "utilization": util.detach(),
            "entropy": float(router_entropy(route.probs.detach())),
            "aux_loss": aux.detach(),
            "z_loss": z.detach(),
            "tokens": n_tokens,
            "counts": counts.detach(),
        }

    def pop_stats(self) -> dict | None:
        """Return and clear accumulated routing stats (for bias updates / logging)."""

        if self._tokens.item() == 0:
            return None
        counts = self._counts.clone()
        tokens = int(self._tokens.item())
        self._counts.zero_()
        self._tokens.zero_()
        util = counts / max(tokens, 1)
        return {
            "counts": counts,
            "tokens": tokens,
            "utilization": util,
            "load_imbalance": float(util.max() - util.mean()) if util.numel() else 0.0,
            "dead_experts": int((counts == 0).sum().item()),
        }

    @torch.no_grad()
    def apply_bias_update(self, rate: float | None = None) -> float | None:
        rate = self.cfg.router.bias_update_rate if rate is None else rate
        stats = self.pop_stats()
        if stats is None or rate <= 0:
            return None
        update_router_bias(self.router.bias, stats["counts"], rate)
        return stats["load_imbalance"]

    def extra_repr(self) -> str:
        return (
            f"{self.layer_name}: hidden={self.hidden}, intermediate={self.intermediate}, "
            f"skeleton_shards={len(self.widths)}, width={self.widths[0]}, "
            f"variants={self.cfg.variants}, mode={self.mode}, "
            f"clustered={self.widths != sorted(self.widths) or not _contiguous(self.index_sets)}"
        )
