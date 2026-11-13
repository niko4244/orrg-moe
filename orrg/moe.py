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


def _expert_range(indices: torch.Tensor, ratio: float) -> torch.Tensor:
    """The first ``ratio`` fraction of a group's neurons (ratio=1 returns all)."""

    if ratio >= 1.0:
        return indices
    # ratio <= 1 is validated in MoEConfig, so this never exceeds the shard.
    return indices[: max(1, int(round(indices.numel() * ratio)))]


def _contiguous(index_sets: list[torch.Tensor]) -> bool:
    for indices in index_sets:
        if indices.numel() and int(indices[-1] - indices[0]) != indices.numel() - 1:
            return False
    return True


class MoEGroup(nn.Module):
    """One base shard plus its residual experts; dispatch lives in ``MoELayer``."""

    def __init__(self, base: nn.Module, residuals: list[nn.Module]):
        super().__init__()
        self.base = base
        self.residuals = nn.ModuleList(residuals)


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
        # neurons, not extra slices of the axis. ``skeleton_scale`` < 1 keeps a
        # prefix of that axis, so the skeleton no longer reproduces the seed FFN
        # and the conversion is approximate; MoEConfig rejects it together with
        # cluster_shards, which needs the whole axis.
        self.skeleton_width = max(1, int(round(spec.intermediate * cfg.skeleton_scale)))
        if cfg.dispatch == "fused" and (
            len(spec.up) != 2
            or len(spec.down) != 1
            or spec.up_bias
            or spec.out_bias
            or not hasattr(spec.module, "gate_proj")
        ):
            raise ValueError(
                "dispatch='fused' needs a gated FFN with gate_proj/up_proj/down_proj "
                "and no biases"
            )
        if index_sets is None:
            bounds = shard_bounds(self.skeleton_width, cfg.shared + cfg.groups)
            index_sets = [torch.arange(start, end) for start, end in bounds]
        if len(index_sets) != cfg.shared + cfg.groups:
            raise ValueError(
                f"expected {cfg.shared + cfg.groups} index sets, got {len(index_sets)}"
            )
        union = torch.cat([idx.long() for idx in index_sets])
        if union.numel() != self.skeleton_width or union.unique().numel() != self.skeleton_width:
            raise ValueError(
                f"index sets must partition the retained intermediate range exactly "
                f"({self.skeleton_width} neurons)"
            )
        self.index_sets = [idx.long() for idx in index_sets]
        self.widths = [int(idx.numel()) for idx in self.index_sets]
        self.mode = cfg.mode
        self.expert_mode = cfg.expert_mode
        self.dispatch = cfg.dispatch
        self._fused = None
        # Expert width is a fraction of the group's shard. ratio=1 gives the
        # historical "residual is a copy of the base range" behaviour.
        self.expert_indices = [
            _expert_range(indices, cfg.expert_width_ratio) for indices in self.index_sets[cfg.shared :]
        ]
        self.expert_widths = [int(idx.numel()) for idx in self.expert_indices]

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
            expert_idx = self.expert_indices[g]
            residuals = [
                narrow_module(
                    spec.module,
                    spec,
                    indices=expert_idx,
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
        if self.mode == "growth" and cfg.expert_mode == "replace":
            # Variant 0 is the group's base shard, so the router chooses among
            # ``variants`` options instead of ``variants - 1`` residuals.
            n_groups_router, n_variants_router = cfg.groups, cfg.variants
            n_experts = cfg.groups * cfg.variants
        elif self.mode == "growth":
            n_groups_router, n_variants_router = cfg.groups, max(1, cfg.variants - 1)
        else:
            n_groups_router, n_variants_router = 1, cfg.shared + cfg.groups
        self.router = TokenRouter(
            hidden=spec.hidden,
            n_groups=n_groups_router,
            n_variants=n_variants_router,
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

        if self.expert_mode == "replace":
            return self._forward_replace(flat).reshape(shape)

        if self.dispatch == "fused":
            if torch.is_grad_enabled():
                raise RuntimeError(
                    "dispatch='fused' has no autograd path; run it under torch.no_grad() "
                    "or set dispatch='python'"
                )
            if self._fused is None:
                from .dispatch import FusedDispatch

                self._fused = FusedDispatch(self)
            return self._fused.forward(flat).reshape(shape)

        # Recombination is a sum of shard outputs. Under bf16/fp16 every partial
        # sum rounds, and eight shards cost ~1e-2 relative per block, so the
        # accumulator is float32: the only error left is the final cast.
        acc = torch.float32 if flat.dtype in (torch.bfloat16, torch.float16) else flat.dtype
        out = None
        for module in self.shared:
            part = module(flat).to(acc)
            out = part if out is None else out + part
        for group in self.groups:
            part = group.base(flat).to(acc)
            out = part if out is None else out + part
        if out is None:  # cfg with zero shards is rejected in MoEConfig
            out = torch.zeros_like(flat, dtype=acc)

        if self.cfg.n_residuals > 0 and self.cfg.top_residuals > 0:
            route = self.router(flat)
            index, weight = route.index, route.weight  # [N, G, k]
            contribution = torch.zeros(flat.shape, dtype=acc, device=flat.device)
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
                            # The router computes probabilities in float32; line the
                            # operands up with the accumulator or a bf16 checkpoint
                            # raises (and fp32 inputs would silently promote).
                            scale = selected_w[tokens].to(acc).unsqueeze(-1)
                            contribution.index_add_(0, tokens, expert(flat[tokens]).to(acc) * scale)
                        start = end
            out = out + contribution
            self._record(route, flat.shape[0], groups=True)

        if (
            self.training
            and self.cfg.router_distill_coef > 0
            and self.expert_mode == "additive"  # targets are residual-only
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

        return out.to(flat.dtype).reshape(shape)

    def _forward_replace(self, flat: torch.Tensor) -> torch.Tensor:
        """Group top-1 over ``{base} ∪ residuals``; the selected variant replaces."""

        acc = torch.float32 if flat.dtype in (torch.bfloat16, torch.float16) else flat.dtype
        out = None
        for module in self.shared:
            part = module(flat).to(acc)
            out = part if out is None else out + part
        if out is None:
            out = torch.zeros_like(flat, dtype=acc)

        route = self.router(flat)
        index = route.index[:, :, 0]  # [N, G] one variant per group
        probability = route.weight[:, :, 0]
        # A replacement is a hard switch: the selected variant contributes its
        # full output, so selecting variant 0 reproduces the dense block exactly.
        # Subtracting the detached value keeps the forward pass at exactly 1 while
        # still passing a gradient to the router probability (straight-through).
        weight = (probability - probability.detach() + 1.0).to(acc)
        contribution = torch.zeros(flat.shape, dtype=acc, device=flat.device)
        for gi, group in enumerate(self.groups):
            variants = [group.base, *group.residuals]
            selected = index[:, gi]
            order = torch.argsort(selected)
            counts = torch.bincount(selected, minlength=len(variants))
            ends = counts.cumsum(0).tolist()
            start = 0
            for ei, module in enumerate(variants):
                end = ends[ei]
                if end > start:
                    tokens = order[start:end]
                    scale = weight[tokens, gi].unsqueeze(-1)
                    contribution.index_add_(0, tokens, module(flat[tokens]).to(acc) * scale)
                start = end
        self._record(route, flat.shape[0], groups=True)
        self.last_distill_loss = None
        return (out + contribution).to(flat.dtype)

    def _forward_partition(self, flat: torch.Tensor) -> torch.Tensor:
        route = self.router(flat)
        acc = torch.float32 if flat.dtype in (torch.bfloat16, torch.float16) else flat.dtype
        out = torch.zeros_like(flat, dtype=acc)
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
                weighted = expert(flat[tokens]).to(acc) * selected_w[tokens].to(acc).unsqueeze(-1)
                out = out.index_add(0, tokens, weighted)
        self._record(route, flat.shape[0], groups=False)
        return out.to(flat.dtype)

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
        geometry = ""
        if self.cfg.skeleton_scale != 1.0:
            geometry += f", skeleton={self.skeleton_width}/{self.intermediate}"
        if self.cfg.expert_width_ratio != 1.0:
            geometry += f", expert_width={self.expert_widths[0]}"
        return (
            f"{self.layer_name}: hidden={self.hidden}, intermediate={self.intermediate}, "
            f"skeleton_shards={len(self.widths)}, width={self.widths[0]}, "
            f"variants={self.cfg.variants}, mode={self.mode}, experts={self.expert_mode}"
            f"{geometry}, "
            f"clustered={self.widths != sorted(self.widths) or not _contiguous(self.index_sets)}"
        )
