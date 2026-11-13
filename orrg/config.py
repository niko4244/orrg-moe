"""Configuration objects for ORRG dense -> MoE conversion.

Design invariants (see spec/ORRG-SPEC-v3.md):

1. The *skeleton* (shared shards + one base shard per group) tiles the dense FFN's
   intermediate dimension exactly, so it is function-preserving on its own.
2. Extra variants per group are *residual* experts: copies of the group's base
   shard whose down-projection is zeroed. They output exactly 0 at t=0, so the
   converted model is lossless, but they are free to grow instead of being
   identical copies of a full FFN.
3. Routing over residuals is grouped and structurally balanced: every group
   contributes, so no expert can be starved by a load-balancing loss fighting
   specialization.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict, replace
from typing import Sequence


@dataclass
class RouterConfig:
    """Settings for the per-token router (the model-agnostic "tokenizer")."""

    routing: str = "grouped"  # "grouped" | "flat"
    group_top_k: int = 1  # residual experts active per group per token
    flat_top_k: int = 1  # used when routing == "flat" (partition mode)
    jitter: float = 0.0  # gaussian noise added to logits while training
    normalize_top_k: bool = True  # renormalize selected weights to sum to 1
    bias_update_rate: float = 0.0  # aux-loss-free balancing step size (0 = off)
    dtype: str = "float32"  # router always runs in fp32

    def __post_init__(self) -> None:
        if self.routing not in ("grouped", "flat"):
            raise ValueError(f"routing must be 'grouped' or 'flat', got {self.routing!r}")
        if self.group_top_k < 1 or self.flat_top_k < 1:
            raise ValueError("top_k values must be >= 1")
        if self.jitter < 0:
            raise ValueError("jitter must be >= 0")


# Tag set on a RouterConfig that ``MoEConfig`` derived, so a later
# ``dataclasses.replace`` is not mistaken for a caller conflict.
_DERIVED_ROUTER = "_orrg_derived_group_top_k"


@dataclass
class MoEConfig:
    """Shape of the converted mixture of experts."""

    shared: int = 1  # always-on skeleton shards
    groups: int = 4  # routed groups; each contributes at least one base shard
    variants: int = 4  # shards per group (variant 0 = base, 1..V-1 = residuals)
    top_residuals: int = 1  # residual experts active per token per group
    mode: str = "growth"  # "growth" (additive residuals) | "partition" (top-k over shards)
    expert_mode: str = "additive"  # "additive" (base + Σ residuals) | "replace" (one variant, replaces)
    expert_width_ratio: float = 1.0  # residual width as a fraction of its group shard (1.0 = legacy)
    skeleton_scale: float = 1.0  # fraction of the intermediate axis the skeleton tiles (1.0 = lossless)
    router_warm_start: str = "none"  # "none" | "cluster" (nearest-centroid on seed hidden states)
    dispatch: str = "python"  # "python" (reference) | "fused" (grouped-GEMM inference path)
    residual_init: str = "zero"  # "zero" (lossless) | "copy" (naive) | "scaled" (magnitude lever)
    top_k_partition: int = 1  # only for mode="partition"
    hidden_size: int | None = None  # overrides auto-detection
    include: Sequence[str] = ()  # regex allow-list for module names
    exclude: Sequence[str] = ()  # regex deny-list for module names
    freeze_non_moe: bool = False
    freeze_skeleton: bool = False
    cluster_shards: bool = False  # activation-clustered instead of contiguous shards
    cluster_tokens: int = 1024  # calibration tokens kept per layer
    cluster_iters: int = 12
    cluster_seed: int = 0
    router_distill_coef: float = 0.0  # router distillation weight during training
    distill_tokens: int = 0  # tokens per layer used for the target pass (0 = all)
    distill_target_temp: float = 1.0  # <1 sharpens the contribution target
    residual_noise: float = 0.0  # symmetry-breaking jitter on residual up-projections
    down_init_scale: float = 0.05  # residual down-projection scale when residual_init="scaled"
    router: RouterConfig | None = None

    def __post_init__(self) -> None:
        # The routed-expert count is not independently configurable. The reference
        # forward loops over ``route.index.shape[-1]``, i.e. the router's k, while
        # the report quotes ``top_residuals``; if they disagree the model claims k
        # experts and routes one. Make the router follow ``top_residuals``.
        routed = self.mode == "growth" and self.expert_mode == "additive"
        wanted = max(1, self.top_residuals)
        if self.router is None:
            self.router = RouterConfig(group_top_k=wanted if routed else 1)
        elif routed and self.router.group_top_k != wanted:
            # ``dataclasses.replace(cfg, top_residuals=N)`` carries the old router
            # over, which is the normal way to change k and must stay quiet. A
            # router this class derived is tagged, so an untagged one that
            # disagrees can only have come from the caller.
            if not getattr(self.router, _DERIVED_ROUTER, False):
                warnings.warn(
                    f"router.group_top_k={self.router.group_top_k} disagrees with "
                    f"top_residuals={self.top_residuals}; routing k={wanted} instead. "
                    f"Pass RouterConfig(group_top_k={wanted}) to set it explicitly.",
                    stacklevel=2,
                )
            self.router = replace(self.router, group_top_k=wanted)
        if routed:
            # Tag a copy so the caller's object is untouched and a later
            # ``replace(cfg, top_residuals=N)`` stays quiet.
            self.router = replace(self.router)
            setattr(self.router, _DERIVED_ROUTER, True)
        if self.mode not in ("growth", "partition"):
            raise ValueError(f"mode must be 'growth' or 'partition', got {self.mode!r}")
        if self.expert_mode not in ("additive", "replace"):
            raise ValueError(
                f"expert_mode must be 'additive' or 'replace', got {self.expert_mode!r}"
            )
        if not (0.0 < self.expert_width_ratio <= 1.0):
            raise ValueError(
                f"expert_width_ratio must be in (0, 1], got {self.expert_width_ratio!r}"
            )
        if not (0.0 < self.skeleton_scale <= 1.0):
            raise ValueError(
                f"skeleton_scale must be in (0, 1], got {self.skeleton_scale!r}"
            )
        if self.cluster_shards and self.skeleton_scale != 1.0:
            raise ValueError(
                "cluster_shards needs the whole intermediate axis; "
                "skeleton_scale < 1 is only supported with contiguous shards"
            )
        if self.router_warm_start not in ("none", "cluster"):
            raise ValueError(
                f"router_warm_start must be 'none' or 'cluster', got {self.router_warm_start!r}"
            )
        if self.dispatch not in ("python", "fused"):
            raise ValueError(f"dispatch must be 'python' or 'fused', got {self.dispatch!r}")
        if self.dispatch == "fused":
            if self.mode != "growth":
                raise ValueError("dispatch='fused' is only defined for growth mode")
            if self.expert_mode != "additive":
                raise ValueError("dispatch='fused' is only defined for expert_mode='additive'")
            if self.top_k_partition != 1:
                raise ValueError("dispatch='fused' does not use top_k_partition")
        if self.expert_mode == "replace":
            if self.mode == "partition":
                raise ValueError("expert_mode='replace' is only defined for growth mode")
            if self.top_residuals < 1:
                raise ValueError("expert_mode='replace' needs top_residuals >= 1")
            if self.router.routing == "grouped" and self.router.group_top_k != 1:
                raise ValueError(
                    "expert_mode='replace' selects exactly one variant per group; "
                    "set router.group_top_k=1"
                )
        if self.residual_init not in ("zero", "copy", "scaled"):
            raise ValueError(
                f"residual_init must be 'zero', 'copy' or 'scaled', got {self.residual_init!r}"
            )
        if self.shared < 0:
            raise ValueError("shared must be >= 0")
        if self.groups < 1:
            raise ValueError("groups must be >= 1")
        if self.variants < 1:
            raise ValueError("variants must be >= 1")
        if self.shared + self.groups * self.variants < 2:
            raise ValueError("need at least two shards")
        if self.mode == "growth":
            if self.top_residuals < 0:
                raise ValueError("top_residuals must be >= 0")
            if self.top_residuals > self.variants - 1:
                raise ValueError(
                    "top_residuals exceeds the number of residual experts per group "
                    f"({self.variants - 1})"
                )
        if self.mode == "partition" and self.top_k_partition > self.shared + self.groups:
            raise ValueError("top_k_partition exceeds the number of skeleton shards")
        if self.cluster_tokens < 8 or self.cluster_iters < 1:
            raise ValueError("clustering needs at least 8 calibration tokens and 1 iteration")
        if self.router_distill_coef < 0:
            raise ValueError("router_distill_coef must be >= 0")
        if self.distill_target_temp <= 0:
            raise ValueError("distill_target_temp must be > 0")
        if self.residual_noise < 0:
            raise ValueError("residual_noise must be >= 0")
        if self.down_init_scale <= 0:
            raise ValueError("down_init_scale must be > 0")

    @property
    def n_shards(self) -> int:
        return self.shared + self.groups * self.variants

    @property
    def residual_experts_per_group(self) -> int:
        return self.variants - 1

    @property
    def n_residuals(self) -> int:
        return self.groups * self.residual_experts_per_group

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "MoEConfig":
        payload = dict(payload)
        payload["router"] = RouterConfig(**payload.get("router", {}))
        payload["include"] = tuple(payload.get("include", ()))
        payload["exclude"] = tuple(payload.get("exclude", ()))
        return cls(**payload)

    def describe(self) -> str:
        text = (
            f"shared={self.shared} groups={self.groups} variants={self.variants} "
            f"residuals/group={self.residual_experts_per_group} "
            f"active_residuals={self.top_residuals} shards={self.n_shards} mode={self.mode} "
            f"sharding={'clustered' if self.cluster_shards else 'contiguous'}"
        )
        # Only annotate non-defaults, so the historical string is unchanged.
        if self.expert_mode != "additive":
            text += f" experts={self.expert_mode}"
        if self.expert_width_ratio != 1.0:
            text += f" expert_width={self.expert_width_ratio:g}x"
        if self.skeleton_scale != 1.0:
            text += f" skeleton={self.skeleton_scale:g}x"
        if self.router_warm_start != "none":
            text += f" warm_start={self.router_warm_start}"
        return text
