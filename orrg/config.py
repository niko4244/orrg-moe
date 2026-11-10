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

from dataclasses import dataclass, field, asdict
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


@dataclass
class MoEConfig:
    """Shape of the converted mixture of experts."""

    shared: int = 1  # always-on skeleton shards
    groups: int = 4  # routed groups; each contributes at least one base shard
    variants: int = 4  # shards per group (variant 0 = base, 1..V-1 = residuals)
    top_residuals: int = 1  # residual experts active per token per group
    mode: str = "growth"  # "growth" (additive residuals) | "partition" (top-k over shards)
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
    router: RouterConfig = field(default_factory=RouterConfig)

    def __post_init__(self) -> None:
        if self.mode not in ("growth", "partition"):
            raise ValueError(f"mode must be 'growth' or 'partition', got {self.mode!r}")
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
        return (
            f"shared={self.shared} groups={self.groups} variants={self.variants} "
            f"residuals/group={self.residual_experts_per_group} "
            f"active_residuals={self.top_residuals} shards={self.n_shards} mode={self.mode} "
            f"sharding={'clustered' if self.cluster_shards else 'contiguous'}"
        )
