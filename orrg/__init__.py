"""ORRG: lossless, model-agnostic dense -> mixture-of-experts conversion."""

from .cluster import (
    balanced_kmeans,
    clustering_stats,
    collect_activations,
    collect_hidden_states,
    contiguous_indices,
    kmeans,
)
from .config import MoEConfig, RouterConfig
from .convert import ConversionReport, convert_model, freeze_non_moe, freeze_skeleton, load_converted, save_converted
from .distill import contribution_targets, kl_divergence, on_policy_kd, router_distill_loss
from .ffn import FFNSpec, discover_ffn_modules
from .metrics import (
    collect_metrics,
    conversion_error,
    dead_experts,
    expert_diversity,
    load_imbalance,
)
from .moe import MoELayer
from .router import TokenRouter, update_router_bias
from .scale import ModelShape, estimate_training, converted_scale
from .train import TrainConfig, evaluate_loss, router_distill_total, set_router_distill, train

__version__ = "0.1.0"

__all__ = [
    "MoEConfig",
    "RouterConfig",
    "ConversionReport",
    "convert_model",
    "freeze_non_moe",
    "freeze_skeleton",
    "save_converted",
    "load_converted",
    "FFNSpec",
    "discover_ffn_modules",
    "MoELayer",
    "TokenRouter",
    "update_router_bias",
    "kl_divergence",
    "router_distill_loss",
    "contribution_targets",
    "on_policy_kd",
    "collect_metrics",
    "conversion_error",
    "dead_experts",
    "expert_diversity",
    "load_imbalance",
    "ModelShape",
    "converted_scale",
    "estimate_training",
    "TrainConfig",
    "train",
    "evaluate_loss",
    "set_router_distill",
    "router_distill_total",
    "balanced_kmeans",
    "collect_activations",
    "collect_hidden_states",
    "clustering_stats",
    "contiguous_indices",
    "kmeans",
]
