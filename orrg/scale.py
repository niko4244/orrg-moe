"""Parameter / FLOP / memory accounting for planning a 9B-A2B or 27B-A6B run.

The converted model has an *elastic* active-parameter count: the skeleton
(shared + one base shard per group) is always on, and each group can additionally
activate 0..V-1 residual experts. That makes the serving cost a choice rather
than a property of the checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

H100_BF16_PEAK_FLOPS = 989e12  # dense bf16, no sparsity


@dataclass
class ModelShape:
    hidden: int
    intermediate: int
    layers: int
    vocab: int = 0
    n_heads: int = 0
    n_kv_heads: int = 0
    gated: bool = True
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.n_heads == 0:
            self.n_heads = max(1, self.hidden // 128)
        if self.n_kv_heads == 0:
            self.n_kv_heads = self.n_heads
        if self.hidden % self.n_heads:
            raise ValueError("hidden must be divisible by n_heads")

    @property
    def head_dim(self) -> int:
        return self.hidden // self.n_heads

    @property
    def embedding_params(self) -> int:
        return self.vocab * self.hidden * (1 if self.tie_embeddings else 2)

    @property
    def attention_params_per_layer(self) -> int:
        kv = self.n_kv_heads * self.head_dim
        return self.hidden * self.hidden + 2 * self.hidden * kv + self.hidden * self.hidden

    @property
    def ffn_multiplier(self) -> int:
        return 3 if self.gated else 2

    @property
    def ffn_params_per_layer(self) -> int:
        return self.ffn_multiplier * self.hidden * self.intermediate

    @property
    def non_ffn_params(self) -> int:
        per_layer = self.attention_params_per_layer + 2 * self.hidden
        return self.embedding_params + self.layers * per_layer

    @property
    def dense_params(self) -> int:
        return self.non_ffn_params + self.layers * self.ffn_params_per_layer

    def converted(self, shared: int, groups: int, variants: int, top_residuals: int = 1) -> dict:
        """Total and active parameters for an ORRG carving."""

        n_shards = shared + groups * variants
        # The skeleton tiles the seed exactly: its shard width is inter/(shared+groups).
        shard_width = self.intermediate / (shared + groups)
        total_width = shard_width * n_shards
        active_width = shard_width * (shared + groups * (1 + top_residuals))
        ffn_total = self.layers * self.ffn_multiplier * self.hidden * total_width
        ffn_active = self.layers * self.ffn_multiplier * self.hidden * active_width
        total = int(self.non_ffn_params + ffn_total)
        active = int(self.non_ffn_params + ffn_active)
        return {
            "shared": shared,
            "groups": groups,
            "variants": variants,
            "n_shards": n_shards,
            "shard_width": shard_width,
            "total_params": total,
            "active_params": active,
            "active_fraction": active / total,
            "expansion": total / max(self.dense_params, 1),
            "ffn_share_of_total": ffn_total / max(total, 1),
            "flops_per_token": 2 * active,
            "kv_bytes_per_token": 2
            * self.layers
            * self.n_kv_heads
            * self.head_dim
            * 2,  # bf16 k and v
        }

    def active_sweep(self, shared: int, groups: int, variants: int) -> dict[int, dict]:
        return {
            k: self.converted(shared, groups, variants, top_residuals=k)
            for k in range(0, variants)
        }


@dataclass
class TrainingBudget:
    tokens: float
    mfu: float = 0.25
    peak_flops: float = H100_BF16_PEAK_FLOPS
    gpus: int = 1

    def gpu_hours(self, active_params: int) -> float:
        flops = 6.0 * active_params * self.tokens
        return flops / (self.mfu * self.peak_flops) / 3600.0

    def wall_clock_days(self, active_params: int) -> float:
        return self.gpu_hours(active_params) / max(self.gpus, 1) / 24.0


def estimate_training(
    active_params: int,
    tokens: float,
    *,
    mfu: float = 0.25,
    gpus: int = 1,
    peak_flops: float = H100_BF16_PEAK_FLOPS,
) -> dict:
    budget = TrainingBudget(tokens=tokens, mfu=mfu, peak_flops=peak_flops, gpus=gpus)
    return {
        "tokens": tokens,
        "gpu_hours": budget.gpu_hours(active_params),
        "wall_clock_days": budget.wall_clock_days(active_params),
        "petaflop_days": 6.0 * active_params * tokens / 1e15 / 86400,
    }


def finetune_memory_bytes(total_params: int, *, bytes_per_param: float = 16.0) -> dict:
    """Rough full-finetune state: bf16 weights + grads + fp32 master + Adam m/v."""

    weights = 2 * total_params
    full = int(bytes_per_param * total_params)
    return {
        "weights_gb": weights / 1e9,
        "full_finetune_gb": full / 1e9,
        "lora_or_frozen_skeleton_gb": (weights + 4 * total_params * 0.1) / 1e9,
    }


def converted_scale(shape: ModelShape, shared: int, groups: int, variants: int) -> dict:
    sweep = shape.active_sweep(shared, groups, variants)
    return {
        "dense": {
            "params": shape.dense_params,
            "flops_per_token": 2 * shape.dense_params,
        },
        "converted": sweep,
        "expansion": sweep[0]["expansion"],
        "memory": finetune_memory_bytes(sweep[0]["total_params"]),
    }


def format_scale(scale: dict) -> str:
    dense = scale["dense"]
    lines = [
        f"dense seed      : {dense['params']/1e9:8.3f} B total / active",
        f"expansion       : {scale['expansion']:8.2f} x",
        f"full finetune   : {scale['memory']['full_finetune_gb']:8.1f} GB of optimizer state",
        "",
        "k_res   total(B)   active(B)   active%   TFLOPs/token",
    ]
    for k, row in scale["converted"].items():
        lines.append(
            f"{k:5d}   {row['total_params']/1e9:8.3f}   {row['active_params']/1e9:8.3f}   "
            f"{100*row['active_fraction']:6.1f}%   {row['flops_per_token']/1e12:8.3f}"
        )
    return "\n".join(lines)
