"""Convert any dense decoder-style model into an ORRG mixture of experts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import torch
from torch import nn

from .cluster import ClusterStats, clustering_stats, collect_activations, index_sets_for
from .config import MoEConfig
from .ffn import FFNSpec, discover_ffn_modules, unsupported_ffn_candidates
from .moe import MoELayer


@dataclass
class ConversionReport:
    """Everything measured during conversion, including the lossless check."""

    cfg: MoEConfig
    layer_names: list[str]
    hidden: int
    intermediate: int
    dense_params: int
    total_params: int
    shard_width: int
    per_layer: list[dict] = field(default_factory=list)
    max_carve_error: float = 0.0
    logit_max_abs: float | None = None
    logit_kl: float | None = None
    non_moe_params: int = 0
    skeleton_params: int = 0
    residual_params: int = 0
    router_params: int = 0
    clustering: list[ClusterStats] = field(default_factory=list)

    @property
    def mode(self) -> str:
        return self.cfg.mode

    @property
    def expert_params(self) -> int:
        """Parameters of one residual expert (a single shard)."""

        return int(self.per_layer[0]["expert_params"]) if self.per_layer else 0

    def active_params(self, top_residuals: int | None = None) -> int:
        """Parameters touched by one token, measured from the built modules."""

        if self.mode == "partition":
            n_skeleton = self.cfg.shared + self.cfg.groups
            return int(self.non_moe_params + self.skeleton_params * self.cfg.top_k_partition / n_skeleton)
        k = self.cfg.top_residuals if top_residuals is None else top_residuals
        k = max(0, min(k, self.cfg.residual_experts_per_group))
        active = float(self.non_moe_params + self.skeleton_params + self.router_params)
        if k and self.residual_params:
            per_expert = self.residual_params / self.cfg.n_residuals
            active += k * self.cfg.groups * per_expert
        return int(active)

    def summary(self) -> str:
        lines = [
            f"mode            : {self.mode} ({self.cfg.describe()})",
            f"layers          : {len(self.layer_names)}",
            f"hidden/inter    : {self.hidden} / {self.intermediate}",
            f"shard width     : {self.shard_width}",
            f"dense params    : {self.dense_params/1e9:8.3f} B",
            f"converted total : {self.total_params/1e9:8.3f} B",
            f"active (k={self.cfg.top_residuals})     : {self.active_params()/1e9:8.3f} B",
            f"expansion       : {self.total_params/max(self.dense_params,1):8.2f} x",
            f"routers         : {self.router_params/1e6:8.3f} M (always active)",
            f"carve error     : {self.max_carve_error:.3e} (relative, per layer)",
        ]
        if self.logit_max_abs is not None:
            lines.append(f"logit max |diff|: {self.logit_max_abs:.3e}")
        if self.logit_kl is not None:
            lines.append(f"conversion KL   : {self.logit_kl:.3e}")
        if self.clustering:
            mean_cohesion = sum(c.cohesion for c in self.clustering) / len(self.clustering)
            mean_random = sum(c.baseline for c in self.clustering) / len(self.clustering)
            lines.append(
                f"shard mode      : clustered (cohesion {mean_cohesion:.3f}, "
                f"random baseline {mean_random:.3f})"
            )
        else:
            lines.append("shard mode      : contiguous")
        return "\n".join(lines)


def _set_submodule(model: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, attr = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, attr, module)


def _run(model: nn.Module, probe: Any) -> torch.Tensor:
    with torch.no_grad():
        out = model(**probe) if isinstance(probe, Mapping) else model(probe)
    return out.logits if hasattr(out, "logits") else out


def _capture_outputs(model: nn.Module, specs: list[FFNSpec], probe: Any):
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for spec in specs:
        module = model.get_submodule(spec.name)
        handles.append(
            module.register_forward_hook(
                lambda _m, _i, o, key=spec.name: captured.__setitem__(
                    key, (o[0] if isinstance(o, (tuple, list)) else o).detach()
                )
            )
        )
    try:
        logits = _run(model, probe)
    finally:
        for handle in handles:
            handle.remove()
    return captured, logits


def convert_model(
    model: nn.Module,
    cfg: MoEConfig | None = None,
    *,
    hidden_size: int | None = None,
    verify: bool = True,
    probe: Any | None = None,
    calibration: Iterable[Mapping] | None = None,
) -> tuple[nn.Module, ConversionReport]:
    """Replace every discovered dense FFN with an ORRG MoE layer.

    ``probe`` is a batch (dict of model kwargs or a tensor) used to check that
    the converted model reproduces the dense one. Verification requires it.
    ``calibration`` is required when ``cfg.cluster_shards`` is set: those batches
    are run through the seed to collect pre-down-projection activations, which
    are clustered into balanced index sets per layer.
    """

    cfg = cfg or MoEConfig()
    if verify and probe is None:
        raise ValueError("verify=True requires a probe batch (input_ids or tensor)")

    specs = discover_ffn_modules(
        model,
        hidden_size=hidden_size or cfg.hidden_size,
        include=cfg.include,
        exclude=cfg.exclude,
    )
    if not specs:
        split = unsupported_ffn_candidates(model, hidden_size=hidden_size or cfg.hidden_size)
        hint = (
            f" Found {len(split)} split-FFN blocks (e.g. {split[0].name}) whose module also owns "
            "attention parameters; OPT-style fc1/fc2 at layer level are not supported."
            if split
            else ""
        )
        raise ValueError("no shardable FFN blocks discovered in this model." + hint)

    index_sets_by_layer: dict[str, list[torch.Tensor]] = {}
    cluster_stats: list[ClusterStats] = []
    if cfg.cluster_shards:
        if calibration is None:
            raise ValueError("cluster_shards=True requires calibration batches")
        activations = collect_activations(
            model,
            specs,
            calibration,
            max_tokens=cfg.cluster_tokens,
            seed=cfg.cluster_seed,
        )
        for spec in specs:
            layer_acts = activations.pop(spec.name)
            index_sets = index_sets_for(
                layer_acts,
                spec.intermediate,
                cfg.shared + cfg.groups,
                iters=cfg.cluster_iters,
                seed=cfg.cluster_seed,
            )
            index_sets_by_layer[spec.name] = index_sets
            cluster_stats.append(
                clustering_stats(spec.name, layer_acts, index_sets, seed=cfg.cluster_seed)
            )

    hidden = specs[0].hidden
    intermediate = specs[0].intermediate
    dense_params = sum(p.numel() for p in model.parameters())

    dense_outputs: dict[str, torch.Tensor] = {}
    logits_before = None
    if verify:
        dense_outputs, logits_before = _capture_outputs(model, specs, probe)

    per_layer: list[dict] = []
    for spec in specs:
        layer = MoELayer(
            spec, cfg, layer_name=spec.name, index_sets=index_sets_by_layer.get(spec.name)
        )
        expert_params = sum(p.numel() for p in layer.groups[0].residuals[0].parameters()) if cfg.variants > 1 else 0
        per_layer.append(
            {
                "name": spec.name,
                "hidden": spec.hidden,
                "intermediate": spec.intermediate,
                "dense_params": spec.dense_params,
                "expert_params": expert_params,
                "residual_params": sum(
                    p.numel() for g in layer.groups for e in g.residuals for p in e.parameters()
                ),
            }
        )
        _set_submodule(model, spec.name, layer)

    total_params = sum(p.numel() for p in model.parameters())
    skeleton_total = 0
    residual_total = 0
    router_total = 0
    for entry in per_layer:
        module = model.get_submodule(entry["name"])
        skeleton = sum(p.numel() for p in module.shared.parameters()) + sum(
            p.numel() for group in module.groups for p in group.base.parameters()
        )
        residuals = sum(
            p.numel() for group in module.groups for expert in group.residuals for p in expert.parameters()
        )
        routers = sum(p.numel() for p in module.router.parameters())
        entry["skeleton_params"] = skeleton
        entry["residual_params"] = residuals
        entry["router_params"] = routers
        skeleton_total += skeleton
        residual_total += residuals
        router_total += routers

    report = ConversionReport(
        cfg=cfg,
        layer_names=[s.name for s in specs],
        hidden=hidden,
        intermediate=intermediate,
        dense_params=dense_params,
        total_params=total_params,
        shard_width=int(layer.widths[0]),
        per_layer=per_layer,
        non_moe_params=total_params - skeleton_total - residual_total - router_total,
        skeleton_params=skeleton_total,
        residual_params=residual_total,
        router_params=router_total,
        clustering=cluster_stats,
    )

    if verify:
        converted_outputs, logits_after = _capture_outputs(model, specs, probe)
        errors = []
        for name in report.layer_names:
            ref = dense_outputs[name].float()
            got = converted_outputs[name].float()
            denom = ref.abs().max().clamp_min(1e-6)
            errors.append(float((ref - got).abs().max() / denom))
            for entry in per_layer:
                if entry["name"] == name:
                    entry["carve_error"] = errors[-1]
        report.max_carve_error = max(errors)
        ref, got = logits_before.float(), logits_after.float()
        report.logit_max_abs = float((ref - got).abs().max())
        log_p = torch.log_softmax(got, dim=-1)
        log_q = torch.log_softmax(ref, dim=-1)
        report.logit_kl = float((log_q.exp() * (log_q - log_p)).sum(-1).mean())
        for entry in per_layer:
            entry.setdefault("carve_error", report.max_carve_error)

    if cfg.freeze_non_moe:
        freeze_non_moe(model)
    if cfg.freeze_skeleton:
        freeze_skeleton(model)
    return model, report


def _moe_param_ids(model: nn.Module) -> set[int]:
    ids: set[int] = set()
    for module in model.modules():
        if getattr(module, "is_orrg_moe", False):
            ids.update(id(p) for p in module.parameters())
    return ids


def freeze_non_moe(model: nn.Module) -> nn.Module:
    """Freeze everything outside the converted MoE blocks (attention, embeddings)."""

    keep = _moe_param_ids(model)
    for param in model.parameters():
        param.requires_grad = id(param) in keep
    return model


def freeze_skeleton(model: nn.Module) -> nn.Module:
    """Train only routers and residual experts; freeze the lossless skeleton.

    This is the ORRG-A training recipe: the seed function is preserved
    bit-for-bit by the frozen skeleton, capacity only enters through residuals.
    """

    keep: set[int] = set()
    for module in model.modules():
        if not getattr(module, "is_orrg_moe", False):
            continue
        keep.update(id(p) for p in module.router.parameters())
        for group in module.groups:
            for expert in group.residuals:
                keep.update(id(p) for p in expert.parameters())
    for param in model.parameters():
        param.requires_grad = id(param) in keep
    return model


def moe_parameters(model: nn.Module) -> list[torch.nn.Parameter]:
    ids = _moe_param_ids(model)
    return [p for p in model.parameters() if id(p) in ids]


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_converted(model: nn.Module, cfg: MoEConfig, path: str, *, only_moe: bool = True) -> str:
    """Save an ORRG checkpoint. ``only_moe=False`` saves the whole model."""

    prefix = ""
    names: set[str] = set()
    if only_moe:
        for module_name, module in model.named_modules():
            if not getattr(module, "is_orrg_moe", False):
                continue
            prefix = f"{module_name}." if module_name else ""
            names.update(f"{prefix}{n}" for n, _ in module.named_parameters())
            names.update(f"{prefix}{n}" for n, _ in module.named_buffers())
    state = model.state_dict()
    if only_moe:
        state = {k: v for k, v in state.items() if k in names}
    torch.save({"format": "orrg-1", "config": cfg.to_dict(), "state_dict": state}, path)
    return path


def load_converted(
    model: nn.Module,
    cfg: MoEConfig | None = None,
    path: str | None = None,
    *,
    strict: bool = False,
) -> tuple[nn.Module, ConversionReport]:
    """Convert ``model`` and restore a previously saved ORRG checkpoint."""

    payload = torch.load(path, map_location="cpu") if path else {}
    cfg = cfg or (MoEConfig.from_dict(payload["config"]) if payload else MoEConfig())
    model, report = convert_model(model, cfg, verify=False)
    if payload:
        model.load_state_dict(payload["state_dict"], strict=strict)
    return model, report
