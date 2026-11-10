"""Model-agnostic discovery of feed-forward blocks.

No architecture registry is required. A block is treated as an FFN when its
2-D parameters describe a widened hidden -> intermediate -> hidden map:

    * some projection writes the intermediate axis (``out == intermediate``),
    * some projection consumes it (``in == intermediate``),
    * ``intermediate > hidden``.

That single rule covers Llama/Qwen/Mistral (gate_proj, up_proj, down_proj),
GPT-2 (Conv1D c_fc/c_proj), OPT/Phi (fc1/fc2) and any other widened MLP,
because the actual forward pass is never re-implemented: blocks are sharded by
slicing their parameters along the intermediate axis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import torch
from torch import nn

try:  # transformers' Conv1D is used by GPT-2 style models
    from transformers.pytorch_utils import Conv1D as _Conv1D
except Exception:  # pragma: no cover - transformers is optional at import time
    _Conv1D = None


@dataclass(frozen=True)
class ParamSlice:
    """A 2-D parameter and which axis is its *output* axis."""

    name: str
    out_axis: int

    def shape_of(self, tensor: torch.Tensor) -> tuple[int, int]:
        return tuple(tensor.shape)  # type: ignore[return-value]


@dataclass
class FFNSpec:
    """Everything needed to slice one dense FFN into neuron shards."""

    name: str
    module: nn.Module = field(repr=False, compare=False)
    hidden: int = 0
    intermediate: int = 0
    up: list[ParamSlice] = field(default_factory=list)
    down: list[ParamSlice] = field(default_factory=list)
    up_bias: list[str] = field(default_factory=list)
    out_bias: list[str] = field(default_factory=list)

    @property
    def dense_params(self) -> int:
        return sum(p.numel() for p in self.module.parameters())

    def __post_init__(self) -> None:
        if not self.up or not self.down:
            raise ValueError(f"{self.name}: incomplete FFN spec")


def _owner_params(module: nn.Module) -> Iterable[tuple[str, nn.Module, str, nn.Parameter]]:
    for owner_name, owner in module.named_modules():
        for pname, param in owner.named_parameters(recurse=False):
            full = f"{owner_name}.{pname}" if owner_name else pname
            yield full, owner, pname, param


def _out_axis(owner: nn.Module, pname: str, param: torch.Tensor) -> int | None:
    """Infer the output axis of a 2-D weight without architecture knowledge."""

    if pname != "weight" or param.ndim != 2:
        return None
    if isinstance(owner, nn.Linear):
        return 0
    if isinstance(owner, nn.Conv1d):
        return 0
    if _Conv1D is not None and isinstance(owner, _Conv1D):
        return 1  # Conv1D stores (in, out)
    # Generic fallback: a bias of length shape[1] means axis 1 is the output axis.
    bias = getattr(owner, "bias", None)
    if isinstance(bias, nn.Parameter) and bias.ndim == 1:
        if bias.numel() == param.shape[1] and bias.numel() != param.shape[0]:
            return 1
        if bias.numel() == param.shape[0] and bias.numel() != param.shape[1]:
            return 0
    # Last resort: assume torch's (out, in) convention.
    return 0


def describe_ffn(module: nn.Module, hidden: int | None = None) -> FFNSpec | None:
    """Return a shardable description of ``module`` or None if it is not an FFN."""

    entries: list[tuple[str, int, tuple[int, ...]]] = []
    for name, owner, pname, param in _owner_params(module):
        if param.ndim != 2:
            continue
        axis = _out_axis(owner, pname, param)
        if axis is None:
            continue
        entries.append((name, axis, tuple(param.shape)))

    if len(entries) < 2:
        return None

    # Which values of the intermediate width appear as both an output and an input?
    kinds: dict[int, set[str]] = {}
    for _, axis, shape in entries:
        kinds.setdefault(shape[axis], set()).add("out")
        kinds.setdefault(shape[1 - axis], set()).add("in")

    candidates = [dim for dim, k in kinds.items() if k == {"in", "out"}]
    if hidden is not None:
        candidates = [d for d in candidates if d != hidden]
    if not candidates:
        return None
    intermediate = max(candidates)

    up = [ParamSlice(n, a) for n, a, s in entries if s[a] == intermediate]
    down = [ParamSlice(n, a) for n, a, s in entries if s[1 - a] == intermediate and s[a] != intermediate]
    if not up or not down:
        return None

    # Hidden width is the input width of the widest up-projection.
    hidden_inferred = max(shape[1 - axis] for name, axis, shape in entries if name in {u.name for u in up})
    if hidden is not None:
        hidden_inferred = hidden
    if intermediate <= hidden_inferred:
        return None  # only widened MLPs are converted; this also rejects attention

    up_names = {u.name for u in up}
    down_names = {d.name for d in down}
    up_bias: list[str] = []
    out_bias: list[str] = []
    for name, owner, pname, param in _owner_params(module):
        if pname != "bias" or param.ndim != 1:
            continue
        parent = name.rsplit(".", 1)[0]
        weight_name = f"{parent}.weight"
        if weight_name in up_names:
            up_bias.append(name)
        elif weight_name in down_names:
            out_bias.append(name)

    return FFNSpec(
        name="",
        module=module,
        hidden=hidden_inferred,
        intermediate=intermediate,
        up=up,
        down=down,
        up_bias=up_bias,
        out_bias=out_bias,
    )


def _model_hidden_size(model: nn.Module) -> int | None:
    config = getattr(model, "config", None)
    for attr in ("hidden_size", "n_embd", "d_model", "model_dim"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


DEFAULT_EXCLUDE = (
    r"(^|\.)(attn|attention|self_attn|self_attention|attentions?)(\.|$)",
    r"(^|\.)(lm_head|embed|embeddings?|wte|wpe|pos_emb)(\.|$)",
    r"(^|\.)(router|gate|gating)(\.|$)",
)


def _matches(name: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, name) for pattern in patterns)


def is_self_contained(spec: FFNSpec) -> bool:
    """True when the module holds only FFN parameters.

    OPT-style models place ``fc1``/``fc2`` directly on the decoder layer, so the
    only module containing both also contains attention and norms. Converting
    such a block would replace attention, so it is rejected explicitly.
    """

    allowed = {p.name for p in spec.up + spec.down}
    allowed.update(spec.up_bias)
    allowed.update(spec.out_bias)
    return all(name in allowed for name, _ in spec.module.named_parameters())


def unsupported_ffn_candidates(model: nn.Module, hidden_size: int | None = None) -> list[FFNSpec]:
    """FFN-shaped blocks that also own non-FFN parameters (split-FFN models)."""

    hidden = hidden_size or _model_hidden_size(model)
    out = []
    for name, module in model.named_modules():
        if not name or getattr(module, "is_orrg_moe", False):
            continue
        spec = describe_ffn(module, hidden=hidden)
        if spec is not None and not is_self_contained(spec):
            spec.name = name
            out.append(spec)
    return out


def discover_ffn_modules(
    model: nn.Module,
    *,
    hidden_size: int | None = None,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> list[FFNSpec]:
    """Find the innermost shardable FFN blocks of any decoder-style model."""

    hidden = hidden_size or _model_hidden_size(model)
    specs: list[FFNSpec] = []
    for name, module in model.named_modules():
        if not name or name.endswith(".weight"):
            continue
        if getattr(module, "is_orrg_moe", False):
            continue
        spec = describe_ffn(module, hidden=hidden)
        if spec is None:
            continue
        spec.name = name
        specs.append(spec)

    # Keep the innermost candidates: drop any spec that contains another spec.
    names = [s.name for s in specs]
    innermost = [s for s in specs if not any(o != s.name and o.startswith(s.name + ".") for o in names)]

    selected: list[FFNSpec] = []
    for spec in innermost:
        if include and not _matches(spec.name, include):
            continue
        # Attention also has a hidden -> wider -> hidden shape (GPT-2 fuses qkv),
        # so name-based exclusion is the reliable structural discriminator.
        if _matches(spec.name, DEFAULT_EXCLUDE):
            continue
        if exclude and _matches(spec.name, exclude):
            continue
        if spec.intermediate < 4:
            continue
        if not is_self_contained(spec):
            continue
        selected.append(spec)
    return selected
