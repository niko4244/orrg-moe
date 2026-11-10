"""Function-preserving neuron sharding of a dense FFN.

The FFN is additive in its intermediate neurons:

    FFN(x) = act(x W_up^T + b_up) W_down^T + b_down

so splitting the intermediate axis into disjoint index sets and summing the
shard outputs reproduces the dense block exactly (up to floating-point order).
Shards are materialized as narrow copies of the original module, which keeps
memory at one model's worth of weights and lets each shard run with the
architecture's own fast forward pass.

Index sets may be contiguous (cheap default) or produced by activation
clustering (``orrg.cluster``); the math is identical.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from .ffn import FFNSpec


def shard_bounds(intermediate: int, n_shards: int) -> list[tuple[int, int]]:
    """Split ``intermediate`` into ``n_shards`` contiguous, near-equal ranges."""

    if n_shards < 1:
        raise ValueError("n_shards must be >= 1")
    if n_shards > intermediate:
        raise ValueError(f"cannot shard {intermediate} neurons into {n_shards} shards")
    edges = [round(i * intermediate / n_shards) for i in range(n_shards + 1)]
    return [(edges[i], edges[i + 1]) for i in range(n_shards)]


def shard_widths(intermediate: int, n_shards: int) -> list[int]:
    return [end - start for start, end in shard_bounds(intermediate, n_shards)]


def _take(param: torch.Tensor, axis: int, indices: torch.Tensor) -> torch.Tensor:
    return param.data.index_select(axis, indices.to(param.device)).contiguous().clone()


def narrow_module(
    module: nn.Module,
    spec: FFNSpec,
    start: int | None = None,
    end: int | None = None,
    *,
    indices: torch.Tensor | None = None,
    keep_out_bias: bool,
    zero_down: bool = False,
) -> nn.Module:
    """Return a narrow copy of ``module`` covering a set of intermediate neurons.

    ``indices`` selects the neurons (any order, disjoint sets). ``start``/``end``
    are a convenience for a contiguous range. ``keep_out_bias`` must be true for
    exactly one always-on shard so the hidden-sized output bias is not counted
    once per shard. ``zero_down`` makes the shard output exactly zero (residual
    experts that are grown instead of copied).
    """

    if indices is None:
        if start is None or end is None:
            raise ValueError("provide indices or a start/end range")
        if end <= start:
            raise ValueError("empty shard range")
        indices = torch.arange(start, end)
    if indices.dtype != torch.long:
        indices = indices.long()
    width = int(indices.numel())
    if width == 0:
        raise ValueError("empty shard")

    clone = copy.deepcopy(module)

    with torch.no_grad():
        for slice_spec in spec.up:
            param = clone.get_parameter(slice_spec.name)
            param.data = _take(param, slice_spec.out_axis, indices)
            owner = _owner(clone, slice_spec.name)
            if hasattr(owner, "out_features"):
                owner.out_features = width
            if hasattr(owner, "nf"):  # transformers Conv1D stores output width in .nf
                owner.nf = width

        for bias_name in spec.up_bias:
            param = clone.get_parameter(bias_name)
            param.data = _take(param, 0, indices)

        for slice_spec in spec.down:
            param = clone.get_parameter(slice_spec.name)
            param.data = _take(param, 1 - slice_spec.out_axis, indices)
            owner = _owner(clone, slice_spec.name)
            if hasattr(owner, "in_features"):
                owner.in_features = width
            if hasattr(owner, "nx"):  # transformers Conv1D stores input width in .nx
                owner.nx = width

        for bias_name in spec.out_bias:
            param = clone.get_parameter(bias_name)
            param.data = param.data.clone() if keep_out_bias else torch.zeros_like(param.data)

        if zero_down:
            for slice_spec in spec.down:
                param = clone.get_parameter(slice_spec.name)
                param.data = torch.zeros_like(param.data)
            for bias_name in spec.out_bias:
                param = clone.get_parameter(bias_name)
                param.data = torch.zeros_like(param.data)

    return clone


def _owner(module: nn.Module, param_name: str) -> nn.Module:
    parent = param_name.rsplit(".", 1)[0]
    return module.get_submodule(parent) if parent else module


def shard_parameter_count(module: nn.Module, spec: FFNSpec, width: int) -> int:
    """Parameter count of one shard of ``width`` neurons, without materializing it."""

    total = 0
    params = dict(spec.module.named_parameters())
    for slice_spec in spec.up:
        shape = list(params[slice_spec.name].shape)
        shape[slice_spec.out_axis] = width
        total += int(torch.tensor(shape).prod())
    for slice_spec in spec.down:
        shape = list(params[slice_spec.name].shape)
        shape[1 - slice_spec.out_axis] = width
        total += int(torch.tensor(shape).prod())
    for _ in spec.up_bias:
        total += width
    for bias_name in spec.out_bias:
        total += params[bias_name].numel()  # hidden-sized bias kept once per shard
    return total
