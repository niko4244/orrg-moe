"""Fused grouped-GEMM expert dispatch (inference path).

``MoELayer``'s reference forward loops over groups and variants in Python: one
gather, three linears and an ``index_add_`` per segment, plus a host sync from
``counts.cumsum(0).tolist()``. For ``s4 g4 v14`` that is ~56 segments per layer.

This module replaces the whole dispatch with one sorted token list and two
``torch._grouped_mm`` calls per layer -- gate and up stacked along N as one
N=2I GEMM, then ``down`` -- with the skeleton shards included as always-full
segments. It is a drop-in for the additive growth
mode used for serving; it carries no autograd path, so the caller must be inside
``torch.no_grad()`` (``MoELayer`` enforces that).

Supported FFNs: gated two-matrix blocks exposing ``gate_proj``/``up_proj``/
``down_proj`` (Llama, Qwen2/Mistral family) with no biases. ``MoEConfig`` rejects
the option for other geometries rather than silently falling back.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F


# ``_grouped_mm`` costs ~30 us per segment whatever the rows, so a decode step pays
# for ~1,000 segment launches. Below this many packed rows the dispatch is instead two
# batched matmuls (shards uniform, residual segments padded to a fixed capacity). That
# padding is `segments / groups` = 13x the real rows and grows with `tokens * k`, which
# is what bounds the win. Measured across the crossing band
# (`experiments/gate_band.py`): 0.57-0.97x of grouped up to 8,192 rows, 1.26-1.59x from
# 10,240 rows up. Rows is a proxy for that capacity, so k=2 crosses at a lower row count
# than k=1; 8,192 is the largest row count where both k measure no loss.
#
# Which path a packed size takes: the single source of truth for the gate, so a
# measurement can force a path without reproducing the decision.
_FORCE_PATH: str | None = None


BATCHED_BELOW_ROWS = 8_192
# ...and not below this either: decode b1 packs 12-16 rows and measures a tie
# (1.03-1.05x, inside the floor), so the cheaper path stays.
MIN_BATCHED_ROWS = 64


def path_for(rows: int) -> str:
    """The dispatch path a packed row count takes in the shipped build."""

    if _FORCE_PATH is not None:
        return _FORCE_PATH
    return "batched" if MIN_BATCHED_ROWS <= rows <= BATCHED_BELOW_ROWS else "grouped"


@contextlib.contextmanager
def force_path(name: str | None):
    """Diagnostic only: pin a path so a sweep can price shapes the gate excludes.

    Serving never sets this; it exists so the report can measure batched and grouped
    on the same shape instead of inferring one of them.
    """

    global _FORCE_PATH
    if name not in (None, "batched", "grouped"):
        raise ValueError(f"unknown dispatch path {name!r}")
    previous, _FORCE_PATH = _FORCE_PATH, name
    try:
        yield
    finally:
        _FORCE_PATH = previous


def grouped_mm(a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """``torch._grouped_mm(a [M, K], b [S, K, N], offs)`` with cumulative offsets."""

    return torch._grouped_mm(a, b, offs=offs)


def residual_capacity(tokens: int, k: int) -> int:
    """Rows per residual segment in the batched layout.

    A single (group, variant) segment can hold every routed row of that group, so
    the only statically safe capacity is ``tokens * k``. Sizing it from the mean
    would be smaller and cheaper, but it is wrong exactly when the router is
    peaked: the surplus rows would be dropped rather than routed. Finding the real
    maximum needs a host sync, which is the cost this path exists to avoid.
    """

    return max(1, tokens * k)


class FusedDispatch:
    """Stacked expert weights plus a single sorted dispatch per forward."""

    def __init__(self, layer):
        self.layer = layer
        cfg = layer.cfg
        self.groups = len(layer.groups)
        self.variants = cfg.variants
        self.k = cfg.top_residuals
        self.shards = list(layer.shared) + [group.base for group in layer.groups]
        self.residuals = [list(group.residuals) for group in layer.groups]
        self.sources = self.shards + [m for group in self.residuals for m in group]
        self._cache = None
        self._key = None
        self._halves = None
        self._halves_key = None

    def _split_halves(self):
        """Contiguous gate/up halves of the stacked weights, for the batched path.

        ``_stacked()`` is what refreshes ``_key`` after an in-place weight change, so
        it must run before the cache check: reading ``_key`` first would compare two
        stale keys after ``copy_``/an optimizer step and serve outdated halves.
        """

        gate_up = self._stacked()[0]
        if self._halves is None or self._halves_key != self._key:
            width = gate_up.shape[-1] // 2
            self._halves = (
                gate_up[..., :width].contiguous(), gate_up[..., width:].contiguous(),
            )
            self._halves_key = self._key
        return self._halves
    def _stacked(self):
        """Stacked [S, H, I] / [S, I, H] weights, rebuilt when a weight changes."""

        # The key needs storage identity, not just the version counter: a
        # ``deepcopy`` of a model keeps the cache object *and* the version
        # counters, so a copied model would otherwise serve the original's
        # stacked weights. In-place edits (optimizer steps, ``copy_``) keep the
        # pointer and bump ``_version``; ``.to(dtype)`` and deepcopies change the
        # pointer. Between them every realistic mutation invalidates.
        key = tuple(
            (p._version, p.data_ptr(), p.dtype, p.device)
            for m in self.sources
            for p in (m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight)
        )
        if self._cache is not None and key == self._key:
            return self._cache

        # Segments may have different widths (clustered shards, expert_width_ratio
        # below 1). Every stack shares one padded width, since shard and residual
        # segments are concatenated into a single weight tensor. Padding the
        # intermediate axis with zeros is exact: a padded neuron gets gate=up=0,
        # so it writes 0 through its zero-weight down rows.
        def pad(weight: torch.Tensor, dim: int, width: int) -> torch.Tensor:
            missing = width - weight.shape[dim]
            if not missing:
                return weight
            return F.pad(weight, (0, missing) if dim == 1 else (0, 0, 0, missing))

        residual = [m for group in self.residuals for m in group]
        widest = max(m.gate_proj.weight.shape[0] for m in self.shards + residual)
        # grouped_mm requires 16-byte-aligned strides, so round the padded
        # intermediate width up to a multiple of 16 / itemsize elements.
        itemsize = self.shards[0].gate_proj.weight.element_size()
        widest = -(-widest * itemsize // 16) * 16 // itemsize

        def stack(modules):
            gate = torch.stack([pad(m.gate_proj.weight.t(), 1, widest) for m in modules])
            up = torch.stack([pad(m.up_proj.weight.t(), 1, widest) for m in modules])
            down = torch.stack([pad(m.down_proj.weight.t(), 0, widest) for m in modules])
            return gate.contiguous(), up.contiguous(), down.contiguous()

        sg, su, sd = stack(self.shards)
        rg, ru, rd = stack(residual)
        # gate and up see the same rows, so one N=2I GEMM replaces two per forward.
        self._cache = (
            torch.cat([torch.cat([sg, su], dim=-1), torch.cat([rg, ru], dim=-1)]),
            torch.cat([sd, rd]),
        )
        self._key = key
        self._halves = None
        self._halves_key = None
        return self._cache

    def _prepare(self, flat: torch.Tensor):
        """Routing and packing shared by ``forward`` and the measurement probes.

        Returns ``(packed, offs, rows_sorted, scales_sorted)``: the rows sent
        through both GEMMs (shard rows first, then routed residual rows sorted by
        segment), their cumulative segment offsets, and the token index and router
        scale of every residual row. Probes that measure an alternative GEMM
        structure or skeleton layout reuse this rather than restating the routing.
        """

        tokens = flat.shape[0]
        acc = torch.float32 if flat.dtype in (torch.bfloat16, torch.float16) else flat.dtype

        if self.k == 0:  # skeleton only: the reference skips routing entirely
            segment = torch.empty(0, dtype=torch.long, device=flat.device)
            rows = segment
            scales = torch.empty(0, dtype=acc, device=flat.device)
        else:
            route = self.layer.router(flat)
            # One row per (token, group, slot), group-major within a token, so the
            # group offsets and the token indices each come from a single arange.
            slots = self.k * self.groups
            offsets = torch.arange(self.groups, device=flat.device).repeat_interleave(self.k)
            segment = (route.index.reshape(tokens, -1) + offsets * (self.variants - 1)).reshape(-1)
            rows = torch.arange(tokens, device=flat.device).repeat_interleave(slots)
            scales = route.weight.to(acc).reshape(-1)

        order = torch.argsort(segment, stable=True)
        segment_sorted, rows_sorted, scales_sorted = segment[order], rows[order], scales[order]

        n_shards = len(self.shards)
        # scatter_add over a fixed-size buffer, not bincount: bincount reads the
        # maximum index back to the host to size its output, which syncs (and is
        # illegal during CUDA graph capture).
        residual_counts = torch.zeros(
            self.groups * (self.variants - 1), dtype=torch.long, device=flat.device,
        )
        residual_counts.scatter_add_(0, segment_sorted, torch.ones_like(segment_sorted))
        counts = torch.cat([
            torch.full((n_shards,), tokens, device=flat.device, dtype=torch.long),
            residual_counts,
        ])
        offs = counts.cumsum(0).to(torch.int32)

        packed = torch.cat([flat.repeat(n_shards, 1), flat[rows_sorted]])
        return packed, offs, rows_sorted, scales_sorted

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        tokens = flat.shape[0]
        packed, offs, rows_sorted, scales_sorted = self._prepare(flat)
        if path_for(packed.shape[0]) == "batched":
            return self._batched(tokens, packed, offs, rows_sorted, scales_sorted)
        return self._grouped(tokens, packed, offs, rows_sorted, scales_sorted)

    def _batched(self, tokens, packed, offs, rows_sorted, scales_sorted):
        """Two batched matmuls: shards as one bmm, residual segments padded to capacity."""

        acc = torch.float32 if packed.dtype in (torch.bfloat16, torch.float16) else packed.dtype
        n_shards = len(self.shards)
        gate_w, up_w = self._split_halves()
        _, down_w = self._stacked()

        shard_rows = packed[: n_shards * tokens].view(n_shards, tokens, -1)
        shard_gate = torch.bmm(shard_rows, gate_w[:n_shards])
        shard_up = torch.bmm(shard_rows, up_w[:n_shards])
        skeleton = torch.bmm(F.silu(shard_gate) * shard_up, down_w[:n_shards]).sum(0).to(acc)

        contribution = torch.zeros_like(skeleton, dtype=acc)
        residual_rows = packed[n_shards * tokens :]
        if residual_rows.shape[0]:
            # offs is cumulative over the shard segments then the residual ones, so
            # subtracting the shard total gives residual-relative segment ends. No
            # host sync anywhere: the capacity comes from the token count.
            ends = offs[n_shards:].to(torch.long) - offs[n_shards - 1].to(torch.long)
            starts = torch.cat([
                torch.zeros(1, dtype=torch.long, device=packed.device), ends[:-1],
            ])
            lanes = torch.arange(residual_capacity(tokens, self.k),
                                 device=packed.device)
            positions = (starts.unsqueeze(1) + lanes.unsqueeze(0)).clamp(
                0, residual_rows.shape[0] - 1,
            )
            valid = lanes.unsqueeze(0) < (ends - starts).clamp(min=0).unsqueeze(1)

            gathered = residual_rows[positions]
            gate = torch.bmm(gathered, gate_w[n_shards:])
            up = torch.bmm(gathered, up_w[n_shards:])
            out = torch.bmm(F.silu(gate) * up, down_w[n_shards:])
            weight = torch.where(valid, scales_sorted[positions],
                                 torch.zeros((), dtype=acc, device=packed.device))
            contribution.index_add_(
                0, rows_sorted[positions].reshape(-1),
                (out.to(acc) * weight.unsqueeze(-1)).reshape(-1, contribution.shape[-1]),
            )
        return (skeleton + contribution).to(packed.dtype)

    def _grouped(self, tokens, packed, offs, rows_sorted, scales_sorted):
        """Two grouped GEMMs: one N=2I for gate/up, one for down."""

        acc = torch.float32 if packed.dtype in (torch.bfloat16, torch.float16) else packed.dtype
        n_shards = len(self.shards)
        gate_up_w, down_w = self._stacked()
        # Two N=I GEMMs only pay on the synthetic harness, where a segment is 384
        # wide; at the real 1120 they are 8-18% slower above 200 MB of intermediate
        # (reports/T3-FOUR-PROBES.md 1d), so the gate/up split is fixed at N=2I.
        gate_up = grouped_mm(packed, gate_up_w, offs)
        gate, up = gate_up.chunk(2, dim=-1)
        out = grouped_mm(F.silu(gate) * up, down_w, offs)

        skeleton = out[: n_shards * tokens].view(n_shards, tokens, -1).sum(0).to(acc)
        residual = out[n_shards * tokens :]
        contribution = torch.zeros_like(skeleton, dtype=acc)
        contribution.index_add_(0, rows_sorted, residual * scales_sorted.unsqueeze(-1))
        return (skeleton + contribution).to(packed.dtype)
