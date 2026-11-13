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

import torch
import torch.nn.functional as F


def grouped_mm(a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """``torch._grouped_mm(a [M, K], b [S, K, N], offs)`` with cumulative offsets."""

    return torch._grouped_mm(a, b, offs=offs)


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
        acc = torch.float32 if flat.dtype in (torch.bfloat16, torch.float16) else flat.dtype
        n_shards = len(self.shards)

        packed, offs, rows_sorted, scales_sorted = self._prepare(flat)
        gate_up_w, down_w = self._stacked()
        # One N=2I GEMM. Two N=I GEMMs only pay on the synthetic harness, where a
        # segment is 384 wide; at the real 1120 they are 8-18% slower above 200 MB
        # (reports/T3-FOUR-PROBES.md 1d), so there is no size branch.
        gate_up = grouped_mm(packed, gate_up_w, offs)
        gate, up = gate_up.chunk(2, dim=-1)
        out = grouped_mm(F.silu(gate) * up, down_w, offs)

        skeleton = out[: n_shards * tokens].view(n_shards, tokens, -1).sum(0).to(acc)
        residual = out[n_shards * tokens :]
        contribution = torch.zeros_like(skeleton, dtype=acc)
        contribution.index_add_(0, rows_sorted, residual * scales_sorted.unsqueeze(-1))
        return (skeleton + contribution).to(flat.dtype)
