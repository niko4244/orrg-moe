"""Two dispatch measurements on one MoE layer (no attention).

1. **Gate/up GEMM shape.** One N=2I grouped GEMM (what the library does) against
   two N=I ones, from decode to 98k packed rows at k=1. The library prologue is
   shared with the library itself (``FusedDispatch._prepare``), so only the GEMM
   structure differs; the two arms are asserted to produce the same numbers.
   This covers k=1 up to 98k rows only — the full-model, k=2 and larger-prefill
   sweep is ``experiments/prefill_structure.py``, whose answer is that the split
   form wins once the [M, 2I] intermediate passes ~1.2 GB.

2. **Skeleton replication.** The fused path feeds the skeleton shards as
   ``flat.repeat(n_shards, 1)``, so the same token rows go through the GEMM eight
   times. Compare against one dense FFN over concatenated shard weights — same
   function, no replication, eight fewer segments — and assert the dense variant
   reproduces the layer before timing it.

    PYTHONPATH=. python experiments/dispatch_crossover.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.harness import DEVICE, GROUPS, HIDDEN, build_layer, timed_ms, timed_pair
from orrg.dispatch import grouped_mm

TOKENS = (16, 64, 512, 2048, 4096, 8192)
PROBE_TOKENS = (64, 512, 2048)
TOL = 2e-2


def gemm_forward(state, flat, split_w=None):
    """The layer through the library prologue, with a selectable gate/up structure."""

    packed, offs, rows, scales = state._prepare(flat)
    tokens, n_shards = flat.shape[0], len(state.shards)
    gate_up_w, down_w = state._stacked()
    if split_w is None:
        gate_up = grouped_mm(packed, gate_up_w, offs)
        gate, up = gate_up.chunk(2, dim=-1)
    else:
        gate = grouped_mm(packed, split_w[0], offs)
        up = grouped_mm(packed, split_w[1], offs)
    out = grouped_mm(F.silu(gate) * up, down_w, offs)

    skeleton = out[: n_shards * tokens].view(n_shards, tokens, -1).sum(0).float()
    contribution = torch.zeros(tokens, HIDDEN, dtype=torch.float32, device=DEVICE)
    contribution.index_add_(0, rows, out[n_shards * tokens :].float() * scales.unsqueeze(-1))
    return (skeleton + contribution).to(flat.dtype)


def residual_only(state, flat):
    """Residual contribution alone, through the same prologue and grouped stacks."""

    packed, offs, rows, scales = state._prepare(flat)
    gate_up_w, down_w = state._stacked()
    tokens, n_shards = flat.shape[0], len(state.shards)
    residual_offs = offs[n_shards:] - offs[n_shards - 1]
    gate_up = grouped_mm(packed[n_shards * tokens :], gate_up_w[n_shards:], residual_offs)
    gate, up = gate_up.chunk(2, dim=-1)
    out = grouped_mm(F.silu(gate) * up, down_w[n_shards:], residual_offs)

    contribution = torch.zeros(tokens, HIDDEN, dtype=torch.float32, device=DEVICE)
    contribution.index_add_(0, rows, out.float() * scales.unsqueeze(-1))
    return contribution


def dense_skeleton(layer):
    """Shard weights scattered into one dense FFN: the skeleton without replication."""

    width = layer.skeleton_width
    gate = torch.zeros(HIDDEN, width, device=DEVICE, dtype=torch.bfloat16)
    up = torch.zeros(HIDDEN, width, device=DEVICE, dtype=torch.bfloat16)
    down = torch.zeros(width, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
    modules = list(layer.shared) + [group.base for group in layer.groups]
    for module, indices in zip(modules, layer.index_sets):
        idx = indices.to(DEVICE)
        gate[:, idx] = module.gate_proj.weight.t()
        up[:, idx] = module.up_proj.weight.t()
        down[idx, :] = module.down_proj.weight.t()
    return gate, up, down


def assert_close(name: str, got: torch.Tensor, want: torch.Tensor) -> float:
    drift = (got.float() - want.float()).abs().max().item()
    scale = want.float().abs().max().item() or 1.0
    assert drift <= TOL * scale, f"{name} diverges: max|diff| {drift:.4g} vs scale {scale:.4g}"
    return drift


def main() -> int:
    layer, report = build_layer(1)
    state = layer._fused
    n_shards = len(state.shards)
    rows_per_token = n_shards + GROUPS * layer.cfg.top_residuals
    print(f"shards={n_shards} rows/token={rows_per_token} active={report.active_params()/1e6:.1f}M")

    with torch.no_grad():
        gate_up_w, _ = state._stacked()
        width = gate_up_w.shape[-1] // 2
        split_w = (gate_up_w[..., :width].contiguous(), gate_up_w[..., width:].contiguous())
        x0 = torch.randn(32, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
        library = layer(x0)
        print(f"equivalence: split {assert_close('split', gemm_forward(state, x0, split_w), library):.3g} "
              f"| dense-skeleton {assert_close('dense', dense_variant_seed(layer, state, x0), library):.3g}")

        print("\n1. gate/up GEMM shape, k=1 (ms per layer call):")
        print("| tokens | packed rows | one N=2I | two N=I | two/one |")
        print("|---|---|---|---|---|")
        crossover = {}
        for tokens in TOKENS:
            x = torch.randn(tokens, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
            one, two = timed_pair(lambda: layer(x), lambda: gemm_forward(state, x, split_w))
            crossover[tokens] = {"rows": tokens * rows_per_token, "one_ms": one, "two_ms": two}
            print(f"| {tokens} | {tokens * rows_per_token} | {one:.3f} | {two:.3f} | "
                  f"{two / one:.2f} |")

        print("\n2. skeleton replication (ms per layer call):")
        print("| tokens | library forward | dense-skeleton | saved | repeat() alone |")
        print("|---|---|---|---|---|")
        weights = dense_skeleton(layer)
        probe = {}
        for tokens in PROBE_TOKENS:
            x = torch.randn(tokens, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
            library_ms, dense_ms = timed_pair(lambda: layer(x), lambda: dense_forward(state, weights, x))
            repeat_ms = timed_ms(lambda: x.repeat(n_shards, 1))
            probe[tokens] = {"library_ms": library_ms, "dense_ms": dense_ms, "repeat_ms": repeat_ms}
            print(f"| {tokens} | {library_ms:.3f} | {dense_ms:.3f} | "
                  f"{100 * (1 - dense_ms / library_ms):.1f}% | {repeat_ms:.3f} |")

    out = Path("experiments/results/dispatch_crossover.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "rows_per_token": rows_per_token,
        "gate_up": {str(k): v for k, v in crossover.items()},
        "skeleton": {str(k): v for k, v in probe.items()},
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


def dense_forward(state, weights, flat):
    gate, up, down = weights
    act = F.silu(flat @ gate) * (flat @ up)
    return ((act @ down).float() + residual_only(state, flat)).to(flat.dtype)


def dense_variant_seed(layer, state, flat):
    return dense_forward(state, dense_skeleton(layer), flat)


if __name__ == "__main__":
    raise SystemExit(main())
