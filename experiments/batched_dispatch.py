"""Can the dispatch be done in four launches instead of 2 x 60 segments?

``torch._grouped_mm`` costs ~30 us per segment whatever the rows (probe §1e), so a
decode step pays ~1,000 segment launches. This prototype replaces the segment loop
with two batched matmuls:

* shard segments are uniform (every shard sees every token), so they are one
  ``bmm`` over the eight stacked shard weights - no padding, no waste;
* residual segments are small and ragged, so they are gathered into a fixed
  ``[segments, capacity, K]`` block, batched, and scattered back with the invalid
  lanes masked out. Capacity is derived from the token count, not from a host-side
  max, so nothing syncs.

Everything else (router, argsort, packing, epilogue) is the library's own
``FusedDispatch._prepare``, and the result is asserted against ``layer(x)`` before
anything is timed.

    PYTHONPATH=. python experiments/batched_dispatch.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.harness import DEVICE, HIDDEN, build_layer, timed_pair
from orrg.dispatch import grouped_mm

TOKENS = (8, 64, 512, 2048, 4096, 8192)
TOL = 2e-2


def residual_capacity(tokens: int, k: int = 1) -> int:
    """Statically safe rows per residual segment: every routed row of a group."""

    return max(1, tokens * k)


def batched_forward(state, flat: torch.Tensor) -> torch.Tensor:
    packed, offs, rows_sorted, scales = state._prepare(flat)
    tokens = flat.shape[0]
    gate_up, down = state._stacked()
    width = gate_up.shape[-1] // 2
    gate_w, up_w = gate_up[..., :width], gate_up[..., width:]
    n_shards = len(state.shards)

    # Shards: uniform rows, so a plain batched matmul over the stacked weights.
    shard_rows = packed[: n_shards * tokens].view(n_shards, tokens, -1)
    shard_gate = torch.bmm(shard_rows, gate_w[:n_shards])
    shard_up = torch.bmm(shard_rows, up_w[:n_shards])
    shard_out = torch.bmm(F.silu(shard_gate) * shard_up, down[:n_shards])
    skeleton = shard_out.sum(0).float()

    residual_rows = packed[n_shards * tokens :]
    rows_w = gate_w[n_shards:]
    segments = rows_w.shape[0]
    if segments == 0:
        return skeleton.to(flat.dtype)

    ends = offs[n_shards:].to(torch.long)
    starts = torch.cat([offs[n_shards - 1 : n_shards].to(torch.long), ends[:-1]])
    shard_total = offs[n_shards - 1].to(torch.long)
    starts, ends = starts - shard_total, ends - shard_total
    counts = (ends - starts).clamp(min=0)

    capacity = residual_capacity(tokens, state.k)
    lanes = torch.arange(capacity, device=flat.device)
    positions = starts.unsqueeze(1) + lanes.unsqueeze(0)
    valid = lanes.unsqueeze(0) < counts.unsqueeze(1)
    positions = positions.clamp(0, max(residual_rows.shape[0] - 1, 0))

    gathered = residual_rows[positions]  # [segments, capacity, K]
    gate = torch.bmm(gathered, rows_w)
    up = torch.bmm(gathered, up_w[n_shards:])
    out = torch.bmm(F.silu(gate) * up, down[n_shards:])  # [segments, capacity, H]

    token_of = rows_sorted[positions]
    # One router scale per row, gathered into the segment layout and zeroed on the
    # padding lanes so the scatter cannot add anything spurious.
    weight = torch.where(valid, scales[positions], torch.zeros((), dtype=scales.dtype,
                                                             device=flat.device))
    contribution = torch.zeros(tokens, HIDDEN, dtype=torch.float32, device=flat.device)
    contribution.index_add_(
        0, token_of.reshape(-1), (out.float() * weight.unsqueeze(-1)).reshape(-1, HIDDEN),
    )
    return (skeleton + contribution).to(flat.dtype)


def main() -> int:
    layer, report = build_layer(1)
    state = layer._fused
    n_shards = len(state.shards)
    segments = state.groups * (state.variants - 1)
    print(f"shards={n_shards} residual segments={segments} active={report.active_params()/1e6:.1f}M")

    with torch.no_grad():
        for tokens in TOKENS:
            x = torch.randn(tokens, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
            want = layer(x)
            got = batched_forward(state, x)
            drift = (got.float() - want.float()).abs().max().item()
            scale = want.float().abs().max().item() or 1.0
            assert drift <= TOL * scale, f"{tokens} tokens diverges: {drift:.4g} / {scale:.4g}"

        print("\n| tokens | grouped_mm | batched bmm | bmm/grouped |")
        print("|---|---|---|---|")
        rows = {}
        for tokens in TOKENS:
            x = torch.randn(tokens, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
            grouped, batched = timed_pair(lambda: layer(x), lambda: batched_forward(state, x))
            capacity = residual_capacity(tokens, state.k)
            rows[tokens] = {"grouped_ms": grouped, "batched_ms": batched, "capacity": capacity}
            print(f"| {tokens} | {grouped:.3f} ms | {batched:.3f} ms | {batched / grouped:.2f} |")

    out = Path("experiments/results/batched_dispatch.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": torch.cuda.get_device_name(0), "rows": rows}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
