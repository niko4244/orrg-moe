"""Does the batched dispatch beat grouped, on the shipped code?

The first version of this probe compared a full MoE layer against a stand-alone
``batched_forward`` prototype. That measured a different quantity from the one the
gate asks about — routing plus dispatch on one side, dispatch alone on the other — and
it put the crossover roughly 3x higher in row count than the model-level sweep finds.
Both arms here are the library's own ``_grouped`` and ``_batched`` on identical
``_prepare`` output, with equality asserted, so the numbers answer the same question
``orrg.dispatch.path_for`` answers.

    PYTHONPATH=. python experiments/batched_dispatch.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.harness import HIDDEN, build_layer, timed_pair

TOKENS = (8, 64, 512, 1_024, 2_048, 4_096, 8_192)
TOL = 2e-3  # bf16 accumulation differences between two summation orders


def main() -> int:
    layer, _ = build_layer(k=1)
    st = layer._fused
    print(f"{torch.cuda.get_device_name(0)}; shards={len(st.shards)} "
          f"residual segments={st.groups * (st.variants - 1)}")

    with torch.no_grad():
        rows = {}
        for tokens in TOKENS:
            x = torch.randn(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
            packed, offs, sorted_rows, scales = st._prepare(x)
            want = st._grouped(tokens, packed, offs, sorted_rows, scales)
            got = st._batched(tokens, packed, offs, sorted_rows, scales)
            drift = (got.float() - want.float()).abs().max().item()
            scale = want.float().abs().max().item() or 1.0
            assert drift <= TOL * scale, f"{tokens} tokens diverges: {drift:.4g} / {scale:.4g}"

            grouped, batched = timed_pair(
                lambda: st._grouped(tokens, packed, offs, sorted_rows, scales),
                lambda: st._batched(tokens, packed, offs, sorted_rows, scales),
            )
            rows[tokens] = {
                "grouped_ms": grouped,
                "batched_ms": batched,
                "rows": int(packed.shape[0]),
                "capacity": tokens,
            }
            print(f"| {tokens} | {packed.shape[0]} | {grouped:.2f} ms | "
                  f"{batched:.2f} ms | {batched / grouped:.2f} |")
    out = Path("experiments/results/batched_dispatch.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": torch.cuda.get_device_name(0), "rows": rows}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
