"""Why a bare layer and the full model disagree about prefill b8.

`experiments/batched_dispatch.py` prices one layer on random inputs and places the
batched path ahead up to ~24.6k packed rows. On the full model, prefill b8 — 12,288
rows at k=1, 16,384 at k=2 — loses to grouped. This measures the same weights in three
conditions so the gap is attributed rather than guessed:

1. ``bare``     one MoE layer, random inputs, nothing else resident.
2. ``resident`` all eight MoE layers of a loaded model, called back to back.
3. ``e2e``      the model's own forward on a batch-8 / batch-64, 128-token prompt.

Peak allocated memory is recorded next to every timing: the batched path pads each
residual segment to ``tokens * k`` rows, so its temporaries scale with the token count
and the gap between the bare and resident cases should track that.

    PYTHONPATH=. python experiments/probe_vs_model.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

import orrg.dispatch as dispatch
from experiments.harness import DEVICE, VOCAB, build_model, moe_layers
from orrg import MoEConfig, convert_model

HIDDEN = 768
INTER = 3072
SHARED = 4
GROUPS = 4
VARIANTS = 14
SEGMENTS = GROUPS * (VARIANTS - 1)
PREFILL_SEQ = 128
BATCHES = (8, 64)
KS = (1, 2)
PATHS = ("grouped", "batched")


def timed(fn, reps: int = 3) -> tuple[float, float]:
    """Milliseconds per call and peak MiB above the pre-call baseline."""

    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / reps * 1e3, torch.cuda.max_memory_allocated() / 2**20


def build(k: int):
    cfg = MoEConfig(shared=SHARED, groups=GROUPS, variants=VARIANTS,
                    top_residuals=k, hidden_size=HIDDEN, dispatch="fused")
    model, _ = convert_model(build_model(max_pos=PREFILL_SEQ + 8), cfg, verify=False)
    model.eval()
    return model, moe_layers(model)


def measure(k: int, batch: int) -> dict:
    model, layers = build(k)
    tokens = batch * PREFILL_SEQ
    ids = torch.randint(0, VOCAB, (batch, PREFILL_SEQ), device=DEVICE)
    x = torch.randn(tokens, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
    out = {"k": k, "batch": batch, "tokens": tokens,
           "rows": tokens * (SHARED + GROUPS + k * GROUPS),
           "capacity": tokens * k,
           "padding_gb": SEGMENTS * tokens * k * 1536 * 2 * 2 / 2**30}
    with torch.no_grad():
        for path in PATHS:
            with dispatch.force_path(path):
                bare = timed(lambda: layers[0](x))
                resident = timed(lambda: [layer(x) for layer in layers])
                e2e = timed(lambda: model(input_ids=ids, use_cache=False))
            out[f"bare_{path}_ms"], out[f"bare_{path}_mib"] = bare
            out[f"resident_{path}_ms"], out[f"resident_{path}_mib"] = resident
            out[f"e2e_{path}_ms"], out[f"e2e_{path}_mib"] = e2e
    for name in ("bare", "resident", "e2e"):
        out[f"{name}_ratio"] = out[f"{name}_batched_ms"] / out[f"{name}_grouped_ms"]
    torch.cuda.empty_cache()
    return out


def main() -> int:
    print(f"{torch.cuda.get_device_name(0)}; segments={SEGMENTS}, prefill seq={PREFILL_SEQ}")
    records = [measure(k, batch) for k in KS for batch in BATCHES]

    header = f"{'k':>2} {'batch':>5} {'rows':>6} {'cap':>5} {'pad GB':>6}"
    header += "".join(f" {n + ' g/b':>13} {'mib g/b':>13}" for n in ("bare", "resident", "e2e"))
    print(header)
    for r in records:
        line = (f"{r['k']:>2} {r['batch']:>5} {r['rows']:>6} {r['capacity']:>5} "
                f"{r['padding_gb']:>6.2f}")
        for name in ("bare", "resident", "e2e"):
            line += (f" {r[f'{name}_grouped_ms']:>6.1f}/{r[f'{name}_batched_ms']:>6.1f}"
                     f" {r[f'{name}_ratio']:>5.2f}")
            line += f" {r[f'{name}_grouped_mib']:>6.0f}/{r[f'{name}_batched_mib']:>6.0f}"
        print(line)

    out = Path("experiments/results/probe_vs_model.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "geometry": {"hidden": HIDDEN, "inter": INTER, "shared": SHARED,
                     "groups": GROUPS, "variants": VARIANTS, "segments": SEGMENTS},
        "prefill_seq": PREFILL_SEQ,
        "records": records,
    }, indent=2))
    print(f"\nwrote {out}")

    worst = min(records, key=lambda r: r["e2e_ratio"])
    print(f"\nmodel-level worst: k={worst['k']} batch={worst['batch']} "
          f"e2e {worst['e2e_ratio']:.2f}x (bare {worst['bare_ratio']:.2f}x), "
          f"peak {worst['e2e_batched_mib']:.0f} vs {worst['e2e_grouped_mib']:.0f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
