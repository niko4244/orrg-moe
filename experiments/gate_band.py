"""Measure the dispatch crossing in the row band around `BATCHED_BELOW_ROWS`.

The serving sweep brackets the gate only from a distance: its shapes are 96-1,024
rows (decode) and 12,288+ (prefill b8). That left the top edge — 4,096 rows — and the
band above it unmeasured, so the report could only call the bound "conservative". This
measures the band itself on the full model with the diagnostic `force_path` override:
the same weights, batched and grouped adjacent per repeat, warm-up discarded, three
repeats stored, and the order flipped every other repeat so neither arm inherits the
clock ramp.

Each row count is reached with one sequence of that many tokens, so both arms see the
same attention work; only the MoE dispatch changes.

    PYTHONPATH=. python experiments/gate_band.py
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch

import orrg.dispatch as dispatch
from experiments.probe_vs_model import build
from experiments.harness import DEVICE, VOCAB

# k=1 packs 12 rows/token, k=2 packs 16, so these bracket 4,096 on both sides.
TOKENS = (128, 192, 256, 320, 384, 512, 576, 640, 704, 768, 896)
REPEATS = 3
SHARDS = 8
GROUPS = 4
KS = (1, 2)


def time_once(model, ids) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3


def main() -> int:
    print(f"{torch.cuda.get_device_name(0)}; gate {dispatch.MIN_BATCHED_ROWS}-"
          f"{dispatch.BATCHED_BELOW_ROWS} rows")
    records = []
    for k in KS:
        model, _ = build(k)
        for tokens in TOKENS:
            ids = torch.randint(0, VOCAB, (1, tokens), device=DEVICE)
            rows = tokens * (SHARDS + k * GROUPS)
            samples = {"batched": [], "grouped": []}
            for path in ("batched", "grouped"):  # discarded warm-up
                with dispatch.force_path(path):
                    time_once(model, ids)
            for repeat in range(REPEATS):
                order = ("batched", "grouped") if repeat % 2 == 0 else ("grouped", "batched")
                for path in order:
                    with dispatch.force_path(path):
                        samples[path].append(time_once(model, ids))
            ratio = statistics.median(samples["batched"]) / statistics.median(samples["grouped"])
            records.append({"k": k, "tokens": tokens, "rows": rows,
                            "batched_ms": samples["batched"],
                            "grouped_ms": samples["grouped"], "ratio": ratio})
            print(f"| {k} | {tokens} | {rows} | {dispatch.path_for(rows)} | "
                  + " | ".join(f"{v:.1f}" for v in samples["grouped"])
                  + " | " + " | ".join(f"{v:.1f}" for v in samples["batched"])
                  + f" | {ratio:.2f} |")
        del model
        torch.cuda.empty_cache()

    out = Path("experiments/results/gate_band.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "repeats": REPEATS,
        "min_batched_rows": dispatch.MIN_BATCHED_ROWS,
        "batched_below_rows": dispatch.BATCHED_BELOW_ROWS,
        "records": records,
    }, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
