"""Full-model prefill: does the gate/up structure change the k2/k1 ratio?

The layer-level probe (``dispatch_crossover.py``) measures k=1 only and stops at
98k packed rows; the full-model runs saw prefill b64 at k=2 fall from 0.82x of k=1
to 0.72x when the fused N=2I path replaced the three-GEMM one. This measures both
structures across batch sizes for both k, with the arms alternated, so the
structure effect and the run-to-run effect are separated.

The split arm reuses ``FusedDispatch._prepare`` — the same routing and packing the
library runs — and only changes the gate/up GEMM structure, so the two arms differ
in exactly one thing. Its output is asserted equal to the library's before timing.

    PYTHONPATH=. python experiments/prefill_structure.py
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.harness import (
    DEVICE, GROUPS, HIDDEN, SHARED, VARIANTS, build_model, moe_layers,
)
from orrg import MoEConfig, convert_model
from orrg.dispatch import grouped_mm

VOCAB = 8192
PREFILL_SEQ, BATCH, REPS, ROUNDS = 128, 64, 5, 3
BATCHES = (16, 32, 48, 64, 96)


_HALVES: dict[tuple, tuple] = {}


def split_halves(state):
    """Contiguous halves of the stacked gate/up weights, cached by weight identity."""

    halves = _HALVES.get(state._key)
    if halves is None:
        gate_up_w = state._stacked()[0]
        width = gate_up_w.shape[-1] // 2
        halves = (
            gate_up_w[..., :width].contiguous(), gate_up_w[..., width:].contiguous(),
        )
        _HALVES[state._key] = halves
    return halves


def split_forward(state, flat):
    """The library forward with gate/up as two N=I GEMMs instead of one N=2I."""

    packed, offs, rows_sorted, scales_sorted = state._prepare(flat)
    tokens, n_shards = flat.shape[0], len(state.shards)
    acc = torch.float32 if flat.dtype in (torch.bfloat16, torch.float16) else flat.dtype
    _, down_w = state._stacked()
    gate_w, up_w = split_halves(state)
    gate = grouped_mm(packed, gate_w, offs)
    up = grouped_mm(packed, up_w, offs)
    out = grouped_mm(F.silu(gate) * up, down_w, offs)

    skeleton = out[: n_shards * tokens].view(n_shards, tokens, -1).sum(0).to(acc)
    contribution = torch.zeros_like(skeleton, dtype=acc)
    contribution.index_add_(0, rows_sorted,
                            out[n_shards * tokens :] * scales_sorted.unsqueeze(-1))
    return (skeleton + contribution).to(flat.dtype)


def timed_tps(fn, tokens: int, reps: int = REPS) -> float:
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    return tokens / statistics.median(samples)


def main() -> int:
    torch.manual_seed(0)
    ids_by_batch = {b: torch.randint(0, VOCAB, (b, PREFILL_SEQ), device=DEVICE)
                    for b in BATCHES}
    models, originals = {}, {}
    for k in (1, 2):
        cfg = MoEConfig(shared=SHARED, groups=GROUPS, variants=VARIANTS,
                        top_residuals=k, hidden_size=HIDDEN, dispatch="fused")
        model, _ = convert_model(build_model(max_pos=PREFILL_SEQ + 8), cfg, verify=False)
        models[k] = model
        with torch.no_grad():  # one forward builds every layer's fused state
            model(input_ids=ids_by_batch[BATCH], use_cache=False)
        layers = moe_layers(model)
        originals[k] = [(lay, lay._fused.forward) for lay in layers]

        with torch.no_grad():  # assert the split arm is the same function
            reference = model(input_ids=ids_by_batch[BATCH], use_cache=False).logits.float()
            for lay in layers:
                lay._fused.forward = lambda flat, s=lay._fused: split_forward(s, flat)
            split = model(input_ids=ids_by_batch[BATCH], use_cache=False).logits.float()
            for lay, fwd in originals[k]:
                lay._fused.forward = fwd
        drift = (reference - split).abs().max().item()
        assert drift <= 0.05 * reference.abs().max(), f"split arm diverges at k={k}: {drift}"
        print(f"k={k}: split vs fused max|diff| {drift:.4g} at b{BATCH}", flush=True)

    def measure(k: int, split: bool, batch: int) -> float:
        for lay, fwd in originals[k]:
            lay._fused.forward = (lambda flat, s=lay._fused: split_forward(s, flat)) if split else fwd
        ids = ids_by_batch[batch]
        with torch.no_grad():
            return timed_tps(lambda: models[k](input_ids=ids, use_cache=False),
                             batch * PREFILL_SEQ)

    for k in (1, 2):  # discard a sweep so the first arm does not measure the clock ramp
        for batch in BATCHES:
            measure(k, False, batch)
            measure(k, True, batch)

    samples = {(k, b, s): [] for k in (1, 2) for b in BATCHES for s in (False, True)}
    for round_index in range(ROUNDS):  # alternate which arm goes first
        order = (False, True) if round_index % 2 == 0 else (True, False)
        for k in (1, 2):
            for batch in BATCHES:
                for split in order:
                    samples[(k, batch, split)].append(measure(k, split, batch))
        print(f"round {round_index + 1}/{ROUNDS} done", flush=True)

    stats = {key: statistics.median(v) for key, v in samples.items()}
    rows_per_token = {k: SHARED + GROUPS + GROUPS * k for k in (1, 2)}
    # The gate/up output width is the stacked 2 x per-segment width, not 2 x the
    # model's FFN width: conversion splits the FFN across shards and variants.
    width = {k: moe_layers(models[k])[0]._fused._stacked()[0].shape[-1] for k in (1, 2)}
    print(f"\ngate/up intermediate width: k1 {width[1]} k2 {width[2]} elements")
    print(f"prefill (seq {PREFILL_SEQ}), tok/s, median of {ROUNDS} alternating rounds:")
    print("| k | batch | packed rows | intermediate MB | one N=2I | two N=I | two/one |")
    print("|---|---|---|---|---|---|---|")
    ratios = {}
    for k in (1, 2):
        for batch in BATCHES:
            rows = batch * PREFILL_SEQ * rows_per_token[k]
            fused, split = stats[(k, batch, False)], stats[(k, batch, True)]
            ratios[(k, batch)] = split / fused
            print(f"| {k} | {batch} | {rows} | {rows * width[k] * 2 / 1e6:.0f} | "
                  f"{fused:.0f} | {split:.0f} | {split / fused:.2f} |")
    print("\nk2/k1 per batch (one N=2I / two N=I): " + "  ".join(
        f"b{b} {stats[(2, b, False)] / stats[(1, b, False)]:.2f}/"
        f"{stats[(2, b, True)] / stats[(1, b, True)]:.2f}" for b in BATCHES))

    out = Path("experiments/results/prefill_structure.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "batches": list(BATCHES), "seq": PREFILL_SEQ, "rounds": ROUNDS, "reps": REPS,
        "stats": {f"k{k}_b{b}_{'split' if s else 'fused'}": v
                  for (k, b, s), v in stats.items()},
        "split_over_fused": {f"k{k}_b{b}": v for (k, b), v in ratios.items()},
    }, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
