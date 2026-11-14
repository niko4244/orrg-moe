"""Does the "A2B" label cash out in wall-clock? k=1 vs k=2, fused vs reference.

Seven models at one scaled-down 9B geometry (s4 g4 v14, f=1) on a single card:

* ``fused_k{k}``     converted model, ``dispatch="fused"`` (grouped GEMM, inference only)
* ``python_k{k}``    the same weights, reference per-segment Python dispatch
* ``dense_active_k{k}``  a dense model with the same *active* parameter count
* ``dense_total``    a dense model with the same *total* parameter count

k=2 routes two residual experts per group per token instead of one, which adds
active parameters without adding any stored ones. The dense baseline is rebuilt
per k so the comparison stays like-for-like, and the k2/k1 ratio isolates what the
extra active parameters cost on this code path.

Each measurement is repeated three times after a discarded warm-up sweep, and the
ratios are formed per repeat, so the reported spread is run-to-run.

    PYTHONPATH=. python experiments/serving_throughput.py
"""

from __future__ import annotations

import copy
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

import orrg.dispatch as orrg_dispatch
from orrg import MoEConfig, convert_model

HIDDEN, INTER, LAYERS, HEADS, VOCAB = 768, 3072, 8, 8, 8192
SHARED, GROUPS, VARIANTS = 4, 4, 14
KS = (1, 2)
BATCHES = (1, 8, 64)
PREFILL_SEQ = 128
DECODE_STEPS = 24
REPEATS = 3
DEVICE = torch.device("cuda")


def build(inter: int):
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=inter,
        num_hidden_layers=LAYERS, num_attention_heads=HEADS, num_key_value_heads=HEADS,
        tie_word_embeddings=True, max_position_embeddings=PREFILL_SEQ + DECODE_STEPS + 8,
        hidden_act="silu",
    )
    model = LlamaForCausalLM(cfg).to(device=DEVICE, dtype=torch.bfloat16).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def moe_layers(model):
    return [m for m in model.modules() if getattr(m, "is_orrg_moe", False)]


def median_time(fn, reps: int = 3, warm: int = 1) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
    return statistics.median(times)


@torch.no_grad()
def prefill_tps(model, batch: int) -> float:
    ids = torch.randint(0, VOCAB, (batch, PREFILL_SEQ), device=DEVICE)
    return batch * PREFILL_SEQ / median_time(lambda: model(input_ids=ids, use_cache=False))


@torch.no_grad()
def decode_tps(model, batch: int, steps: int = DECODE_STEPS) -> float:
    # generate() owns the cache bookkeeping; a hand-rolled loop silently recomputes.
    prompt = torch.randint(0, VOCAB, (batch, 4), device=DEVICE)

    def run():
        model.generate(prompt, max_new_tokens=steps, min_new_tokens=steps,
                       do_sample=False, use_cache=True, pad_token_id=0)

    run()
    torch.cuda.synchronize()
    started = time.perf_counter()
    reps = 2
    for _ in range(reps):
        run()
    torch.cuda.synchronize()
    return batch * steps * reps / (time.perf_counter() - started)


def sweep(model) -> dict[str, float]:
    out = {}
    for batch in BATCHES:
        out[f"prefill_b{batch}"] = prefill_tps(model, batch)
        out[f"decode_b{batch}"] = decode_tps(model, batch)
    return out


def summarise(samples: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Median with the observed range: timings are noisy on one side only."""

    return {
        key: {
            "median": statistics.median(s[key] for s in samples),
            "min": min(s[key] for s in samples),
            "max": max(s[key] for s in samples),
        }
        for key in samples[0]
    }


def fmt(value: float) -> str:
    return f"{value/1e3:.1f}k" if value >= 1000 else f"{value:.0f}"


def build_arm(k: int):
    """Convert one seed at one k; return (reference, fused copy, report)."""

    torch.manual_seed(0)  # same seed weights for every k
    cfg = MoEConfig(
        shared=SHARED, groups=GROUPS, variants=VARIANTS,
        top_residuals=k, hidden_size=HIDDEN,
    )
    reference, report = convert_model(build(INTER), cfg, verify=False)
    fused = copy.deepcopy(reference)
    for layer in moe_layers(fused):
        layer.dispatch = "fused"
        layer._fused = None

    # Never time a fused path that does not reproduce the reference.
    ids = torch.randint(0, VOCAB, (2, PREFILL_SEQ), device=DEVICE)
    with torch.no_grad():
        a = reference(input_ids=ids, use_cache=False).logits.float()
        b = fused(input_ids=ids, use_cache=False).logits.float()
    assert (a - b).abs().max() <= 0.05 * a.abs().max(), f"fused path diverges at k={k}"
    return reference, fused, report


def main() -> int:
    models, params, active = {}, {}, {}
    report = None
    for k in KS:
        reference, fused, report = build_arm(k)
        models[f"fused_k{k}"] = fused
        models[f"python_k{k}"] = reference
        active[k] = report.active_params(k)
        fixed = VOCAB * HIDDEN + LAYERS * 4 * HIDDEN * HIDDEN
        inter = max(64, round((active[k] - fixed) / (LAYERS * 3 * HIDDEN)))
        torch.manual_seed(0)
        models[f"dense_active_k{k}"] = build(inter)
    torch.manual_seed(0)
    models["dense_total"] = build(
        max(64, round((report.total_params - fixed) / (LAYERS * 3 * HIDDEN)))
    )
    params = {name: sum(p.numel() for p in m.parameters()) for name, m in models.items()}
    print("params: " + ", ".join(f"{n}={v/1e6:.1f}M" for n, v in params.items()), flush=True)
    extra = active[2] - active[1]
    print(
        f"active: k1={active[1]/1e6:.1f}M k2={active[2]/1e6:.1f}M "
        f"(+{extra/1e6:.1f}M, +{100*extra/active[1]:.1f}%)", flush=True,
    )

    # Every fused shape is measured on a real batched arm and a real grouped arm, on the
    # same weights, inside the same repeat and the same batch: a clock ramp, an
    # allocator stall or another process pausing the card then hits both arms instead of
    # biasing one. The shipped arm stays so the tables can show what serving actually
    # does with the gate.
    shipped = orrg_dispatch.BATCHED_BELOW_ROWS
    fused_names = [n for n in models if n.startswith("fused_")]
    all_names = list(models) + [f"{n}_{p}" for n in fused_names for p in ("batched", "grouped")]

    def sweep_pair(model) -> tuple[dict[str, float], dict[str, float]]:
        batched, grouped = {}, {}
        for batch in BATCHES:
            for path, acc in (("batched", batched), ("grouped", grouped)):
                with orrg_dispatch.force_path(path):
                    acc[f"prefill_b{batch}"] = prefill_tps(model, batch)
                    acc[f"decode_b{batch}"] = decode_tps(model, batch)
        return batched, grouped

    # Discard one full pass: the first arm touched on each shape otherwise measures the
    # GPU clock ramp.
    for name, model in models.items():
        if name.startswith("fused_"):
            sweep_pair(model)
        sweep(model)
    print("warm-up sweep done", flush=True)

    samples: dict[str, list[dict[str, float]]] = {name: [] for name in all_names}
    for repeat in range(REPEATS):
        for name, model in models.items():
            samples[name].append(sweep(model))
            if name.startswith("fused_"):
                batched, grouped = sweep_pair(model)
                samples[f"{name}_batched"].append(batched)
                samples[f"{name}_grouped"].append(grouped)
            torch.cuda.empty_cache()
        print(f"repeat {repeat + 1}/{REPEATS} done", flush=True)

    stats = {name: summarise(runs) for name, runs in samples.items()}
    keys = list(stats["fused_k1"])

    def ratio(a: str, b: str) -> dict[str, dict[str, float]]:
        return {
            key: {
                "median": statistics.median(x[key] / y[key] for x, y in zip(samples[a], samples[b])),
                "min": min(x[key] / y[key] for x, y in zip(samples[a], samples[b])),
                "max": max(x[key] / y[key] for x, y in zip(samples[a], samples[b])),
            }
            for key in keys
        }

    names = all_names
    print("\n| metric | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 1) + "|")
    for key in keys:
        print(f"| {key} | " + " | ".join(
            f"{fmt(stats[n][key]['median'])} [{fmt(stats[n][key]['min'])}-"
            f"{fmt(stats[n][key]['max'])}]" for n in names
        ) + " |")

    pairs = [
        ("fused_k1_batched", "fused_k1_grouped"), ("fused_k2_batched", "fused_k2_grouped"),
        ("fused_k1", "fused_k1_grouped"), ("fused_k2", "fused_k2_grouped"),
        ("fused_k1", "python_k1"), ("fused_k2", "python_k2"),
        ("fused_k1", "dense_active_k1"), ("fused_k2", "dense_active_k2"),
        ("fused_k1", "dense_total"), ("fused_k2", "dense_total"),
        ("fused_k2", "fused_k1"),
        ("fused_k2_grouped", "fused_k1_grouped"),
    ]
    print("\nratios (median of 3 repeats [min-max]):")
    for a, b in pairs:
        r = ratio(a, b)
        print(f"  {a} / {b}")
        for key in keys:
            print(f"    {key:12s} {r[key]['median']:.2f} [{r[key]['min']:.2f}-{r[key]['max']:.2f}]")

    out = Path("experiments/results/serving_throughput.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "geometry": {"hidden": HIDDEN, "inter": INTER, "layers": LAYERS,
                     "shared": SHARED, "groups": GROUPS, "variants": VARIANTS, "ks": list(KS)},
        "params": params,
        "active": {str(k): v for k, v in active.items()},
        "repeats": REPEATS,
        "batched_below_rows": {str(key): value for key, value in shipped.items()},
        "min_batched_rows": orrg_dispatch.MIN_BATCHED_ROWS,
        "batched_row_budget": orrg_dispatch.BATCHED_ROW_BUDGET,
        "prefill_seq": PREFILL_SEQ,
        "stats": stats,
        "ratios": {f"{a}_over_{b}": ratio(a, b) for a, b in pairs},
        "samples": samples,
    }, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
