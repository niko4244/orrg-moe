"""Can CUDA graphs close the b64 decode gap? Capture is blocked inside _grouped_mm.

A. Try to capture the whole fused dispatch. torch 2.11's ``_grouped_mm`` copies
   ``offs`` to the host inside the call, which is illegal during capture; a CPU
   ``offs`` is rejected by the op instead. So the dispatch cannot be graphed.
B. Capture what a graph *can* cover — ``FusedDispatch._prepare`` (router, argsort,
   counts, gather), the part Python and launch overhead live in — and leave the two
   grouped GEMMs eager.
C. Time ``_grouped_mm`` against rows-per-segment. If the cost is per-segment host
   work rather than arithmetic it is flat in rows, and a graph cannot remove it.
D. Layer budget at the decode shape.
E. End-to-end decode throughput: eager, prologue-graphed, and dense-at-active.

    PYTHONPATH=. python experiments/cuda_graph_decode.py
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.harness import (
    DEVICE, HIDDEN, INTER, LAYERS, VOCAB, build_layer, build_model, moe_layers,
)
from orrg import MoEConfig, convert_model
from orrg.dispatch import grouped_mm

TOKENS = (64, 512)
BATCHES = (8, 64)
DECODE_STEPS = 24


def timed(fn, reps: int = 20, warm: int = 5) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def capture(fn, static_in):
    """Capture ``fn(static_in)``; returns (graph, output)."""

    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn(static_in)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(static_in)
    return graph, out


def finish(state, packed, offs, rows, scales, tokens):
    """The two grouped GEMMs and the epilogue, on an already-prepared prologue."""

    gate_up_w, down_w = state._stacked()
    gate_up = grouped_mm(packed, gate_up_w, offs)
    gate, up = gate_up.chunk(2, dim=-1)
    out = grouped_mm(F.silu(gate) * up, down_w, offs)

    n_shards = len(state.shards)
    skeleton = out[: n_shards * tokens].view(n_shards, tokens, -1).sum(0).float()
    contribution = torch.zeros_like(skeleton)
    contribution.index_add_(0, rows, out[n_shards * tokens :].float() * scales.unsqueeze(-1))
    return (skeleton + contribution).to(packed.dtype)


def graphed_forward(state):
    """Drop-in forward whose prologue is a replayed graph, GEMMs left eager."""

    cache = {}

    def forward(flat):
        key = tuple(flat.shape)
        entry = cache.get(key)
        if entry is None:
            static_in = torch.zeros_like(flat)
            static_in.copy_(flat)
            graph, inner = capture(lambda t: state._prepare(t), static_in)
            entry = (static_in, graph) + tuple(inner)
            cache[key] = entry
        static_in, graph, packed, offs, rows, scales = entry
        static_in.copy_(flat)
        graph.replay()
        return finish(state, packed, offs, rows, scales, flat.shape[0])

    return forward


@torch.no_grad()
def decode_tps(model, batch: int, steps: int = DECODE_STEPS, reps: int = 2) -> float:
    prompt = torch.randint(0, VOCAB, (batch, 4), device=DEVICE)

    def run():
        model.generate(prompt, max_new_tokens=steps, min_new_tokens=steps,
                       do_sample=False, use_cache=True, pad_token_id=0)

    for _ in range(2):  # warm up, including any capture
        run()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(reps):
        run()
    torch.cuda.synchronize()
    return batch * steps * reps / (time.perf_counter() - started)


def main() -> int:
    layer, report = build_layer(1)
    state = layer._fused
    n_shards = len(state.shards)
    segments = n_shards + state.groups * (state.variants - 1)
    print(f"RTX 5060 Ti | torch {torch.__version__} | {segments} segments/layer "
          f"({n_shards} shards + {segments - n_shards} residual), "
          f"active {report.active_params()/1e6:.1f}M", flush=True)
    results = {"segments": segments, "capture": {}, "prologue": {}, "grouped": {}}

    with torch.no_grad():
        x = torch.randn(64, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
        try:
            capture(lambda t: layer(t), x)
            results["capture"]["dispatch"] = "OK"
            print("A. capture whole dispatch: OK")
        except Exception as exc:  # noqa: BLE001 - the message is the finding
            results["capture"]["dispatch"] = f"{type(exc).__name__}: {exc}"
            print(f"A. capture whole dispatch: {type(exc).__name__}: {str(exc)[:120]}")

        # B. the prologue a graph can cover
        print("\nB. prologue (_prepare) eager vs graphed:")
        print("| tokens | eager | graphed | saved |")
        print("|---|---|---|---|")
        for tokens in TOKENS:
            flat = torch.randn(tokens, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
            eager = timed(lambda: state._prepare(flat))
            try:
                graph, _ = capture(lambda t: state._prepare(t), flat)
                graphed = timed(graph.replay)
            except Exception as exc:  # noqa: BLE001
                results["prologue"][tokens] = {"error": str(exc)[:200]}
                print(f"| {tokens} | {eager:.3f} | capture failed | {str(exc)[:50]} |")
                continue
            results["prologue"][tokens] = {"eager_ms": eager, "graphed_ms": graphed}
            print(f"| {tokens} | {eager:.3f} ms | {graphed:.3f} ms | "
                  f"{100*(1-graphed/eager):.1f}% |")

        # C. arithmetic or schedule bound?
        print("\nC. one _grouped_mm, 56 segments, varying rows per segment:")
        print("| rows/segment | total rows | grouped_mm | plain GEMM | ratio |")
        print("|---|---|---|---|---|")
        weight = torch.randn(56, HIDDEN, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
        for capacity in (4, 8, 16, 32, 64, 128):
            counts = torch.full((56,), capacity, dtype=torch.long)
            offs = counts.cumsum(0).to(torch.int32).to(DEVICE)
            flat = torch.randn(56 * capacity, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
            grouped = timed(lambda: grouped_mm(flat, weight, offs))
            plain = timed(lambda: flat @ weight[0])
            results["grouped"][capacity] = {"grouped_ms": grouped, "plain_ms": plain}
            print(f"| {capacity} | {56*capacity} | {grouped:.3f} ms | {plain:.3f} ms | "
                  f"{grouped/plain:.1f}x |")

        # D. layer budget at the decode shape
        flat = torch.randn(64, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
        layer_ms = timed(lambda: layer(flat))
        prologue_ms = timed(lambda: state._prepare(flat))
        packed, offs, rows, scales = state._prepare(flat)
        gemms_ms = timed(lambda: finish(state, packed, offs, rows, scales, 64))
        print(f"\nD. layer budget at 64 tokens: total {layer_ms:.3f} ms | "
              f"prologue {prologue_ms:.3f} ms | GEMMs+epilogue {gemms_ms:.3f} ms")
        results["budget_64"] = {"layer_ms": layer_ms, "prologue_ms": prologue_ms,
                                "gemms_ms": gemms_ms}

        # E. end-to-end
        torch.manual_seed(0)
        cfg = MoEConfig(shared=4, groups=4, variants=14, top_residuals=1,
                        hidden_size=HIDDEN, dispatch="fused")
        model, report = convert_model(build_model(INTER), cfg, verify=False)
        fixed = VOCAB * HIDDEN + LAYERS * 4 * HIDDEN * HIDDEN
        inter_active = max(64, round((report.active_params() - fixed) / (LAYERS * 3 * HIDDEN)))
        torch.manual_seed(0)
        dense = build_model(inter_active)
        print(f"\nE. decode tok/s, active {report.active_params()/1e6:.1f}M "
              f"dense {sum(p.numel() for p in dense.parameters())/1e6:.1f}M:")
        print("| batch | eager | prologue-graphed | dense | graph/eager | graph/dense |")
        print("|---|---|---|---|---|---|")
        for batch in BATCHES:
            eager = decode_tps(model, batch)
            layers = moe_layers(model)
            saved = [(lay, lay._fused.forward) for lay in layers]
            for lay in layers:
                lay._fused.forward = graphed_forward(lay._fused)
            graphed = decode_tps(model, batch)
            for lay, fwd in saved:
                lay._fused.forward = fwd
            reference = decode_tps(dense, batch)
            results.setdefault("decode", {})[batch] = {
                "eager": eager, "graphed": graphed, "dense": reference,
            }
            print(f"| {batch} | {eager:.0f} | {graphed:.0f} | {reference:.0f} | "
                  f"{graphed/eager:.2f} | {graphed/reference:.2f} |")

    out = Path("experiments/results/cuda_graph_decode.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
