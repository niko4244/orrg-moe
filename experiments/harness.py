"""Shared geometry, builders and timing for the dispatch probes.

Every probe previously restated the model geometry, the Llama config and its own
interleaved timer. They live here once so a geometry change cannot leave one probe
measuring a different model than the report describes.
"""

from __future__ import annotations

import statistics

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from orrg import MoEConfig, convert_model

HIDDEN, INTER, LAYERS, HEADS, VOCAB = 768, 3072, 8, 8, 8192
SHARED, GROUPS, VARIANTS = 4, 4, 14
DEVICE = torch.device("cuda")


def build_model(inter: int = INTER, max_pos: int = 256) -> LlamaForCausalLM:
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=inter,
        num_hidden_layers=LAYERS, num_attention_heads=HEADS, num_key_value_heads=HEADS,
        tie_word_embeddings=True, max_position_embeddings=max_pos, hidden_act="silu",
    )
    model = LlamaForCausalLM(cfg).to(device=DEVICE, dtype=torch.bfloat16).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def moe_layers(model) -> list:
    return [m for m in model.modules() if getattr(m, "is_orrg_moe", False)]


def build_layer(k: int = 1, inter: int = INTER):
    """One converted model's first MoE layer, with its fused state built."""

    cfg = MoEConfig(shared=SHARED, groups=GROUPS, variants=VARIANTS,
                    top_residuals=k, hidden_size=HIDDEN, dispatch="fused")
    model, report = convert_model(build_model(inter), cfg, verify=False)
    layer = moe_layers(model)[0]
    with torch.no_grad():
        layer(torch.randn(32, HIDDEN, device=DEVICE, dtype=torch.bfloat16))
    return layer, report


def timed_ms(fn, reps: int = 15, warm: int = 5) -> float:
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


def timed_pair(first, second, reps: int = 15, warm: int = 5) -> tuple[float, float]:
    """Median ms for both callables, interleaved so order cannot bias them."""

    for _ in range(warm):
        first()
        second()
    torch.cuda.synchronize()
    a, b = [], []
    for _ in range(reps):
        for fn, acc in ((first, a), (second, b)):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            acc.append(start.elapsed_time(end))
    return statistics.median(a), statistics.median(b)
