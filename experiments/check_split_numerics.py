"""Is the residual bf16 conversion error the shard split, or ORRG's own code?

``orrg convert`` on Qwen2.5-1.5B measures a 9.06B conversion whose block carve
error is ~5e-2 in bf16 and ~1e-7 in fp32 (tiny-model control). This script runs
the same split *outside* ORRG: take one real Qwen2.5 MLP, evaluate it normally,
then evaluate it as 8 contiguous gate/up/down shards summed in float32, and
compare against the dense block output. If the split reproduces the conversion's
error in bf16 and vanishes in fp32, the conversion is exact and the measured
error belongs to the dtype, not to the construction.

    HF_HUB_OFFLINE=1 PYTHONPATH=. python experiments/check_split_numerics.py
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CHUNKS = 8  # shared=4 + groups=4 shards in the 9.06B geometry

TEXT = """
A mixture of experts layer replaces a single feed-forward network with a set of
smaller ones plus a router that decides which of them each token uses. The appeal
is straightforward: a dense model spends the same compute on every token, while a
sparse model can hold far more parameters than it evaluates. The difficulty is
that capacity is only useful if the router learns to send tokens to the experts
that handle them well, and that is a chicken and egg problem.
"""


def split_error(mlp: torch.nn.Module, x: torch.Tensor, chunks: int = CHUNKS) -> float:
    """Relative max error of "8 shard GEMMs summed" vs the dense block."""

    with torch.no_grad():
        full = mlp(x)
        gate, up, down = mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight
        step = gate.shape[0] // chunks
        acc = torch.zeros(full.shape, dtype=torch.float32, device=full.device)
        for i in range(chunks):
            s = slice(i * step, (i + 1) * step)
            y = F.silu(F.linear(x, gate[s])) * F.linear(x, up[s])
            acc += F.linear(y, down[:, s]).float()
        got = acc.to(full.dtype)
        ref = full.float()
        return float((ref - got.float()).abs().max() / ref.abs().max().clamp_min(1e-6))


def main() -> int:
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    model.eval()
    probe = tokenizer(TEXT, return_tensors="pt", truncation=True, max_length=192)

    captured: dict[str, torch.Tensor] = {}
    handle = model.model.layers[0].mlp.register_forward_pre_hook(
        lambda _m, args: captured.__setitem__("x", args[0].detach())
    )
    with torch.no_grad():
        model(**probe)
    handle.remove()

    mlp = model.model.layers[0].mlp
    x_bf16 = captured["x"].reshape(-1, captured["x"].shape[-1])
    mlp_fp32 = copy.deepcopy(mlp).float()
    x_fp32 = x_bf16.float()

    bf16_split = split_error(mlp, x_bf16)
    fp32_split = split_error(mlp_fp32, x_fp32)
    with torch.no_grad():
        bf16_vs_fp32 = float(
            (mlp(x_bf16).float() - mlp_fp32(x_fp32)).abs().max()
            / mlp_fp32(x_fp32).abs().max().clamp_min(1e-6)
        )

    print(f"tokens                 : {x_bf16.shape[0]}")
    print(f"bf16 split vs bf16 dense: {bf16_split:.3e}")
    print(f"fp32 split vs fp32 dense: {fp32_split:.3e}")
    print(f"bf16 dense vs fp32 dense: {bf16_vs_fp32:.3e}")

    result = {
        "model": MODEL,
        "layer": "model.layers.0.mlp",
        "chunks": CHUNKS,
        "tokens": int(x_bf16.shape[0]),
        "relative_max_error": {
            "bf16_split_vs_bf16_dense": bf16_split,
            "fp32_split_vs_fp32_dense": fp32_split,
            "bf16_dense_vs_fp32_dense": bf16_vs_fp32,
        },
    }
    out = Path("experiments/results/check_split_numerics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote                  : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
