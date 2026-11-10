"""Numerical smoke check for the ORRG conversion path (not part of the test suite)."""

from __future__ import annotations

import copy
import time

import torch

from orrg import MoEConfig, collect_metrics, convert_model, discover_ffn_modules
from orrg.convert import _capture_outputs
from orrg.distill import contribution_targets, router_distill_loss

ARCHS = {}


def build():
    from transformers import (
        GPT2Config,
        GPT2LMHeadModel,
        LlamaConfig,
        LlamaForCausalLM,
        MistralConfig,
        MistralForCausalLM,
        PhiConfig,
        PhiForCausalLM,
        Qwen2Config,
        Qwen2ForCausalLM,
    )

    ARCHS["llama"] = LlamaForCausalLM(
        LlamaConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=128,
        )
    )
    ARCHS["gpt2"] = GPT2LMHeadModel(
        GPT2Config(n_embd=32, n_head=4, n_layer=2, n_inner=64, vocab_size=128)
    )
    ARCHS["phi"] = PhiForCausalLM(
        PhiConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=128,
        )
    )
    ARCHS["mistral"] = MistralForCausalLM(
        MistralConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=128,
        )
    )
    ARCHS["qwen2"] = Qwen2ForCausalLM(
        Qwen2Config(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=128,
        )
    )


def main() -> None:
    torch.manual_seed(0)
    build()
    ids = torch.randint(0, 128, (4, 12))

    for name, base_model in ARCHS.items():
        dense = copy.deepcopy(base_model)
        specs = discover_ffn_modules(dense)
        cfg = MoEConfig(shared=1, groups=2, variants=4, top_residuals=1, hidden_size=32)
        dense.eval()
        with torch.no_grad():
            ref = dense(ids).logits
        start = time.time()
        model, report = convert_model(dense, cfg, probe={"input_ids": ids})
        model.eval()
        with torch.no_grad():
            got = model(ids).logits
        diff = (ref - got).abs().max().item()
        n_res = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"{name:8s} ffn={len(specs)} carve_err={report.max_carve_error:.2e} "
            f"logit_diff={diff:.2e} total={report.total_params/1e3:.1f}k "
            f"active={report.active_params()/1e3:.1f}k  {time.time()-start:.2f}s"
        )

        model.train()
        out = model(ids)
        loss = out.logits.float().logsumexp(-1).mean()
        loss.backward()
        moe = [m for m in model.modules() if getattr(m, "is_orrg_moe", False)]
        layer = moe[0]
        gate_grad = layer.router.weight.grad
        down = layer.groups[0].residuals[0]
        down_grad = max(p.grad.abs().max() for p in down.parameters() if p.grad is not None)
        print(
            f"         router_grad={gate_grad.abs().max().item():.2e} "
            f"residual_grad={down_grad.abs().max().item():.2e}"
        )
        metrics = collect_metrics(model)
        print(f"         metrics={ {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()} }")

        x = torch.randn(6, 4, 32)
        targets = contribution_targets(layer, x)
        print(
            f"         target_sums={[round(float(t.sum(-1).mean()), 3) for t in targets]} "
            f"distill={float(router_distill_loss(layer, x)):.3f}"
        )

        # partition-mode exactness
        dense2 = copy.deepcopy(base_model)
        cfg2 = MoEConfig(shared=0, groups=4, variants=1, mode="partition", top_k_partition=4, hidden_size=32)
        dense2.eval()
        with torch.no_grad():
            ref2 = dense2(ids).logits
        m2, rep2 = convert_model(dense2, cfg2, probe={"input_ids": ids})
        m2.eval()
        with torch.no_grad():
            got2 = m2(ids).logits
        print(f"         partition exact diff={(ref2-got2).abs().max().item():.2e} err={rep2.max_carve_error:.2e}")


if __name__ == "__main__":
    main()
