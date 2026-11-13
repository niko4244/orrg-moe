"""Can a residual expert ever leave zero?

With ``residual_init="zero"`` the down-projection is zero, so the task loss sends
no gradient to the residual's gate/up projections -- and in additive mode the
router's weight gradient is also zero, because it multiplies a zero residual. The
model then trains its skeleton and leaves 75% of its parameters frozen.

This trains a tiny converted model under each symmetry breaker the repo ships and
measures, every 25 steps, the residual output norm relative to its group base and
the cosine between them. A residual that never leaves zero stays at norm 0 and
undefined cosine; a working breaker should show rising norm and falling cosine.

Arms: zero, scaled (down x0.05), noise (up jitter only), distill (router CE), and
ortho (explicit up-projection push, which ships in ``orrg/train.py`` but was not in
the original request -- included because it is designed for exactly this pathology).

    PYTHONPATH=. python experiments/residual_dynamics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from t1_capacity import Stream, read_bytes  # noqa: E402

from orrg import MoEConfig, convert_model  # noqa: E402
from orrg.train import TrainConfig, moe_layers, train  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

HIDDEN, INTER, LAYERS, HEADS, VOCAB = 128, 512, 2, 4, 256
BATCH, SEQ, STEPS = 8, 128, 150
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARMS: dict[str, dict] = {
    "zero": {},
    "scaled": {"residual_init": "scaled"},
    "noise": {"residual_noise": 0.01},
    "distill": {"router_distill_coef": 0.5},
    "ortho": {},
}


def build():
    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        tie_word_embeddings=True,
        max_position_embeddings=512,
        hidden_act="silu",
    )
    return LlamaForCausalLM(cfg).to(DEVICE)


@torch.no_grad()
def residual_stats(model, probe) -> dict:
    layers = moe_layers(model)
    captured: dict[int, torch.Tensor] = {}
    handles = [
        layer.register_forward_pre_hook(
            lambda _m, args, i=i: captured.__setitem__(i, args[0].detach())
        )
        for i, layer in enumerate(layers)
    ]
    was_training = model.training
    model.eval()
    try:
        model(**probe)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    norms, cosines, zeros, total = [], [], 0, 0
    for i, layer in enumerate(layers):
        x = captured[i]
        for group in layer.groups:
            base = group.base(x).float()
            base_norm = base.norm(dim=-1).mean().clamp_min(1e-12)
            for expert in group.residuals:
                out = expert(x).float()
                norms.append(float(out.norm(dim=-1).mean() / base_norm))
                cosines.append(
                    float(F.cosine_similarity(out.flatten(1), base.flatten(1), dim=-1).mean())
                )
                zeros += int(float(out.abs().max()) == 0.0)
                total += 1
    return {
        "norm_ratio": sum(norms) / len(norms),
        "cosine_to_base": sum(cosines) / len(cosines),
        "zero_fraction": zeros / max(total, 1),
        "per_expert_norm": norms,
    }


def grad_mass(model) -> dict:
    def mass(predicate) -> float:
        total = 0.0
        for layer in moe_layers(model):
            for group in layer.groups:
                for expert in group.residuals:
                    for name, param in expert.named_parameters():
                        if predicate(name) and param.grad is not None:
                            total += float(param.grad.abs().sum())
        return total

    return {
        "gate_up": mass(lambda n: "down" not in n),
        "down": mass(lambda n: "down" in n),
        "router": sum(
            float(p.grad.abs().sum())
            for layer in moe_layers(model)
            for p in layer.router.parameters()
            if p.grad is not None
        ),
    }


def run_arm(name: str, overrides: dict, batches, probe) -> dict:
    torch.manual_seed(0)
    model = build()
    moe = MoEConfig(
        shared=1, groups=2, variants=4, hidden_size=HIDDEN,
        **{k: v for k, v in overrides.items() if k != "router_distill_coef"},
    )
    converted, _ = convert_model(model, moe, verify=False)
    trace = [{"step": 0, **residual_stats(converted, probe)}]

    loss = converted(**batches[0]).loss
    loss.backward()
    grads = grad_mass(converted)
    converted.zero_grad(set_to_none=True)

    if name == "ortho":
        stats = train(
            converted, batches, STEPS,
            TrainConfig(lr=2e-3, orth_step=1e-3, trace_every=25, log_every=0),
            tag=name,
            trace=lambda step: trace.append({"step": step, **residual_stats(converted, probe)}),
        )
    else:
        stats = train(
            converted, batches, STEPS,
            TrainConfig(
                lr=2e-3,
                router_distill_coef=float(overrides.get("router_distill_coef", 0.0)),
                trace_every=25,
                log_every=0,
            ),
            tag=name,
            trace=lambda step: trace.append({"step": step, **residual_stats(converted, probe)}),
        )

    print(
        f"{name:8s} grad(gate/up)={grads['gate_up']:.3e} grad(down)={grads['down']:.3e} "
        f"grad(router)={grads['router']:.3e} | "
        f"norm {trace[0]['norm_ratio']:.3e} -> {trace[-1]['norm_ratio']:.3e} | "
        f"cos {trace[0]['cosine_to_base']:+.3f} -> {trace[-1]['cosine_to_base']:+.3f} | "
        f"zero@{trace[-1]['zero_fraction']:.0%}",
        flush=True,
    )
    return {"arm": name, "grads": grads, "trace": trace, "train": stats}


def main() -> int:
    tokens = read_bytes(4_000_000)
    stream = Stream(tokens, SEQ, BATCH)
    batches = [stream.make_batch(stream.offsets(seed=0, step=i)) for i in range(STEPS + 2)]
    probe = batches[0]

    rows = [run_arm(name, overrides, batches, probe) for name, overrides in ARMS.items()]

    out = Path("experiments/results/residual_dynamics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"hidden": HIDDEN, "inter": INTER, "layers": LAYERS,
                   "steps": STEPS, "batch": BATCH, "seq": SEQ},
        "rows": rows,
    }, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
