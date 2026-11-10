"""Toy-scale head-to-head: dense continued training vs naive upcycling vs ORRG.

Protocol (kept deliberately small; it is a smoke test of the *method*, not
evidence about 9B models):

1. Train one dense byte-level seed on a 3-domain synthetic corpus
   (prose / code / arithmetic) so that specialization is measurable.
2. Convert the seed three ways:
      dense_continued  - no conversion, keep training the dense model
      naive_upcycle    - residual experts are copies (function is *not* preserved)
      orrg_growth      - residual experts have zeroed down-projections (lossless)
3. Fine-tune each arm for the same number of steps on the same batches.
4. Report validation loss overall and per domain, conversion KL, routing health
   and expert diversity.

Run: PYTHONPATH=. python experiments/toy_scale.py --steps 250 --seeds 2
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from orrg import MoEConfig, collect_metrics, convert_model
from orrg.metrics import conversion_error

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROSE_WORDS = (
    "the of and to in is that for it as with was on be at by this had not are but from "
    "or have an they which one you were all we when your can said there use each other "
    "time people year way day man thing woman life child world school state family student"
).split()

CODE_TOKENS = [
    "def", "return", "for", "in", "range", "if", "else", "while", "class", "self",
    "import", "from", "as", "with", "open", "print", "None", "True", "False", "lambda",
]


def make_prose(rng: random.Random, n: int) -> str:
    out = []
    while sum(len(x) for x in out) < n:
        sentence = " ".join(rng.choice(PROSE_WORDS) for _ in range(rng.randint(6, 16)))
        out.append(sentence.capitalize() + rng.choice([".", ".", "?", "!"]))
    return " ".join(out)[:n]


def make_code(rng: random.Random, n: int) -> str:
    out = []
    while sum(len(x) for x in out) < n:
        indent = "    " * rng.randint(0, 2)
        line = f"{indent}{rng.choice(CODE_TOKENS)} "
        if rng.random() < 0.4:
            line += f"x{rng.randint(0, 9)} = {rng.randint(0, 99)}"
        elif rng.random() < 0.7:
            line += f"f(x{rng.randint(0, 9)}, {rng.randint(0, 9)})"
        else:
            line += f"[i for i in range({rng.randint(1, 20)})]"
        out.append(line + "\n")
    return "".join(out)[:n]


def make_math(rng: random.Random, n: int) -> str:
    out = []
    while sum(len(x) for x in out) < n:
        a, b = rng.randint(1, 999), rng.randint(1, 99)
        op = rng.choice(["+", "-", "*", "/"])
        expr = f"{a} {op} {b} = {a + b if op == '+' else a - b if op == '-' else a * b if op == '*' else round(a / b, 3)}"
        out.append(expr)
    return "\n".join(out)[:n]


def build_corpus(seed: int, size: int = 200_000) -> dict[str, str]:
    rng = random.Random(seed)
    return {
        "prose": make_prose(rng, size),
        "code": make_code(rng, size),
        "math": make_math(rng, size),
    }


def encode(text: str) -> torch.Tensor:
    return torch.frombuffer(bytearray(text.encode("utf-8", "ignore")), dtype=torch.uint8).long()


class BatchSource:
    def __init__(self, corpus: dict[str, str], seq: int, seed: int):
        self.seq = seq
        self.data = {k: encode(v) for k, v in corpus.items()}
        self.rng = random.Random(seed)
        self.domains = list(self.data)

    def batch(self, batch_size: int, device: str, fixed: bool = False) -> dict:
        xs, ys, doms = [], [], []
        for i in range(batch_size):
            domain = self.domains[i % len(self.domains)] if fixed else self.rng.choice(self.domains)
            stream = self.data[domain]
            if fixed:
                start = (i // len(self.domains)) * self.seq % (len(stream) - self.seq - 1)
            else:
                start = self.rng.randint(0, len(stream) - self.seq - 2)
            chunk = stream[start : start + self.seq + 1]
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
            doms.append(domain)
        return {
            "input_ids": torch.stack(xs).to(device),
            "labels": torch.stack(ys).to(device),
            "domains": doms,
        }


def build_model(vocab: int, hidden: int, inter: int, layers: int, heads: int, seq: int):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        vocab_size=vocab,
        max_position_embeddings=seq + 8,
        tie_word_embeddings=True,
    )
    return LlamaForCausalLM(config)


@torch.no_grad()
def evaluate(model, source: BatchSource, batch_size: int, iters: int) -> dict:
    model.eval()
    per_domain: dict[str, list[float]] = {d: [] for d in source.domains}
    for _ in range(iters):
        batch = source.batch(batch_size, DEVICE, fixed=True)
        logits = model(input_ids=batch["input_ids"]).logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), batch["labels"].reshape(-1), reduction="none"
        ).reshape(batch["labels"].shape)
        for i, domain in enumerate(batch["domains"]):
            per_domain[domain].append(float(loss[i].mean()))
    model.train()
    result = {d: sum(v) / max(len(v), 1) for d, v in per_domain.items()}
    result["overall"] = sum(result.values()) / len(result)
    return result


def train(model, source: BatchSource, steps: int, lr: float, batch_size: int, *, tag: str = ""):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=max(steps, 1), pct_start=0.1
    )
    model.train()
    losses = []
    for step in range(steps):
        batch = source.batch(batch_size, DEVICE)
        logits = model(input_ids=batch["input_ids"]).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch["labels"].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss))
        for module in model.modules():
            if getattr(module, "is_orrg_moe", False):
                module.apply_bias_update()
        if tag and (step + 1) % max(steps // 4, 1) == 0:
            print(f"    {tag} step {step + 1}/{steps} loss={sum(losses[-50:]) / len(losses[-50:]):.4f}", flush=True)
    return sum(losses[-50:]) / len(losses[-50:])


def run_seed(args, seed: int) -> list[dict]:
    torch.manual_seed(seed)
    random.seed(seed)
    corpus = build_corpus(seed + 1)
    source = BatchSource(corpus, args.seq, seed + 2)
    val_source = BatchSource(corpus, args.seq, seed + 3)

    dense = build_model(256, args.hidden, args.inter, args.layers, args.heads, args.seq).to(DEVICE)
    print(f"  [seed {seed}] training dense seed ({sum(p.numel() for p in dense.parameters())/1e6:.2f}M params)", flush=True)
    train(dense, source, args.seed_steps, args.lr, args.batch)

    probe = {
        "input_ids": val_source.batch(4, DEVICE, fixed=True)["input_ids"],
    }
    seed_val = evaluate(dense, val_source, args.batch, 4)
    rows = []

    arms = {
        "dense_continued": ("dense", MoEConfig(), None),
        "naive_upcycle": (
            "moe",
            MoEConfig(shared=1, groups=4, variants=3, top_residuals=1, residual_init="copy"),
            "non_moe",
        ),
        "orrg_growth": (
            "moe",
            MoEConfig(shared=1, groups=4, variants=3, top_residuals=1, residual_init="zero"),
            "skeleton",
        ),
        "orrg_growth_all": (
            "moe",
            MoEConfig(shared=1, groups=4, variants=3, top_residuals=1, residual_init="zero"),
            None,
        ),
    }

    for name, (kind, cfg, freeze) in arms.items():
        model = copy.deepcopy(dense)
        conversion = None
        if kind == "moe":
            with torch.no_grad():
                dense_logits = model(**probe).logits.clone()
            model, report = convert_model(model, cfg, verify=True, probe=probe)
            with torch.no_grad():
                moe_logits = model(**probe).logits
            conversion = conversion_error(dense_logits, moe_logits)
            model = model.to(DEVICE)
            if freeze == "skeleton":
                from orrg import freeze_skeleton

                freeze_skeleton(model)
            elif freeze == "non_moe":
                from orrg import freeze_non_moe

                freeze_non_moe(model)
        train(model, source, args.steps, args.lr, args.batch, tag=name)
        val = evaluate(model, val_source, args.batch, 4)
        metrics = collect_metrics(model) if kind == "moe" else {}
        rows.append(
            {
                "seed": seed,
                "arm": name,
                "seed_val": seed_val,
                "val": val,
                "conversion": conversion,
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "total_params": sum(p.numel() for p in model.parameters()),
                "metrics": metrics,
                "active_params": report.active_params() if kind == "moe" else sum(p.numel() for p in model.parameters()),
            }
        )
        print(
            f"    {name:16s} val={val['overall']:.4f} "
            f"(prose {val['prose']:.4f} code {val['code']:.4f} math {val['math']:.4f})"
            + (f" conv_kl={conversion['kl']:.2e}" if conversion else ""),
            flush=True,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed-steps", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--inter", type=int, default=384)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq", type=int, default=96)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", default=os.path.join("experiments", "results", "toy_scale.json"))
    args = parser.parse_args()

    start = time.time()
    rows: list[dict] = []
    for seed in range(args.seeds):
        rows.extend(run_seed(args, seed))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "rows": rows}, handle, indent=2)

    arms = sorted({r["arm"] for r in rows})
    print("\n| arm | val loss | prose | code | math | conv KL | trainable | imbalance | dead | diversity |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for arm in arms:
        subset = [r for r in rows if r["arm"] == arm]
        avg = lambda key: sum(r["val"][key] for r in subset) / len(subset)
        conv = [r["conversion"]["kl"] for r in subset if r["conversion"]]
        metrics = [r["metrics"] for r in subset if r["metrics"]]
        print(
            f"| {arm} | {avg('overall'):.4f} | {avg('prose'):.4f} | {avg('code'):.4f} | {avg('math'):.4f} | "
            f"{(sum(conv)/len(conv) if conv else float('nan')):.2e} | "
            f"{sum(r['trainable_params'] for r in subset)/len(subset)/1e6:.2f}M | "
            f"{(sum(m['load_imbalance'] for m in metrics)/len(metrics) if metrics else float('nan')):.4f} | "
            f"{(sum(m['dead_experts'] for m in metrics)/len(metrics) if metrics else float('nan')):.1f} | "
            f"{(sum(m['expert_diversity'] for m in metrics)/len(metrics) if metrics else float('nan')):.4f} |"
        )
    print(f"\nwrote {args.out} in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
