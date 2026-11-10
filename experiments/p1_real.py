"""P1 rung on a real dense checkpoint (SmolLM2-360M) and real text (WikiText-103).

Arms (same batches, same step budget):
    dense_continued    the seed, still training
    naive_upcycle      residual experts are full copies (not lossless)
    orrg_frozen        lossless start, skeleton frozen, only residuals+routers trained
    orrg_all           lossless start, everything trained
    orrg_clustered     orrg_all + activation-clustered shards (calibration pass)
    orrg_distill       orrg_all + contribution-target router distillation

Everything runs offline from the local HF cache.

    PYTHONPATH=. python experiments/p1_real.py --steps 200 --seeds 2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from orrg import (
    MoEConfig,
    convert_model,
    freeze_non_moe,
    freeze_skeleton,
)
from orrg.metrics import collect_metrics, conversion_error, moe_layers, residual_share
from orrg.train import TrainConfig, evaluate_loss, orthogonality_loss, train

MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
WIKITEXT_GLOB = "datasets--Salesforce--wikitext/snapshots/*/wikitext-103-raw-v1/*.parquet"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, local_files_only=True, dtype=torch.float32
    )
    return model


def load_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_tokens(tokenizer, max_chars: int = 6_000_000) -> torch.Tensor:
    import pyarrow.parquet as pq

    root = Path.home() / ".cache" / "huggingface" / "hub"
    files = sorted(glob.glob(str(root / WIKITEXT_GLOB)))
    if not files:
        raise FileNotFoundError(f"no cached wikitext parquet under {root}")

    chunks: list[str] = []
    total = 0
    for path in files:
        table = pq.read_table(path, columns=["text"])
        for value in table.column("text").to_pylist():
            if not value:
                continue
            chunks.append(value)
            total += len(value)
            if total >= max_chars:
                break
        if total >= max_chars:
            break
    blob = "\n".join(chunks)[:max_chars]
    ids = tokenizer(blob, add_special_tokens=False)["input_ids"]
    return torch.tensor(ids, dtype=torch.long)


class WindowBatches:
    """Random training windows or fixed validation windows over one token stream."""

    def __init__(
        self,
        tokens: torch.Tensor,
        *,
        seq: int,
        batch: int,
        seed: int,
        val: bool = False,
        val_fraction: float = 0.05,
    ):
        self.seq = seq
        self.batch = batch
        self.rng = random.Random(seed)
        self.val = val
        split = int(len(tokens) * (1 - val_fraction))
        self.tokens = tokens[split:] if val else tokens[:split]
        self._val_batches = None

    def _make(self, starts: list[int]) -> dict:
        xs = torch.stack([self.tokens[s : s + self.seq] for s in starts])
        ys = torch.stack([self.tokens[s + 1 : s + self.seq + 1] for s in starts])
        return {"input_ids": xs.to(DEVICE), "labels": ys.to(DEVICE)}

    def __iter__(self):
        if self.val:
            if self._val_batches is None:
                n = len(self.tokens) - self.seq - 1
                stride = max(1, n // (8 * self.batch))
                batches = []
                for bi in range(8):
                    starts = [min(n, (bi * self.batch + j) * stride) for j in range(self.batch)]
                    batches.append(self._make(starts))
                self._val_batches = batches
            while True:
                yield from self._val_batches
        while True:
            n = len(self.tokens) - self.seq - 1
            starts = [self.rng.randint(0, n) for _ in range(self.batch)]
            yield self._make(starts)


def moe_base(args, **overrides) -> MoEConfig:
    values = dict(shared=args.shared, groups=args.groups, variants=args.variants, top_residuals=1)
    values.update(overrides)
    return MoEConfig(**values)


def build_arms(args):
    return [
        ("dense_continued", None, None, {}),
        ("naive_upcycle", moe_base(args, residual_init="copy"), "non_moe", {}),
        ("orrg_frozen", moe_base(args), "skeleton", {}),
        ("orrg_all", moe_base(args), None, {}),
        (
            "orrg_mag",
            moe_base(
                args,
                residual_init="scaled",
                down_init_scale=args.down_init_scale,
            ),
            None,
            {},
        ),
        (
            "orrg_orth_mag",
            moe_base(
                args,
                residual_init="scaled",
                down_init_scale=args.down_init_scale,
                residual_noise=args.residual_noise,
            ),
            None,
            {"orth_step": args.orth_step, "orth_margin": args.orth_margin},
        ),
        (
            "orrg_clustered",
            moe_base(args, cluster_shards=True, cluster_tokens=args.cluster_tokens, cluster_iters=8),
            None,
            {},
        ),
        (
            "orrg_distill",
            moe_base(
                args,
                router_distill_coef=args.distill_coef,
                distill_tokens=args.distill_tokens,
            ),
            None,
            {},
        ),
        (
            "orrg_orth",
            moe_base(args, residual_noise=args.residual_noise),
            None,
            {"orth_step": args.orth_step, "orth_margin": args.orth_margin},
        ),
        (
            "orrg_sharp",
            moe_base(
                args,
                router_distill_coef=args.distill_coef,
                distill_tokens=args.distill_tokens,
                distill_target_temp=args.distill_target_temp,
            ),
            None,
            {},
        ),
    ]


def run_arm(args, name, cfg, freeze, tokens, seed, val_batches, train_extra=None) -> dict:
    torch.manual_seed(seed * 1000 + sum(ord(c) for c in name) % 997)
    model = load_model()
    conversion = None
    report = None

    if cfg is not None:
        probe_batches = [val_batches._make([0])]
        probe = {"input_ids": probe_batches[0]["input_ids"].cpu()}
        calibration = None
        if cfg.cluster_shards:
            calibration_iter = iter(val_batches)
            calibration = [next(calibration_iter) for _ in range(2)]
            calibration = [{"input_ids": b["input_ids"].cpu()} for b in calibration]
        with torch.no_grad():
            reference = model(**probe).logits.clone()
        model, report = convert_model(model, cfg, probe=probe, calibration=calibration)
        with torch.no_grad():
            converted = model(**probe).logits
        conversion = conversion_error(reference, converted)
        if freeze == "skeleton":
            freeze_skeleton(model)
        elif freeze == "non_moe":
            freeze_non_moe(model)

    model = model.to(DEVICE)
    before = evaluate_loss(model, val_batches, iters=8, device=DEVICE)

    trace_rows: list[dict] = []

    def record(step: int) -> None:
        # Read routing state from the training forward before the eval pass, which
        # resets last_stats and clears the distillation term in eval mode.
        current = collect_metrics(model) if cfg is not None else {}
        distill_values = [
            float(layer.last_distill_loss.detach())
            for layer in moe_layers(model)
            if layer.last_distill_loss is not None
        ]
        share = residual_share(model, val_batches._make([0])) if cfg is not None else None
        orth = None
        if cfg is not None:
            with torch.no_grad():
                term = orthogonality_loss(model, (train_extra or {}).get("orth_margin", 0.5))
            orth = float(term) if term is not None else None
        val = evaluate_loss(model, val_batches, iters=8, device=DEVICE)
        entry = {
            "step": step,
            "val": val,
            "imbalance": current.get("load_imbalance"),
            "dead": current.get("dead_experts"),
            "diversity": current.get("expert_diversity"),
            "entropy": current.get("router_entropy"),
            "residual_share": share,
            "orth": orth,
            "distill": sum(distill_values) / len(distill_values) if distill_values else None,
        }
        trace_rows.append(entry)
        extra = "".join(
            f" {key}={entry[key]:.4f}"
            for key in ("imbalance", "diversity", "residual_share", "orth", "distill")
            if entry[key] is not None
        )
        print(f"    trace {step} val={val:.4f}{extra}", flush=True)

    ckpt = str(Path("experiments/results") / f"ckpt_{name}.pt")
    train_cfg = TrainConfig(
        lr=args.lr,
        router_distill_coef=cfg.router_distill_coef if cfg is not None else 0.0,
        bias_update_rate=args.bias_rate,
        bf16=True,
        optimizer=args.optimizer,
        log_every=args.log_every,
        trace_every=args.trace_every,
        checkpoint_path=ckpt,
        checkpoint_every=args.ckpt_every,
        resume=args.resume,
        **(train_extra or {}),
    )
    # On resume the checkpoint carries the real step count, so a fresh-model
    # trace at step 0 would be misleading.
    if args.trace_every and not (args.resume and Path(ckpt).exists()):
        record(0)
    stats = train(
        model,
        iter(WindowBatches(tokens, seq=args.seq, batch=args.batch, seed=seed * 7 + 1)),
        args.steps,
        train_cfg,
        tag=name,
        log=print if args.log_every else None,
        trace=record if args.trace_every else None,
    )
    after = evaluate_loss(model, val_batches, iters=8, device=DEVICE)
    metrics = collect_metrics(model) if cfg is not None else {}

    row = {
        "seed": seed,
        "arm": name,
        "val_before": before,
        "val_after": after,
        "delta": after - before,
        "conversion": conversion,
        "trainable_params": stats["trainable_params"],
        "total_params": sum(p.numel() for p in model.parameters()),
        "active_params": report.active_params() if report else None,
        "seconds_per_step": stats["seconds_per_step"],
        "metrics": metrics,
        "trace": trace_rows,
        "clustering": (
            {
                "cohesion": sum(c.cohesion for c in report.clustering) / len(report.clustering),
                "baseline": sum(c.baseline for c in report.clustering) / len(report.clustering),
                "balance": sum(c.balance for c in report.clustering) / len(report.clustering),
            }
            if report is not None and report.clustering
            else None
        ),
    }
    # A finished arm must not be resumed into a second pass.
    if os.path.exists(ckpt):
        os.remove(ckpt)
    del model
    torch.cuda.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--bias-rate", type=float, default=0.0)
    parser.add_argument("--distill-coef", type=float, default=0.1)
    parser.add_argument("--distill-tokens", type=int, default=128)
    parser.add_argument("--optimizer", default="adafactor", choices=["adamw", "adafactor"])
    parser.add_argument("--cluster-tokens", type=int, default=512)
    parser.add_argument("--shared", type=int, default=2)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--trace-every", type=int, default=0)
    parser.add_argument("--orth-step", type=float, default=1e-5)
    parser.add_argument("--orth-margin", type=float, default=0.5)
    parser.add_argument("--distill-target-temp", type=float, default=0.25)
    parser.add_argument("--residual-noise", type=float, default=0.02)
    parser.add_argument("--down-init-scale", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt-every", type=int, default=250)
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--max-chars", type=int, default=6_000_000)
    parser.add_argument("--out", default="experiments/results/p1_real.json")
    args = parser.parse_args()

    tokenizer = load_tokenizer()
    started = time.time()
    tokens = load_tokens(tokenizer, args.max_chars)
    print(f"corpus: {tokens.numel()/1e6:.2f}M tokens; device={DEVICE}", flush=True)

    arms = build_arms(args)
    if args.arms:
        arms = [arm for arm in arms if arm[0] in set(args.arms)]

    rows = []
    for seed in range(args.seeds):
        val_batches = WindowBatches(tokens, seq=args.seq, batch=4, seed=seed, val=True)
        for name, cfg, freeze, train_extra in arms:
            print(f"\n[seed {seed}] {name}", flush=True)
            row = run_arm(args, name, cfg, freeze, tokens, seed, val_batches, train_extra)
            rows.append(row)
            print(
                f"  val {row['val_before']:.4f} -> {row['val_after']:.4f} "
                f"(delta {row['delta']:+.4f}) "
                + (f"conv_kl={row['conversion']['kl']:.2e} " if row["conversion"] else "")
                + f"trainable={row['trainable_params']/1e6:.1f}M "
                f"({row['seconds_per_step']:.2f}s/step)",
                flush=True,
            )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "rows": rows}, handle, indent=2)

    names = [arm[0] for arm in arms]
    print("\n| arm | val before | val after | delta | conv KL | trainable | s/step | imbalance | dead | diversity |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name in names:
        subset = [r for r in rows if r["arm"] == name]
        mean = lambda key: sum(r[key] for r in subset) / len(subset)
        conv = [r["conversion"]["kl"] for r in subset if r["conversion"]]
        metrics = [r["metrics"] for r in subset if r["metrics"]]
        print(
            f"| {name} | {mean('val_before'):.4f} | {mean('val_after'):.4f} | "
            f"{mean('delta'):+.4f} | {(sum(conv)/len(conv) if conv else float('nan')):.2e} | "
            f"{mean('trainable_params')/1e6:.1f}M | {mean('seconds_per_step'):.2f} | "
            f"{(sum(m['load_imbalance'] for m in metrics)/len(metrics) if metrics else float('nan')):.3f} | "
            f"{(sum(m['dead_experts'] for m in metrics)/len(metrics) if metrics else float('nan')):.1f} | "
            f"{(sum(m['expert_diversity'] for m in metrics)/len(metrics) if metrics else float('nan')):.4f} |"
        )
    print(f"\nwrote {args.out} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
