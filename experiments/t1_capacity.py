"""T1 capacity-binding precondition for ORRG-MoE (spec/ORRG-SPEC-v4.md, section 5).

Question the rung answers: on this corpus and this token budget, does *added dense
capacity* buy validation loss? If a 1.5x wider dense model cannot beat the seed on
identical batches, then a MoE arm with extra capacity has nothing to spend it on,
and any MoE-vs-dense comparison at that budget is uninformative.

Design (exactly the spec's T1, scaled by flags):

    arms      dense seed vs the same model with FFN intermediate x1.5
    data      identical batch offsets per (seed, step) for both arms
    budget    identical steps, identical schedule
    seeds     --seeds, paired by seed
    eval      fixed held-out windows, evaluated at 25/50/100% of the run
    report    paired end-val delta with per-seed values, SD, 2-sigma and a t-CI

Nothing is converted to MoE here; this rung is deliberately conversion-free.

    PYTHONPATH=. python experiments/t1_capacity.py --seeds 3 --steps 2000

Artifacts: experiments/results/t1_capacity.json (raw) and a printed table.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WIKITEXT_GLOB = "datasets--Salesforce--wikitext/snapshots/*/wikitext-103-raw-v1/*.parquet"

# Two-sided 95% t quantile for df = 2; used only when seeds == 3.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def read_bytes(max_bytes: int) -> torch.Tensor:
    """WikiText-103 raw as a byte stream, taken from the local HF cache."""

    import pyarrow.parquet as pq

    root = Path.home() / ".cache" / "huggingface" / "hub"
    files = sorted(glob.glob(str(root / WIKITEXT_GLOB)))
    if not files:
        raise FileNotFoundError(f"no cached wikitext parquet under {root}")

    parts: list[bytes] = []
    total = 0
    for path in files:
        table = pq.read_table(path, columns=["text"])
        for value in table.column("text").to_pylist():
            if not value:
                continue
            blob = value.encode("utf-8")
            parts.append(blob)
            total += len(blob)
            if total >= max_bytes:
                break
        if total >= max_bytes:
            break
    joined = b"".join(parts)[:max_bytes]
    if len(joined) < max_bytes // 2:
        raise RuntimeError(f"only {len(joined)} bytes available from the cache")
    # Kept as uint8 so the whole stream fits on the card; cast per batch.
    return torch.frombuffer(bytearray(joined), dtype=torch.uint8)


def build_model(hidden: int, inter: int, layers: int, heads: int, kv_heads: int,
                vocab: int = 256):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        tie_word_embeddings=True,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        hidden_act="silu",
    )
    return LlamaForCausalLM(cfg)


class Stream:
    """Deterministic window sampler over one byte stream.

    Offsets are generated from (split, seed, step) only, so both arms and every
    arm of a seed can be handed byte-identical batches.
    """

    def __init__(self, tokens: torch.Tensor, seq: int, batch: int, val_fraction: float = 0.02):
        self.seq = seq
        self.bs = batch
        self.split = int(len(tokens) * (1 - val_fraction))
        self.train_tokens = tokens[: self.split].to(DEVICE)
        self.val_tokens = tokens[self.split :].to(DEVICE)

    def offsets(self, seed: int, step: int) -> list[int]:
        rng = random.Random((seed * 1_000_003) ^ (step * 7_919) ^ 0x5EED)
        n = len(self.train_tokens) - self.seq - 2
        return [rng.randint(0, n) for _ in range(self.bs)]

    def make_batch(self, offsets: list[int], *, train: bool = True):
        tokens = self.train_tokens if train else self.val_tokens
        idx = torch.tensor(offsets, device=DEVICE).unsqueeze(1) + torch.arange(self.seq, device=DEVICE)
        return {
            "input_ids": tokens[idx].long(),
            "labels": tokens[idx + 1].long(),
        }

    def val_batches(self, count: int = 12) -> list[dict]:
        """Fixed, evenly spaced held-out windows; identical for every arm."""

        n = len(self.val_tokens) - self.seq - 2
        stride = max(1, n // (count * self.bs))
        batches = []
        for b in range(count):
            starts = [min(n, (b * self.bs + j) * stride) for j in range(self.bs)]
            batches.append(self.make_batch(starts, train=False))
        return batches


@torch.no_grad()
def evaluate(model, batches: list[dict]) -> float:
    model.eval()
    total = 0.0
    for batch in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            total += float(model(**batch).loss)
    model.train()
    return total / len(batches)


def run_arm(args, inter: int, seed: int, checkpoints: set[int], val: list[dict]) -> dict:
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = build_model(args.hidden, inter, args.layers, args.heads, args.kv_heads).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                                  betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.steps, pct_start=0.03
    )
    # Offsets depend only on (seed, step), so both arms of a seed see byte-identical
    # batches. The list is built once here and shared across arms by construction.
    rng = random.Random((seed * 1_000_003) ^ 0x5EED)
    n_train = args.offsets_n
    offsets_by_step = [
        [rng.randint(0, n_train) for _ in range(args.batch)] for _ in range(args.steps)
    ]

    rows = []
    started = time.time()
    model.train()
    train_losses: list[float] = []
    for step in range(args.steps):
        batch = args.stream.make_batch(offsets_by_step[step])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            loss = model(**batch).loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        train_losses.append(float(loss.detach()))
        if (step + 1) in checkpoints:
            val_loss = evaluate(model, val)
            rows.append({
                "step": step + 1,
                "fraction": (step + 1) / args.steps,
                "val": val_loss,
                "train": sum(train_losses[-50:]) / len(train_losses[-50:]),
                "tokens": (step + 1) * args.batch * args.seq,
            })
            print(f"      step {step + 1:5d} val={val_loss:.4f} "
                  f"train={rows[-1]['train']:.4f}", flush=True)

    elapsed = time.time() - started
    tokens = args.steps * args.batch * args.seq
    return {
        "seed": seed,
        "inter": inter,
        "params": params,
        "steps": args.steps,
        "tokens": tokens,
        "tokens_per_param": tokens / params,
        "seconds": elapsed,
        "seconds_per_step": elapsed / args.steps,
        "tokens_per_second": tokens / elapsed,
        "trace": rows,
        "final_val": rows[-1]["val"],
        "final_train": rows[-1]["train"],
        "train_val_gap": rows[-1]["train"] - rows[-1]["val"],
    }


def paired_stats(rows: list[dict], wide: str, dense: str) -> dict:
    by_seed = {}
    for row in rows:
        by_seed.setdefault(row["seed"], {})[row["arm"]] = row
    pairs = [(s, d[wide], d[dense]) for s, d in sorted(by_seed.items()) if wide in d and dense in d]
    diffs = [w["final_val"] - d["final_val"] for _, w, d in pairs]
    n = len(diffs)
    mean = statistics.fmean(diffs) if n else float("nan")
    sd = statistics.stdev(diffs) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    t = T975.get(n - 1)
    return {
        "n": n,
        "per_seed": {s: {"wide": w["final_val"], "dense": d["final_val"], "delta": w["final_val"] - d["final_val"]}
                     for s, w, d in pairs},
        "deltas": diffs,
        "mean": mean,
        "sd": sd,
        "two_sigma": 2 * sd if n > 1 else float("nan"),
        "ci95": (mean - t * se, mean + t * se) if t and n > 1 else None,
        "gate_threshold": max(2 * sd, 0.005) if n > 1 else 0.005,
        "gate_pass": bool(n > 1 and mean < -max(2 * sd, 0.005)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--inter", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--max-bytes", type=int, default=140_000_000)
    parser.add_argument("--out", default="experiments/results/t1_capacity.json")
    args = parser.parse_args()

    started = time.time()
    tokens = read_bytes(args.max_bytes)
    args.stream = Stream(tokens, args.seq, args.batch)
    args.offsets_n = len(args.stream.train_tokens) - args.seq - 2
    val = args.stream.val_batches()

    total_tokens = args.steps * args.batch * args.seq
    print(f"corpus: {len(tokens)/1e6:.1f}M bytes ({len(tokens)/1e6:.1f}M tokens at 1 byte/token); "
          f"train {len(args.stream.train_tokens)/1e6:.1f}M, held-out "
          f"{len(args.stream.val_tokens)/1e6:.1f}M", flush=True)
    print(f"budget: {total_tokens/1e6:.1f}M tokens = {args.steps} steps x {args.batch} x {args.seq}; "
          f"device={DEVICE}", flush=True)

    checkpoints = {max(1, args.steps // 4), max(1, args.steps // 2), args.steps}
    rows = []
    for seed in range(args.seeds):
        for name, inter in (("dense", args.inter), ("wide", int(args.inter * 1.5))):
            print(f"\n[seed {seed}] {name} (inter={inter})", flush=True)
            row = run_arm(args, inter, seed, checkpoints, val)
            row["arm"] = name
            rows.append(row)
            print(f"    params {row['params']/1e6:.2f}M, {row['tokens_per_param']:.1f} tok/param, "
                  f"{row['seconds_per_step']:.3f}s/step, {row['tokens_per_second']/1e3:.1f}k tok/s", flush=True)

    stats = paired_stats(rows, "wide", "dense")
    payload = {
        "args": {k: v for k, v in vars(args).items() if k not in ("stream",)},
        "corpus": {"bytes": len(tokens), "train": len(args.stream.train_tokens),
                   "val": len(args.stream.val_tokens)},
        "rows": rows,
        "paired": stats,
        "seconds": time.time() - started,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("\n| arm | seed | params | tok/param | val @25% | val @50% | val final | train (last50) | gap |")
    print("|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        t = {round(x["fraction"] * 4): x["val"] for x in row["trace"]}
        print(f"| {row['arm']} | {row['seed']} | {row['params']/1e6:.2f}M | "
              f"{row['tokens_per_param']:.1f} | {t.get(1, float('nan')):.4f} | "
              f"{t.get(2, float('nan')):.4f} | {row['final_val']:.4f} | "
              f"{row['final_train']:.4f} | {row['train_val_gap']:+.4f} |")

    print(f"\npaired end-val delta (wide - dense), negative = extra capacity pays")
    print(f"  n={stats['n']}  per-seed={[round(x, 4) for x in stats['deltas']]}")
    print(f"  mean={stats['mean']:+.4f}  sd={stats['sd']:.4f}  2sd={stats['two_sigma']:.4f}")
    if stats["ci95"]:
        print(f"  95% CI (t, df={stats['n']-1}) = [{stats['ci95'][0]:+.4f}, {stats['ci95'][1]:+.4f}]")
    print(f"  gate: mean < -max(2sd, 0.005) = -{stats['gate_threshold']:.4f} -> "
          f"{'PASS' if stats['gate_pass'] else 'FAIL'}")
    print(f"\nwrote {args.out} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
