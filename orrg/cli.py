"""Command line entry point: python -m orrg ..."""

from __future__ import annotations

import argparse
import json
import sys

import torch


def _shape_from_args(args) -> "object":
    from .scale import ModelShape

    return ModelShape(
        hidden=args.hidden,
        intermediate=args.inter,
        layers=args.layers,
        vocab=args.vocab,
        n_heads=args.heads,
        n_kv_heads=args.kv_heads,
        gated=not args.ungated,
        tie_embeddings=not args.untied,
    )


def cmd_scale(args) -> int:
    from .scale import converted_scale, estimate_training, format_scale

    shape = _shape_from_args(args)
    scale = converted_scale(shape, args.shared, args.groups, args.variants)
    print(format_scale(scale))
    if args.tokens:
        for k, row in scale["converted"].items():
            budget = estimate_training(row["active_params"], args.tokens, mfu=args.mfu, gpus=args.gpus)
            print(
                f"  k={k}: {budget['gpu_hours']:8.0f} GPU-hours "
                f"({budget['wall_clock_days']:.1f} days on {args.gpus} GPUs at {args.mfu:.0%} MFU)"
            )
    if args.json:
        print(json.dumps(scale, indent=2))
    return 0


def cmd_inspect(args) -> int:
    from transformers import AutoConfig, AutoModelForCausalLM

    from .ffn import discover_ffn_modules

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    config = AutoConfig.from_pretrained(args.model)
    hidden = getattr(config, "hidden_size", getattr(config, "n_embd", None))
    if args.random:
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    specs = discover_ffn_modules(
        model,
        hidden_size=hidden or args.hidden,
        include=tuple(args.include or ()),
        exclude=tuple(args.exclude or ()),
    )
    dense = sum(p.numel() for p in model.parameters())
    print(f"{len(specs)} FFN blocks, hidden={specs[0].hidden if specs else None}, "
          f"intermediate={specs[0].intermediate if specs else None}, dense={dense/1e9:.3f}B")
    for spec in specs[: args.limit]:
        print(f"  {spec.name:60s} {spec.hidden:5d} -> {spec.intermediate:6d} "
              f"({spec.dense_params/1e6:7.2f}M params, {len(spec.up)} up / {len(spec.down)} down)")
    if len(specs) > args.limit:
        print(f"  ... {len(specs) - args.limit} more")
    return 0


def cmd_convert(args) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .config import MoEConfig
    from .convert import convert_model, save_converted

    cfg = MoEConfig(
        shared=args.shared,
        groups=args.groups,
        variants=args.variants,
        top_residuals=args.top_residuals,
        mode=args.mode,
        top_k_partition=args.top_k,
        residual_init=args.residual_init,
        include=tuple(args.include or ()),
        exclude=tuple(args.exclude or ()),
    )
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    probe = None
    if args.verify:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        batch = tokenizer("The quick brown fox jumps over", return_tensors="pt")
        probe = dict(batch)
    model, report = convert_model(model, cfg, verify=args.verify, probe=probe)
    print(report.summary())
    if args.out:
        save_converted(model, cfg, args.out)
        print(f"saved -> {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orrg", description="Lossless dense -> MoE conversion")
    sub = parser.add_subparsers(dest="command", required=True)

    scale = sub.add_parser("scale", help="parameter / FLOP / memory planning")
    scale.add_argument("--hidden", type=int, required=True)
    scale.add_argument("--inter", type=int, required=True)
    scale.add_argument("--layers", type=int, required=True)
    scale.add_argument("--vocab", type=int, default=0)
    scale.add_argument("--heads", type=int, default=0)
    scale.add_argument("--kv-heads", type=int, default=0)
    scale.add_argument("--shared", type=int, default=2)
    scale.add_argument("--groups", type=int, default=6)
    scale.add_argument("--variants", type=int, default=8)
    scale.add_argument("--ungated", action="store_true")
    scale.add_argument("--untied", action="store_true")
    scale.add_argument("--tokens", type=float, default=0)
    scale.add_argument("--mfu", type=float, default=0.25)
    scale.add_argument("--gpus", type=int, default=8)
    scale.add_argument("--json", action="store_true")
    scale.set_defaults(func=cmd_scale)

    inspect = sub.add_parser("inspect", help="list discovered FFN blocks")
    inspect.add_argument("--model", required=True)
    inspect.add_argument("--hidden", type=int, default=None)
    inspect.add_argument("--bf16", action="store_true")
    inspect.add_argument("--random", action="store_true", help="use the config only, no weights")
    inspect.add_argument("--include", nargs="*", default=None)
    inspect.add_argument("--exclude", nargs="*", default=None)
    inspect.add_argument("--limit", type=int, default=8)
    inspect.set_defaults(func=cmd_inspect)

    conv = sub.add_parser("convert", help="convert a HF checkpoint into an ORRG MoE")
    conv.add_argument("--model", required=True)
    conv.add_argument("--out", default=None)
    conv.add_argument("--shared", type=int, default=2)
    conv.add_argument("--groups", type=int, default=6)
    conv.add_argument("--variants", type=int, default=8)
    conv.add_argument("--top-residuals", type=int, default=1)
    conv.add_argument("--mode", choices=["growth", "partition"], default="growth")
    conv.add_argument("--top-k", type=int, default=1)
    conv.add_argument("--residual-init", choices=["zero", "copy"], default="zero")
    conv.add_argument("--include", nargs="*", default=None)
    conv.add_argument("--exclude", nargs="*", default=None)
    conv.add_argument("--verify", action="store_true")
    conv.set_defaults(func=cmd_convert)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
