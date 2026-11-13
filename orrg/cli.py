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
    scale = converted_scale(
        shape,
        args.shared,
        args.groups,
        args.variants,
        skeleton_scale=args.skeleton_scale,
        expert_width_ratio=args.expert_ratio,
        expert_mode=args.expert_mode,
    )
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


def moe_config_from_args(args) -> "object":
    """Turn parsed ``convert`` arguments into a ``MoEConfig``.

    ``--top-k`` and ``--top-residuals`` are two spellings of "experts routed per
    group per token". The accounting (``top_residuals``) and the router
    (``RouterConfig.group_top_k``) have to agree: the reference forward takes its
    slot count from the router, so setting only ``top_residuals`` reports k
    experts and routes one.
    """

    from .config import MoEConfig

    if args.mode == "partition" and args.top_residuals is not None:
        # Partition mode routes over disjoint skeleton shards and disables the
        # residual variants, so a flag named --top-residuals cannot mean anything
        # here; letting it pick the shard count was the confusing part.
        raise SystemExit(
            "partition mode has no residual experts; use --top-k for the shard count"
        )
    if (
        args.top_k is not None
        and args.top_residuals is not None
        and args.top_k != args.top_residuals
    ):
        raise SystemExit("--top-k and --top-residuals disagree; pass only one of them")
    routed = args.top_k if args.top_k is not None else args.top_residuals
    routed = 1 if routed is None else routed
    if routed < 0:
        raise SystemExit("--top-k / --top-residuals must be >= 0")
    # ``MoEConfig`` makes the router follow ``top_residuals``, so the router is not
    # set here; partition mode reports no residual experts and routes shards via
    # ``top_k_partition`` instead.
    return MoEConfig(
        shared=args.shared,
        groups=args.groups,
        variants=args.variants,
        top_residuals=0 if args.mode == "partition" else routed,
        mode=args.mode,
        # Only partition mode reads top_k_partition; setting it in growth mode
        # trips unrelated validations (e.g. dispatch='fused' rejects it).
        top_k_partition=max(1, routed) if args.mode == "partition" else 1,
        residual_init=args.residual_init,
        expert_mode=args.expert_mode,
        expert_width_ratio=args.expert_ratio,
        skeleton_scale=args.skeleton_scale,
        router_warm_start=args.router_warm_start,
        dispatch=args.dispatch,
        include=tuple(args.include or ()),
        exclude=tuple(args.exclude or ()),
    )


def cmd_convert(args) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .convert import convert_model, save_converted

    cfg = moe_config_from_args(args)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    probe = None
    if args.verify:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        batch = tokenizer("The quick brown fox jumps over", return_tensors="pt")
        probe = dict(batch)
    calibration = None
    if cfg.router_warm_start == "cluster":
        if probe is None:
            raise SystemExit(
                "--router-warm-start cluster needs calibration data; pass --verify"
            )
        calibration = [probe]
    model, report = convert_model(
        model, cfg, verify=args.verify, probe=probe, calibration=calibration
    )
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
    scale.add_argument("--skeleton-scale", type=float, default=1.0)
    scale.add_argument("--expert-ratio", type=float, default=1.0)
    scale.add_argument("--expert-mode", choices=["additive", "replace"], default="additive")
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
    conv.add_argument(
        "--top-residuals",
        type=int,
        default=None,
        help="experts routed per group per token (alias of --top-k)",
    )
    conv.add_argument("--mode", choices=["growth", "partition"], default="growth")
    conv.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="experts routed per group per token: residual experts in growth mode, "
        "shards in partition mode",
    )
    conv.add_argument("--residual-init", choices=["zero", "copy"], default="zero")
    conv.add_argument("--expert-mode", choices=["additive", "replace"], default="additive")
    conv.add_argument("--expert-ratio", type=float, default=1.0)
    conv.add_argument("--skeleton-scale", type=float, default=1.0)
    conv.add_argument("--router-warm-start", choices=["none", "cluster"], default="none")
    conv.add_argument(
        "--dispatch",
        choices=["python", "fused"],
        default="python",
        help="fused = grouped-GEMM inference path (needs torch.no_grad)",
    )
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
