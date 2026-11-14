"""Tests for the v4 geometry: expert mode, expert width, skeleton scale, warm start.

These are additive to ``tests/test_orrg.py``: the suite as a whole must keep the
default geometry (``skeleton_scale=1``, ``expert_width_ratio=1``, additive
experts) bit-identical to v3, which the first test asserts directly.
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import pytest
import torch

from orrg import MoEConfig, RouterConfig, convert_model
from orrg.scale import ModelShape, converted_scale
from orrg.shard import shard_widths

HIDDEN = 32
INTER = 64
LAYERS = 2
VOCAB = 128


def build_model(arch: str = "llama"):
    from transformers import LlamaConfig, LlamaForCausalLM

    common = dict(
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=VOCAB,
    )
    if arch == "llama":
        return LlamaForCausalLM(LlamaConfig(**common))
    raise ValueError(arch)


def probe_ids(seed: int = 0, batch: int = 2, seq: int = 8) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {"input_ids": torch.randint(0, VOCAB, (batch, seq), generator=generator)}


def growth_cfg(**overrides) -> MoEConfig:
    values = dict(shared=1, groups=2, variants=4, top_residuals=1, hidden_size=HIDDEN)
    values.update(overrides)
    return MoEConfig(**values)


def moe_layers(model):
    return [m for m in model.modules() if getattr(m, "is_orrg_moe", False)]


# ------------------------------------------------------- defaults are unchanged
def test_default_geometry_is_bit_identical_to_v3():
    """f=1, ratio=1, additive must reproduce the v3 layout and the v3 closed form."""

    torch.manual_seed(21)
    converted, report = convert_model(build_model(), growth_cfg(), verify=False)

    assert report.lossless is True
    assert report.conversion_kl is None  # nothing was verified
    for layer in moe_layers(converted):
        assert layer.skeleton_width == INTER
        assert layer.widths == shard_widths(INTER, 3)
        # ratio=1 keeps the historical identity: expert indices are the group's
        assert all(
            torch.equal(a, b) for a, b in zip(layer.expert_indices, layer.index_sets[1:])
        )
        for group in layer.groups:
            base = max(group.base.parameters(), key=lambda p: p.numel())
            for expert in group.residuals:
                weight = max(expert.parameters(), key=lambda p: p.numel())
                assert torch.equal(base, weight), "ratio=1 must be an exact copy"

    shape = ModelShape(
        hidden=4096, intermediate=10240, layers=32, vocab=128000, n_heads=32, n_kv_heads=8
    )
    row = shape.converted(2, 6, 8, 1)
    shard = 10240 / 8
    assert row["total_params"] == int(shape.non_ffn_params + 32 * 3 * 4096 * shard * (2 + 6 * 8))
    assert row["active_params"] == int(shape.non_ffn_params + 32 * 3 * 4096 * shard * (2 + 6 * 2))
    assert row["lossless"] is True


def test_skeleton_scale_one_stays_lossless_and_unlabelled():
    torch.manual_seed(22)
    model = build_model().eval()
    ids = probe_ids(22)
    with torch.no_grad():
        reference = model(**ids).logits

    converted, report = convert_model(model, growth_cfg(), probe=ids)
    converted.eval()
    with torch.no_grad():
        got = converted(**ids).logits

    assert (reference - got).abs().max() < 1e-5
    assert report.lossless is True
    assert report.logit_kl is not None and report.logit_kl < 1e-6
    assert "approximate" not in report.summary()
    assert all(layer.skeleton_width == INTER for layer in moe_layers(converted))


# --------------------------------------------------------------- expert width
def test_expert_width_ratio_narrows_experts_only():
    torch.manual_seed(23)
    model = build_model().eval()
    ids = probe_ids(23)
    with torch.no_grad():
        reference = model(**ids).logits

    cfg = growth_cfg(expert_width_ratio=0.5)
    converted, report = convert_model(model, cfg, probe=ids)
    converted.eval()
    with torch.no_grad():
        got = converted(**ids).logits

    # Zero-init residuals keep the conversion exact even at ratio < 1.
    assert (reference - got).abs().max() < 1e-5
    layer = moe_layers(converted)[0]
    for gi, group in enumerate(layer.groups):
        shard = layer.widths[cfg.shared + gi]
        expected = max(1, round(shard * 0.5))
        assert layer.expert_widths[gi] == expected
        weight = max(group.residuals[0].parameters(), key=lambda p: p.numel())
        assert weight.shape[0] == expected
        base = max(group.base.parameters(), key=lambda p: p.numel())
        assert base.shape[0] == shard, "the base shard must not be narrowed"
    assert report.expert_width == layer.expert_widths[0]
    assert "expert width" in report.summary()


# ------------------------------------------------------------- skeleton scale
def test_skeleton_scale_is_approximate_and_reports_kl():
    ids = probe_ids(24)
    torch.manual_seed(24)
    _, full = convert_model(build_model().eval(), growth_cfg(), probe=ids)
    torch.manual_seed(24)
    converted, half = convert_model(
        build_model().eval(), growth_cfg(skeleton_scale=0.5), probe=ids
    )

    assert half.lossless is False
    assert half.logit_kl is not None and half.logit_kl > 0.0
    # Dropping half the neurons must cost far more than the reconstruction error.
    assert half.max_carve_error > 1e-2
    assert half.max_carve_error > 100 * max(full.max_carve_error, 1e-9)
    assert "approximate" in half.summary()
    assert half.total_params < full.total_params

    layer = moe_layers(converted)[0]
    assert layer.skeleton_width == INTER // 2
    assert all(int(idx.max()) < INTER // 2 for idx in layer.index_sets)
    assert torch.cat(layer.index_sets).unique().numel() == INTER // 2


# -------------------------------------------------------------- replace mode
def test_replace_mode_is_lossless_at_variant_zero():
    torch.manual_seed(25)
    model = build_model().eval()
    ids = probe_ids(25)
    with torch.no_grad():
        reference = model(**ids).logits

    converted, report = convert_model(model, growth_cfg(expert_mode="replace"), probe=ids)

    # The conversion check forces variant 0, so the reported cost is the guarantee.
    assert report.lossless is True
    assert report.max_carve_error < 1e-5
    assert report.logit_kl is not None and report.logit_kl < 1e-6
    assert "variant 0 forced" in report.summary()

    # The same state reproduces the seed when set by hand after conversion.
    for layer in moe_layers(converted):
        torch.nn.init.zeros_(layer.router.weight)
        torch.nn.init.zeros_(layer.router.bias)
    converted.eval()
    with torch.no_grad():
        got = converted(**ids).logits
    assert (reference - got).abs().max() < 1e-5


def test_replace_mode_routes_exactly_one_variant_per_group():
    torch.manual_seed(26)
    ids = probe_ids(26, batch=4, seq=8)
    cfg = growth_cfg(expert_mode="replace")
    converted, _ = convert_model(build_model(), cfg, verify=False)

    layer = moe_layers(converted)[0]
    assert layer.router.n_groups == cfg.groups
    assert layer.router.n_variants == cfg.variants  # variant 0 is the base shard
    assert layer.expert_mode == "replace"

    converted.eval()
    with torch.no_grad():
        converted(**ids)
    stats = layer.pop_stats()
    assert stats is not None
    counts = stats["counts"]
    assert counts.numel() == cfg.groups * cfg.variants
    per_group = counts.reshape(cfg.groups, cfg.variants).sum(dim=1)
    assert torch.equal(per_group, torch.full_like(per_group, 32))  # 4 x 8 tokens
    assert int((counts > 0).sum()) > cfg.groups, "random routing must use residuals too"


def test_replace_mode_active_params_are_flat_in_k():
    shape = ModelShape(
        hidden=1024, intermediate=4096, layers=8, vocab=32000, n_heads=16, n_kv_heads=8
    )
    a0 = shape.converted(2, 4, 6, 0, expert_mode="replace")
    a2 = shape.converted(2, 4, 6, 2, expert_mode="replace")
    assert a0["active_params"] == a2["active_params"]
    assert a0["total_params"] > a0["active_params"]


# ------------------------------------------------------------- warm start (D4)
def test_router_warm_start_cluster_is_deterministic_and_keeps_labels():
    import copy

    ids = probe_ids(27, batch=4, seq=8)
    cfg = growth_cfg(router_warm_start="cluster", cluster_tokens=64, cluster_iters=8)

    torch.manual_seed(27)
    first = build_model()
    second = copy.deepcopy(first)  # same seed weights, so the same hidden states
    converted, report = convert_model(first, cfg, probe=ids, calibration=[ids])
    again, _ = convert_model(second, cfg, probe=ids, calibration=[ids])

    assert len(report.warm_start) == LAYERS
    assert all(w["cluster_agreement"] >= 0.5 for w in report.warm_start)
    assert report.warm_start[0]["clusters"] == cfg.variants - 1  # additive: residuals only
    for a, b in zip(moe_layers(converted), moe_layers(again)):
        assert torch.equal(a.router.weight, b.router.weight)
        assert torch.equal(a.router.bias, b.router.bias)
    assert "router warm start" in report.summary()


# ------------------------------------------------------------ scale accounting
def test_scale_accounting_matches_new_geometry():
    shape = ModelShape(
        hidden=1024, intermediate=4096, layers=8, vocab=32000, n_heads=16, n_kv_heads=8
    )
    shared, groups, variants, k = 2, 4, 6, 1
    for kwargs in (
        {"skeleton_scale": 0.5},
        {"expert_width_ratio": 0.5},
        {"expert_mode": "replace"},
        {"skeleton_scale": 0.5, "expert_width_ratio": 0.5, "expert_mode": "replace"},
    ):
        f = kwargs.get("skeleton_scale", 1.0)
        ratio = kwargs.get("expert_width_ratio", 1.0)
        mode = kwargs.get("expert_mode", "additive")
        row = shape.converted(shared, groups, variants, k, **kwargs)

        shard = 4096 * f / (shared + groups)
        expert = shard * ratio
        skeleton = shared + groups
        total_width = shard * (skeleton + groups * (variants - 1) * ratio)
        if mode == "replace":
            active_width = shard * skeleton
        else:
            active_width = shard * (skeleton + groups * k * ratio)
        per_width = 8 * 3 * 1024
        assert row["total_params"] == int(shape.non_ffn_params + per_width * total_width)
        assert row["active_params"] == int(shape.non_ffn_params + per_width * active_width)
        assert row["skeleton_width"] == pytest.approx(4096 * f)
        assert row["expert_width"] == pytest.approx(expert)
        assert row["lossless"] is (f == 1.0)


def test_report_active_params_follow_replace_accounting():
    torch.manual_seed(28)
    cfg = growth_cfg(expert_mode="replace")
    model, report = convert_model(build_model(), cfg, verify=False)
    shape = ModelShape(
        hidden=HIDDEN,
        intermediate=INTER,
        layers=LAYERS,
        vocab=VOCAB,
        n_heads=4,
        n_kv_heads=2,
        tie_embeddings=False,  # Llama has an untied lm_head
    )
    expected = shape.converted(
        cfg.shared, cfg.groups, cfg.variants, cfg.top_residuals, expert_mode="replace"
    )["active_params"]
    # scale.py excludes routers and counts norms slightly differently; 5% is a
    # structural check, not a bit-level one (that is what the default test does).
    assert report.active_params() == pytest.approx(expected, rel=0.05)
    assert report.active_params(top_residuals=0) == report.active_params(top_residuals=2)
    # Replace mode never activates the whole residual stack.
    assert report.active_params() < report.dense_params + report.residual_params


# ---------------------------------------------------------------- validation
def test_invalid_geometry_is_rejected():
    with pytest.raises(ValueError):
        MoEConfig(shared=1, groups=2, variants=4, mode="partition", expert_mode="replace")
    with pytest.raises(ValueError):
        MoEConfig(
            shared=1,
            groups=2,
            variants=4,
            expert_mode="replace",
            router=RouterConfig(group_top_k=2),
        )
    with pytest.raises(ValueError):
        MoEConfig(shared=1, groups=2, variants=4, skeleton_scale=0.0)
    with pytest.raises(ValueError):
        MoEConfig(shared=1, groups=2, variants=4, expert_width_ratio=1.5)
    with pytest.raises(ValueError):
        MoEConfig(shared=1, groups=2, variants=4, router_warm_start="kmeans")
    with pytest.raises(ValueError):
        MoEConfig(shared=1, groups=2, variants=4, dispatch="grouped")
    with pytest.raises(ValueError):
        MoEConfig(shared=1, groups=2, variants=4, dispatch="fused", expert_mode="replace")
    with pytest.raises(ValueError):
        MoEConfig(shared=1, groups=2, variants=4, dispatch="fused", mode="partition")
    with pytest.raises(ValueError):
        convert_model(
            build_model(),
            MoEConfig(shared=1, groups=2, variants=4, skeleton_scale=0.5, cluster_shards=True),
            verify=False,
        )


def _randomise_routing(model, seed: int) -> None:
    """Distinct residual weights plus a peaked router.

    With ``residual_init="zero"`` every residual is zero, so *any* routing map
    reproduces the skeleton and an equality check against the reference is
    vacuous. Distinct residuals make a wrong segment id, group offset or scale
    gather change the output; a peaked router makes the selection non-trivial.
    """

    generator = torch.Generator().manual_seed(seed)
    for layer in moe_layers(model):
        with torch.no_grad():
            for group in layer.groups:
                for expert in group.residuals:
                    for param in expert.parameters():
                        param.copy_(torch.randn(param.shape, generator=generator))
            layer.router.weight.copy_(
                torch.randn(layer.router.weight.shape, generator=generator) * 0.5
            )
            layer.router.bias.copy_(
                torch.randn(layer.router.bias.shape, generator=generator) * 0.5
            )


def _logits(model, ids):
    with torch.no_grad():
        return model(**ids).logits


def test_fused_dispatch_follows_the_routing_map():
    """Equality must be earned by the routing, not by zeroed residuals."""

    import copy

    ids = probe_ids(37)
    base = build_model()
    reference = copy.deepcopy(base)
    fused = copy.deepcopy(base)
    cfg = growth_cfg(
        top_residuals=2,
        router=RouterConfig(group_top_k=2),
        residual_init="copy",
    )
    convert_model(reference, cfg, probe=ids)
    converted, _ = convert_model(fused, replace(cfg, dispatch="fused"), probe=ids)
    _randomise_routing(reference, 9)
    _randomise_routing(converted, 9)
    reference.eval()
    converted.eval()

    logits = _logits(converted, ids)
    assert all(layer._fused is not None for layer in moe_layers(converted))
    assert torch.allclose(logits, _logits(reference, ids), rtol=1e-4, atol=1e-4)

    # Control 1: the residuals actually contribute, so equality is not skeleton-only.
    zeroed = copy.deepcopy(converted)
    for layer in moe_layers(zeroed):
        with torch.no_grad():
            for group in layer.groups:
                for expert in group.residuals:
                    for param in expert.parameters():
                        param.zero_()
    skeleton_only = _logits(zeroed, ids)
    assert (logits - skeleton_only).abs().max() > 0.01 * logits.abs().max()

    # Control 2: swapping two experts inside a group changes the output, so this
    # test would catch a routing map that picks the wrong variant.
    swapped = copy.deepcopy(converted)
    for layer in moe_layers(swapped):
        for group in layer.groups:
            with torch.no_grad():
                first = group.residuals[0].down_proj.weight.clone()
                second = group.residuals[1].down_proj.weight.clone()
                group.residuals[0].down_proj.weight.copy_(second)
                group.residuals[1].down_proj.weight.copy_(first)
    assert not torch.allclose(logits, _logits(swapped, ids), rtol=1e-4, atol=1e-4)


def test_config_makes_the_router_follow_top_residuals():
    """A caller cannot set top_residuals without the router matching it."""

    import warnings

    assert growth_cfg(top_residuals=3).router.group_top_k == 3
    # A matching explicit router is left alone and says nothing.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (
            growth_cfg(top_residuals=2, router=RouterConfig(group_top_k=2)).router.group_top_k
            == 2
        )
    # A conflicting one is corrected *and* reported, not silently trusted.
    with pytest.warns(UserWarning, match="disagrees with top_residuals"):
        assert (
            growth_cfg(top_residuals=2, router=RouterConfig(group_top_k=1)).router.group_top_k
            == 2
        )
    # Skeleton-only never consults the router, but it must stay valid.
    assert growth_cfg(top_residuals=0).router.group_top_k == 1
    # Replace selects exactly one variant per group (top-1 over base + residuals).
    assert growth_cfg(top_residuals=2, expert_mode="replace").router.group_top_k == 1
    # ``dataclasses.replace`` is the normal way to change k: silent, not a conflict.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert replace(growth_cfg(), top_residuals=3).router.group_top_k == 3
        assert replace(growth_cfg(top_residuals=2), top_residuals=1).router.group_top_k == 1


def test_cli_top_k_configures_the_router():
    """``--top-k`` must reach both the accounting and the router."""

    from orrg.cli import build_parser, moe_config_from_args

    parser = build_parser()
    default = moe_config_from_args(parser.parse_args(["convert", "--model", "unused"]))
    assert (default.top_residuals, default.router.group_top_k) == (1, 1)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the CLI derives the router, so no warning
        two = moe_config_from_args(
            parser.parse_args(["convert", "--model", "unused", "--top-k", "2"])
        )
    assert (two.top_residuals, two.router.group_top_k) == (2, 2)

    alias = moe_config_from_args(
        parser.parse_args(["convert", "--model", "unused", "--top-residuals", "3"])
    )
    assert (alias.top_residuals, alias.router.group_top_k) == (3, 3)

    # Replace mode picks one variant per group regardless of the accounting knob.
    replace = moe_config_from_args(
        parser.parse_args(
            ["convert", "--model", "unused", "--top-k", "2", "--expert-mode", "replace"]
        )
    )
    assert (replace.top_residuals, replace.router.group_top_k) == (2, 1)

    # Partition mode routes shards, so it takes --top-k and has no residual experts.
    partition = moe_config_from_args(
        parser.parse_args(["convert", "--model", "unused", "--mode", "partition", "--top-k", "2"])
    )
    assert (partition.top_residuals, partition.top_k_partition) == (0, 2)
    with pytest.raises(SystemExit):
        moe_config_from_args(
            parser.parse_args(
                ["convert", "--model", "unused", "--mode", "partition", "--top-residuals", "2"]
            )
        )

    with pytest.raises(SystemExit):
        moe_config_from_args(
            parser.parse_args(
                ["convert", "--model", "unused", "--top-k", "2", "--top-residuals", "3"]
            )
        )


def test_converted_model_routes_the_reported_expert_count():
    """The routed count must equal top_residuals, not silently stay at one."""

    from orrg.cli import build_parser, moe_config_from_args

    ids = probe_ids(41)
    cfg = moe_config_from_args(
        build_parser().parse_args(["convert", "--model", "unused", "--top-k", "2"])
    )
    converted, report = convert_model(build_model(), cfg, probe=ids)

    x = torch.randn(6, HIDDEN)
    for layer in moe_layers(converted):
        with torch.no_grad():
            route = layer.router(x)
        assert route.index.shape[-1] == cfg.top_residuals == 2
        # Indices address residuals, never the base shard (id ``variants - 1``).
        assert int(route.index.max()) <= layer.cfg.variants - 2

    assert report.active_params(1) < report.active_params(2) == report.active_params()


def test_fused_dispatch_handles_narrow_experts():
    """ratio<1 makes residuals narrower than shards; the stacks must still fuse."""

    import copy

    ids = probe_ids(44)
    base = build_model()
    reference = copy.deepcopy(base)
    fused = copy.deepcopy(base)
    cfg = growth_cfg(expert_width_ratio=0.5, residual_init="copy", top_residuals=2)
    convert_model(reference, cfg, probe=ids)
    converted, _ = convert_model(fused, replace(cfg, dispatch="fused"), probe=ids)
    _randomise_routing(reference, 12)
    _randomise_routing(converted, 12)
    reference.eval()
    converted.eval()
    assert torch.allclose(
        _logits(converted, ids), _logits(reference, ids), rtol=1e-4, atol=1e-4
    )


def test_fused_dispatch_handles_large_token_counts():
    """Many rows per call (as in prefill) must still match the reference."""

    import copy

    ids = probe_ids(45, batch=8, seq=32)
    base = build_model()
    reference = copy.deepcopy(base)
    fused = copy.deepcopy(base)
    cfg = growth_cfg(top_residuals=2, residual_init="copy")
    convert_model(reference, cfg, probe=ids)
    converted, _ = convert_model(fused, replace(cfg, dispatch="fused"), probe=ids)
    _randomise_routing(reference, 13)
    _randomise_routing(converted, 13)
    reference.eval()
    converted.eval()
    assert torch.allclose(
        _logits(converted, ids), _logits(reference, ids), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("threshold,budget", [(0, 8_192), (10**9, 8_192), (10**9, 24)])
def test_fused_dispatch_batched_and_grouped_paths_agree(threshold, budget, monkeypatch):
    """The small-batch batched path and the grouped path must compute the same thing."""

    import copy

    import orrg.dispatch as dispatch

    monkeypatch.setattr(dispatch, "BATCHED_BELOW_ROWS", threshold)
    # A tiny budget forces the residual block through many pieces, which is the path
    # the default budget only reaches at large token counts.
    monkeypatch.setattr(dispatch, "BATCHED_ROW_BUDGET", budget)
    ids = probe_ids(47)
    base = build_model()
    reference = copy.deepcopy(base)
    fused = copy.deepcopy(base)
    cfg = growth_cfg(
        top_residuals=2,
        router=RouterConfig(group_top_k=2),
        residual_init="copy",
    )
    convert_model(reference, cfg, probe=ids)
    converted, _ = convert_model(fused, replace(cfg, dispatch="fused"), probe=ids)
    _randomise_routing(reference, 15)
    _randomise_routing(converted, 15)
    reference.eval()
    converted.eval()
    assert torch.allclose(
        _logits(converted, ids), _logits(reference, ids), rtol=1e-4, atol=1e-4
    )
    took_batched = any(layer._fused._halves is not None for layer in moe_layers(converted))
    assert took_batched is (threshold > 0), f"threshold {threshold}: batched={took_batched}"


def test_batched_block_is_bounded_by_the_row_budget():
    """No padded block may exceed the budget, whatever the capacity is."""

    from orrg.dispatch import chunk_plan

    budget = 1_024
    for capacity in (1, 7, 64, 1_024, 4_096, 16_384):
        lane_chunk, segment_chunk = chunk_plan(capacity, budget)
        assert lane_chunk <= budget, (capacity, lane_chunk)
        assert lane_chunk * segment_chunk <= budget, (capacity, lane_chunk, segment_chunk)
        # Together the pieces must cover every lane and every segment exactly once.
        lanes = range(0, capacity, lane_chunk)
        assert sum(1 for _ in lanes) * lane_chunk >= capacity
        assert segment_chunk >= 1
    # The default plan is the one the library uses.
    assert chunk_plan(8_192) == chunk_plan(8_192, 8_192)


def test_fused_dispatch_handles_skeleton_only():
    """top_residuals=0 skips routing; the fused path must skip it too."""

    import copy

    ids = probe_ids(39)
    base = build_model()
    reference = copy.deepcopy(base)
    fused = copy.deepcopy(base)
    convert_model(reference, growth_cfg(top_residuals=0), probe=ids)
    converted, _ = convert_model(
        fused, growth_cfg(top_residuals=0, dispatch="fused"), probe=ids
    )
    with torch.no_grad():
        assert torch.allclose(converted(**ids).logits, reference(**ids).logits, atol=1e-6)


def test_fused_dispatch_matches_reference_forward():
    """The grouped-GEMM serving path must reproduce the reference forward."""

    import copy

    ids = probe_ids(34)
    base = build_model()
    reference = copy.deepcopy(base)
    fused = copy.deepcopy(base)
    convert_model(reference, growth_cfg(), probe=ids)
    converted, report = convert_model(fused, growth_cfg(dispatch="fused"), probe=ids)

    with torch.no_grad():
        ref_logits = reference(**ids).logits
        fast_logits = converted(**ids).logits
    assert torch.allclose(ref_logits, fast_logits, atol=1e-5, rtol=1e-4)
    assert report.logit_kl is not None and report.logit_kl < 1e-6
    # The fused path was actually taken, not silently skipped.
    assert all(layer._fused is not None for layer in moe_layers(converted))

    # It carries no autograd, so using it outside no_grad must fail loudly.
    with pytest.raises(RuntimeError, match="autograd"):
        converted(**ids)

    torch.manual_seed(35)
    bf16_model = build_model().to(torch.bfloat16)
    converted_bf16, _ = convert_model(bf16_model, growth_cfg(dispatch="fused"), probe=ids)
    with torch.no_grad():
        out = converted_bf16(**ids).logits
    assert out.dtype == torch.bfloat16 and torch.isfinite(out).all()


@pytest.mark.parametrize("mode", ["additive", "replace"])
def test_conversion_works_in_bfloat16(mode):
    """The real checkpoints are bf16; routing must not promote the residual sum."""

    torch.manual_seed(31)
    model = build_model().to(torch.bfloat16)
    ids = probe_ids(31, batch=2, seq=8)
    converted, report = convert_model(model, growth_cfg(expert_mode=mode), probe=ids)
    assert report.logit_kl is not None and report.logit_kl < 1e-6
    # With a float32 accumulator the only error is the final bf16 cast (~2e-3
    # here). Summing shards in bf16 instead lands near 5e-2, so this bound is
    # what keeps the accumulator honest.
    assert report.max_carve_error < 0.01
    with torch.no_grad():
        logits = converted(**ids).logits
    assert logits.dtype == torch.bfloat16
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("ratio", [0.0, -0.5, 1.0001, 1.5, 2.0])
def test_expert_width_ratio_boundary_is_enforced_by_converter_and_planner(ratio):
    """The planner must not print a geometry the converter refuses."""
    shape = ModelShape(
        hidden=1024, intermediate=4096, layers=8, vocab=32000, n_heads=16, n_kv_heads=8
    )
    with pytest.raises(ValueError, match="expert_width_ratio"):
        MoEConfig(shared=2, groups=4, variants=6, expert_width_ratio=ratio)
    with pytest.raises(ValueError, match="expert_width_ratio"):
        shape.converted(2, 4, 6, expert_width_ratio=ratio)
    with pytest.raises(ValueError, match="expert_width_ratio"):
        converted_scale(shape, 2, 4, 6, expert_width_ratio=ratio)


def test_spec_distillation_rows_are_buildable():
    """§2.5's KD rows must be expressible at ratio=1 with integer k."""

    two_b = ModelShape(
        hidden=2560, intermediate=6912, layers=24, vocab=128000, n_heads=20, n_kv_heads=5
    )
    large = ModelShape(
        hidden=4096, intermediate=10240, layers=32, vocab=128000, n_heads=32, n_kv_heads=8
    )
    for shape, s, g, v, k, total, active in (
        (two_b, 4, 4, 25, 2, 9.00e9, 2.00e9),
        (large, 4, 4, 24, 2, 27.03e9, 5.89e9),
    ):
        # The spec row must be a config the converter accepts...
        MoEConfig(shared=s, groups=g, variants=v, top_residuals=k, skeleton_scale=0.5)
        row = shape.converted(s, g, v, k, skeleton_scale=0.5)
        # ...and the planner must reproduce the numbers printed in the table.
        assert row["total_params"] == pytest.approx(total, rel=0.01)
        assert row["active_params"] == pytest.approx(active, rel=0.01)
        assert row["lossless"] is False
        assert row["expert_width"] == pytest.approx(row["shard_width"])


def test_expert_width_ratio_one_is_the_inclusive_boundary():
    shape = ModelShape(
        hidden=1024, intermediate=4096, layers=8, vocab=32000, n_heads=16, n_kv_heads=8
    )
    cfg = MoEConfig(shared=2, groups=4, variants=6, expert_width_ratio=1.0)
    assert cfg.expert_width_ratio == 1.0
    row = shape.converted(2, 4, 6, expert_width_ratio=1.0)
    assert row["lossless"] is True
    assert row["expert_width"] == pytest.approx(row["shard_width"])

    from orrg.cli import main

    base = [
        "scale", "--hidden", "1024", "--inter", "4096", "--layers", "8",
        "--vocab", "32000", "--shared", "2", "--groups", "4", "--variants", "6",
    ]
    assert main(base + ["--expert-ratio", "1.0"]) == 0
    with pytest.raises(ValueError, match="expert_width_ratio"):
        main(base + ["--expert-ratio", "1.5"])


def test_cli_scale_reports_new_geometry(capsys):
    from orrg.cli import main

    base = ["scale", "--hidden", "1024", "--inter", "4096", "--layers", "8", "--vocab", "32000"]
    assert main(base + ["--shared", "2", "--groups", "4", "--variants", "6"]) == 0
    plain = capsys.readouterr().out
    assert "skeleton" not in plain and "expert mode" not in plain

    assert (
        main(
            base
            + [
                "--shared",
                "2",
                "--groups",
                "4",
                "--variants",
                "6",
                "--skeleton-scale",
                "0.5",
                "--expert-ratio",
                "0.5",
                "--expert-mode",
                "replace",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "approximate conversion" in out
    assert "expert width" in out
    assert "replace" in out
