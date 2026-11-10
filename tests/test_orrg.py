"""Tests for ORRG conversion: architecture coverage, losslessness, routing, scale math."""

from __future__ import annotations

import copy
import math

import pytest
import torch

from orrg import (
    MoEConfig,
    convert_model,
    discover_ffn_modules,
    expert_diversity,
    freeze_non_moe,
    freeze_skeleton,
    load_converted,
    save_converted,
)
from orrg.cluster import balanced_kmeans, collect_activations
from orrg.distill import (
    contribution_entropy,
    contribution_targets,
    kl_divergence,
    router_distill_loss,
)
from orrg.train import (
    TrainConfig,
    orthogonality_loss,
    router_distill_total,
    set_router_distill,
    train,
)
from orrg.metrics import collect_metrics, conversion_error, residual_share
from orrg.router import update_router_bias
from orrg.scale import ModelShape, converted_scale

HIDDEN = 32
INTER = 64
LAYERS = 2
VOCAB = 128
ARCHS = ["llama", "gpt2", "phi", "mistral", "qwen2"]


def build_model(arch: str):
    from transformers import (
        GPT2Config,
        GPT2LMHeadModel,
        LlamaConfig,
        LlamaForCausalLM,
        MistralConfig,
        MistralForCausalLM,
        OPTConfig,
        OPTForCausalLM,
        PhiConfig,
        PhiForCausalLM,
        Qwen2Config,
        Qwen2ForCausalLM,
    )

    common = dict(hidden_size=HIDDEN, intermediate_size=INTER, num_hidden_layers=LAYERS,
                  num_attention_heads=4, num_key_value_heads=2, vocab_size=VOCAB)
    if arch == "llama":
        return LlamaForCausalLM(LlamaConfig(**common))
    if arch == "gpt2":
        return GPT2LMHeadModel(GPT2Config(n_embd=HIDDEN, n_head=4, n_layer=LAYERS, n_inner=INTER, vocab_size=VOCAB))
    if arch == "phi":
        return PhiForCausalLM(PhiConfig(**common))
    if arch == "opt":
        return OPTForCausalLM(OPTConfig(hidden_size=HIDDEN, ffn_dim=INTER, num_hidden_layers=LAYERS,
                                        num_attention_heads=4, word_embed_proj_dim=HIDDEN, vocab_size=VOCAB))
    if arch == "mistral":
        return MistralForCausalLM(MistralConfig(**common))
    if arch == "qwen2":
        return Qwen2ForCausalLM(Qwen2Config(**common))
    raise ValueError(arch)


def probe_ids(seed: int = 0, batch: int = 2, seq: int = 8) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {"input_ids": torch.randint(0, VOCAB, (batch, seq), generator=generator)}


def growth_cfg(**overrides) -> MoEConfig:
    values = dict(shared=1, groups=2, variants=4, top_residuals=1, hidden_size=HIDDEN)
    values.update(overrides)
    return MoEConfig(**values)


# --------------------------------------------------------------------- discovery
@pytest.mark.parametrize("arch", ARCHS)
def test_discovery_finds_every_ffn_and_nothing_else(arch):
    model = build_model(arch)
    specs = discover_ffn_modules(model)
    assert len(specs) == LAYERS
    assert {s.hidden for s in specs} == {HIDDEN}
    assert {s.intermediate for s in specs} == {INTER}
    # attention must never be picked up, including GPT-2's fused qkv
    assert all("attn" not in s.name for s in specs)
    assert all(s.name.endswith(".mlp") for s in specs)


# ------------------------------------------------------------------- losslessness
@pytest.mark.parametrize("arch", ARCHS)
def test_conversion_is_lossless(arch):
    torch.manual_seed(0)
    model = build_model(arch).eval()
    ids = probe_ids()
    with torch.no_grad():
        reference = model(**ids).logits

    model, report = convert_model(model, growth_cfg(), probe=ids)
    model.eval()
    with torch.no_grad():
        converted = model(**ids).logits

    assert report.max_carve_error < 1e-5, report.max_carve_error
    assert (reference - converted).abs().max() < 1e-5
    assert report.logit_max_abs is not None and report.logit_max_abs < 1e-5
    assert report.total_params > report.dense_params


@pytest.mark.parametrize("arch", ["llama", "gpt2"])
def test_partition_mode_is_lossless_when_all_shards_active(arch):
    torch.manual_seed(1)
    model = build_model(arch).eval()
    ids = probe_ids(seed=1)
    with torch.no_grad():
        reference = model(**ids).logits
    cfg = MoEConfig(shared=0, groups=4, variants=1, mode="partition", top_k_partition=4, hidden_size=HIDDEN)
    model, report = convert_model(model, cfg, probe=ids)
    model.eval()
    with torch.no_grad():
        got = model(**ids).logits
    assert report.max_carve_error < 1e-5
    assert (reference - got).abs().max() < 1e-5


def test_sparse_partition_mode_is_not_exact():
    torch.manual_seed(2)
    model = build_model("llama").eval()
    ids = probe_ids(seed=2)
    with torch.no_grad():
        reference = model(**ids).logits
    cfg = MoEConfig(shared=0, groups=4, variants=1, mode="partition", top_k_partition=2, hidden_size=HIDDEN)
    model, _ = convert_model(model, cfg, probe=ids)
    model.eval()
    with torch.no_grad():
        got = model(**ids).logits
    # The sparse regime is approximate by construction; this asserts the check above bites.
    assert (reference - got).abs().max() > 1e-4


# ------------------------------------------------------------------------ growth
def test_residuals_start_at_zero_and_train_from_step_one():
    torch.manual_seed(3)
    model = build_model("llama")
    ids = probe_ids(seed=3)
    model, report = convert_model(model, growth_cfg(), probe=ids)
    layers = [m for m in model.modules() if getattr(m, "is_orrg_moe", False)]
    assert len(layers) == LAYERS

    x = torch.randn(6, HIDDEN)
    for layer in layers:
        for group in layer.groups:
            for expert in group.residuals:
                assert torch.count_nonzero(expert(x)) == 0

    model.train()
    logits = model(**ids).logits
    loss = torch.logsumexp(logits.float(), dim=-1).mean()
    loss.backward()

    layer = layers[0]
    assert layer.router.weight.grad is not None
    # Residual experts receive gradient even though their output is zero:
    # the router weight is non-zero, so the down-projection grows immediately.
    down_grads = [
        p.grad.abs().max().item()
        for group in layer.groups
        for expert in group.residuals
        for name, p in expert.named_parameters()
        if "down_proj" in name and p.grad is not None
    ]
    assert down_grads and max(down_grads) > 0

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    optimizer.step()
    moved = [
        torch.count_nonzero(expert(x)).item()
        for group in layer.groups
        for expert in group.residuals
    ]
    assert any(m > 0 for m in moved), "residual experts never left zero"


def test_grouped_routing_is_structurally_balanced():
    torch.manual_seed(4)
    model = build_model("llama")
    ids = probe_ids(seed=4, batch=64, seq=8)
    cfg = growth_cfg(groups=2, variants=4, top_residuals=1)
    model, _ = convert_model(model, cfg, verify=False)
    model.eval()
    with torch.no_grad():
        model(**ids)

    layers = [m for m in model.modules() if getattr(m, "is_orrg_moe", False)]
    n_tokens = 64 * 8
    for layer in layers:
        counts = layer.last_stats["counts"]
        assert counts.numel() == cfg.n_residuals
        assert int(counts.sum()) == cfg.groups * cfg.top_residuals * n_tokens
        per_group = counts.reshape(cfg.groups, cfg.residual_experts_per_group).sum(-1)
        assert torch.all(per_group == cfg.top_residuals * n_tokens)


def test_router_bias_moves_against_overload():
    bias = torch.zeros(3)
    counts = torch.tensor([90.0, 5.0, 5.0])
    update_router_bias(bias, counts, rate=0.5)
    assert bias[0] < 0 < bias[1]
    assert abs(float(bias.mean())) < 1e-6


# ------------------------------------------------------------------- persistence
def test_save_load_roundtrip():
    torch.manual_seed(5)
    dense = build_model("llama").eval()
    twin = copy.deepcopy(dense)
    ids = probe_ids(seed=5)
    cfg = growth_cfg()

    model, _ = convert_model(dense, cfg, verify=False)
    model.eval()
    with torch.no_grad():
        expected = model(**ids).logits

    directory = "outputs"
    import os

    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "orrg-roundtrip.pt")
    save_converted(model, cfg, path, only_moe=False)
    restored, _ = load_converted(twin, cfg, path)
    restored.eval()
    with torch.no_grad():
        got = restored(**ids).logits
    assert torch.allclose(expected, got, atol=1e-6)


def test_freeze_helpers_select_the_right_parameters():
    torch.manual_seed(6)
    model = build_model("llama")
    model, _ = convert_model(model, growth_cfg(), verify=False)

    freeze_non_moe(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable > 0
    assert all(not p.requires_grad for n, p in model.named_parameters() if ".mlp" not in n)

    model, _ = convert_model(build_model("llama"), growth_cfg(), verify=False)
    freeze_skeleton(model)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert names and all(".router." in n or ".residuals." in n for n in names)


# ---------------------------------------------------------------------- metrics
def test_metrics_and_conversion_error():
    torch.manual_seed(7)
    model = build_model("llama").eval()
    ids = probe_ids(seed=7)
    with torch.no_grad():
        reference = model(**ids).logits
    model, _ = convert_model(model, growth_cfg(), probe=ids)
    model.eval()
    with torch.no_grad():
        got = model(**ids).logits

    error = conversion_error(reference, got)
    assert error["kl"] < 1e-6
    assert error["top1_agreement"] == 1.0

    metrics = collect_metrics(model)
    assert metrics["layers"] == LAYERS
    assert 0.0 <= metrics["router_entropy"] <= 1.0001
    assert expert_diversity(model) >= 0.0


# ---------------------------------------------------------------- distillation
def test_distillation_losses():
    torch.manual_seed(8)
    model = build_model("llama")
    model, _ = convert_model(model, growth_cfg(), verify=False)
    layer = [m for m in model.modules() if getattr(m, "is_orrg_moe", False)][0]

    logits = torch.randn(4, 5, VOCAB)
    assert float(kl_divergence(logits, logits)) == pytest.approx(0.0, abs=1e-6)
    assert float(kl_divergence(logits, logits * 0.5, reverse=True)) > 1e-6

    x = torch.randn(3, 4, HIDDEN)
    targets = contribution_targets(layer, x)
    assert len(targets) == 2
    for target in targets:
        assert torch.allclose(target.sum(-1), torch.ones(target.shape[0]), atol=1e-5)

    loss = router_distill_loss(layer, x)
    assert torch.isfinite(loss) and float(loss) > 0


# ------------------------------------------------------------------------ scale
def test_scale_reproduces_the_27b_plan():
    shape = ModelShape(
        hidden=4096,
        intermediate=10240,
        layers=32,
        vocab=128000,
        n_heads=32,
        n_kv_heads=8,
    )
    scale = converted_scale(shape, shared=2, groups=6, variants=8)

    dense = shape.dense_params
    assert 5.5e9 < dense < 6.7e9

    at_zero = scale["converted"][0]
    assert at_zero["total_params"] > 25e9
    assert at_zero["active_params"] == pytest.approx(dense, rel=1e-6)
    assert 3.5 < scale["expansion"] < 5.5

    at_one = scale["converted"][1]
    assert at_one["active_params"] > at_zero["active_params"]
    assert at_one["active_params"] < 0.45 * at_one["total_params"]


def test_active_parameter_accounting_matches_report():
    torch.manual_seed(9)
    model = build_model("llama")
    cfg = growth_cfg(groups=2, variants=4, top_residuals=1)
    model, report = convert_model(model, cfg, verify=False)

    # Llama has no FFN biases, so the skeleton's parameters equal the dense FFN exactly;
    # active params at k=0 are therefore the dense model plus the new routers.
    assert report.active_params(top_residuals=0) == report.dense_params + report.router_params
    assert report.active_params(top_residuals=0) == (
        report.non_moe_params + report.skeleton_params + report.router_params
    )
    assert report.total_params == (
        report.non_moe_params + report.skeleton_params + report.residual_params + report.router_params
    )
    per_expert = report.residual_params / cfg.n_residuals
    expected = report.non_moe_params + report.skeleton_params + report.router_params + cfg.groups * per_expert
    assert report.active_params(top_residuals=1) == int(expected)
    assert report.active_params(top_residuals=2) > report.active_params(top_residuals=1)


def test_clustered_sharding_is_lossless_and_balanced():
    torch.manual_seed(11)
    model = build_model("llama").eval()
    ids = probe_ids(seed=11, batch=4, seq=8)
    with torch.no_grad():
        reference = model(**ids).logits

    cfg = growth_cfg(cluster_shards=True, cluster_tokens=64, cluster_iters=4)
    model, report = convert_model(model, cfg, probe=ids, calibration=[ids])
    model.eval()
    with torch.no_grad():
        got = model(**ids).logits

    # Clustered shards sum with a different floating-point order than contiguous
    # slices, so allow a slightly looser bound than the 1e-5 contiguous case.
    assert (reference - got).abs().max() < 1e-4
    assert len(report.clustering) == LAYERS
    assert all(stat.cohesion >= 0.0 and stat.baseline >= 0.0 for stat in report.clustering)

    layer = [m for m in model.modules() if getattr(m, "is_orrg_moe", False)][0]
    union = torch.cat(layer.index_sets).sort().values
    assert torch.equal(union, torch.arange(INTER))
    assert max(layer.widths) - min(layer.widths) <= 1


def test_balanced_kmeans_partitions_every_neuron():
    torch.manual_seed(12)
    features = torch.randn(64, 32)
    clusters = balanced_kmeans(features, 8, iters=5)
    union = torch.cat(clusters)
    assert union.numel() == 64
    assert union.unique().numel() == 64
    sizes = [len(c) for c in clusters]
    assert max(sizes) - min(sizes) <= 1


def test_activation_collection_samples_post_activation_states():
    torch.manual_seed(13)
    model = build_model("llama").eval()
    specs = discover_ffn_modules(model)
    ids = probe_ids(seed=13, batch=2, seq=6)
    activations = collect_activations(model, specs, [ids], max_tokens=10)
    for spec in specs:
        tensor = activations[spec.name]
        assert tensor.shape[1] == INTER
        assert 0 < tensor.shape[0] <= 10


def test_router_distillation_gives_router_gradient_at_init():
    torch.manual_seed(14)
    model = build_model("llama")
    ids = probe_ids(seed=14)
    model, _ = convert_model(model, growth_cfg(), verify=False)
    layer = [m for m in model.modules() if getattr(m, "is_orrg_moe", False)][0]

    model.train()
    set_router_distill(model, 0.1)
    with torch.no_grad():
        for group in layer.groups:
            for expert in group.residuals:
                assert torch.count_nonzero(expert(torch.randn(4, HIDDEN))) == 0
    model.zero_grad(set_to_none=True)
    model(**ids)
    distill = router_distill_total(model)
    assert distill is not None and torch.isfinite(distill)
    distill.backward()
    # The task loss cannot move the router while residuals are zero; the
    # distillation term can, which is the point of the mechanism.
    assert layer.router.weight.grad is not None
    assert layer.router.weight.grad.abs().max() > 0


def test_train_loop_with_router_distillation():
    torch.manual_seed(15)
    model = build_model("llama")
    ids = probe_ids(seed=15, batch=2, seq=8)
    model, _ = convert_model(model, growth_cfg(), verify=False)
    freeze_non_moe(model)
    batches = [{"input_ids": ids["input_ids"], "labels": ids["input_ids"]} for _ in range(4)]
    cfg = TrainConfig(lr=1e-3, router_distill_coef=0.1, bf16=False, log_every=0)
    result = train(model, batches, 3, cfg)
    assert math.isfinite(result["loss_mean"])
    assert result["distill_mean"] is not None
    assert result["trainable_params"] > 0


def test_split_ffn_models_are_rejected_with_a_clear_error():
    model = build_model("opt")
    from orrg import discover_ffn_modules as discover

    assert discover(model) == []
    with pytest.raises(ValueError, match="split-FFN"):
        convert_model(model, growth_cfg(), verify=False)


def test_residual_share_is_zero_at_conversion_and_grows_after_update():
    torch.manual_seed(16)
    model = build_model("llama").eval()
    ids = probe_ids(seed=16)
    model, _ = convert_model(model, growth_cfg(), verify=False)
    # Zeroed down-projections make every residual output exactly zero, so the
    # extra capacity starts completely inert.
    assert residual_share(model, ids) == 0.0

    for module in model.modules():
        if getattr(module, "is_orrg_moe", False):
            for group in module.groups:
                for expert in group.residuals:
                    for parameter in expert.parameters():
                        parameter.data.normal_(0.0, 0.05)
    assert residual_share(model, ids) > 0.0


def test_train_calls_the_trace_hook_on_schedule():
    torch.manual_seed(17)
    model = build_model("llama")
    ids = probe_ids(seed=17)
    model, _ = convert_model(model, growth_cfg(), verify=False)
    batches = [{"input_ids": ids["input_ids"], "labels": ids["input_ids"]}]
    calls: list[int] = []
    cfg = TrainConfig(lr=1e-3, bf16=False, log_every=0, trace_every=2)
    train(model, batches, 5, cfg, trace=calls.append)
    assert calls == [2, 4]


def test_orthogonality_pressure_starts_at_the_hinge_and_relaxes():
    torch.manual_seed(18)
    model = build_model("llama")
    ids = probe_ids(seed=18)
    # Exact copies put the hinge at its maximum and give it zero gradient, which
    # is why the pressure needs a symmetry-breaking init to act on.
    model, _ = convert_model(model, growth_cfg(residual_noise=0.05), verify=False)
    start = float(orthogonality_loss(model, margin=0.5))
    assert start < 0.25

    batches = [{"input_ids": ids["input_ids"], "labels": ids["input_ids"]}]
    cfg = TrainConfig(lr=1e-2, bf16=False, log_every=0, orth_step=1e-4, orth_margin=0.5)
    train(model, batches, 5, cfg)
    assert float(orthogonality_loss(model, margin=0.5)) < start


def test_residual_noise_breaks_symmetry_without_breaking_losslessness():
    torch.manual_seed(20)
    model = build_model("llama").eval()
    ids = probe_ids(seed=20)
    with torch.no_grad():
        reference = model(**ids).logits

    model, report = convert_model(model, growth_cfg(residual_noise=0.05), probe=ids)
    model.eval()
    with torch.no_grad():
        converted = model(**ids).logits

    # The jitter touches only up-projections, and the down-projection is zeroed,
    # so the residual output is still exactly zero and conversion is exact.
    assert (reference - converted).abs().max() < 1e-5
    assert report.logit_kl is not None and report.logit_kl < 1e-6
    # But the residuals are no longer exact copies, so a cosine pressure has a
    # gradient to act on.
    assert expert_diversity(model) > 0.0
    assert float(orthogonality_loss(model, margin=0.5)) < 0.25


def test_scaled_residual_init_trades_exactness_for_function():
    torch.manual_seed(21)
    model = build_model("llama").eval()
    ids = probe_ids(seed=21)
    cfg = growth_cfg(residual_init="scaled", down_init_scale=0.05)
    model, report = convert_model(model, cfg, probe=ids)

    # The magnitude lever is explicitly not lossless: residuals contribute from
    # step 0, so conversion KL is small but nonzero.
    assert report.logit_kl is not None
    assert 0.0 < report.logit_kl < 0.1
    assert report.max_carve_error > 0.0
    # And that is exactly what the residual-share metric should detect.
    assert residual_share(model, ids) > 0.0

    with pytest.raises(ValueError, match="down_init_scale"):
        MoEConfig(residual_init="scaled", down_init_scale=0.0, hidden_size=HIDDEN)


def test_sharper_contribution_target_is_less_uniform():
    torch.manual_seed(19)
    model, _ = convert_model(build_model("llama"), growth_cfg(), verify=False)
    for module in model.modules():
        if getattr(module, "is_orrg_moe", False):
            for group in module.groups:
                for expert in group.residuals:
                    for parameter in expert.parameters():
                        parameter.data.normal_(0.0, 0.05)
    layer = [m for m in model.modules() if getattr(m, "is_orrg_moe", False)][0]
    x = torch.randn(16, HIDDEN)
    soft = contribution_entropy(contribution_targets(layer, x, temperature=1.0))
    sharp = contribution_entropy(contribution_targets(layer, x, temperature=0.1))
    assert sharp >= soft
