"""Activation-clustered neuron sharding.

Contiguous sharding is the cheapest partition and is what makes the conversion
exact, but it ignores which neurons fire together. MoEfication-style carving
groups neurons by co-activation so that each shard is a coherent function.

Calibration is model-agnostic: we hook the *input* of each block's
down-projection, which is exactly the post-activation hidden vector of that FFN
(gated, non-gated, Linear or Conv1D alike), then cluster neurons by their
activation profile across a token sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch import nn

from .ffn import FFNSpec


@dataclass
class ClusterStats:
    """Diagnostics for one layer's clustering."""

    layer: str
    n_neurons: int
    n_clusters: int
    cohesion: float  # mean cosine of neurons to their cluster centroid
    baseline: float  # mean cosine to a random centroid
    balance: float  # 1.0 = perfectly equal shard sizes


def _down_owners(spec: FFNSpec) -> list[nn.Module]:
    owners = []
    for slice_spec in spec.down:
        parent = slice_spec.name.rsplit(".", 1)[0]
        owners.append(spec.module.get_submodule(parent))
    return owners


@torch.no_grad()
def collect_activations(
    model: nn.Module,
    specs: list[FFNSpec],
    batches: Iterable[Mapping],
    *,
    max_tokens: int = 4096,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Sample post-activation FFN hidden states per layer.

    ``batches`` is an iterable of model kwargs dicts (e.g. ``{"input_ids": ...}``)
    or of tensors. Only ``max_tokens`` tokens are kept per layer.
    """

    generator = torch.Generator().manual_seed(seed)
    chunks: dict[str, list[torch.Tensor]] = {spec.name: [] for spec in specs}
    counts: dict[str, int] = {spec.name: 0 for spec in specs}
    handles = []

    def make_hook(name: str):
        def hook(_module, args):
            tensor = args[0]
            if not torch.is_tensor(tensor):
                return
            flat = tensor.reshape(-1, tensor.shape[-1]).detach()
            remaining = max_tokens - counts[name]
            if remaining <= 0:
                return
            if flat.shape[0] > remaining:
                pick = torch.randperm(flat.shape[0], generator=generator)[:remaining]
                flat = flat[pick]
            chunks[name].append(flat.to(torch.float32).cpu())
            counts[name] += flat.shape[0]

        return hook

    for spec in specs:
        for owner in _down_owners(spec):
            handles.append(owner.register_forward_pre_hook(make_hook(spec.name)))

    was_training = model.training
    model.eval()
    try:
        for batch in batches:
            if isinstance(batch, Mapping):
                model(**batch)
            else:
                model(batch)
            if all(counts[s.name] >= max_tokens for s in specs):
                break
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return {
        name: torch.cat(parts) if parts else torch.zeros(0, 0)
        for name, parts in chunks.items()
    }


def _balanced_assign(distances: torch.Tensor, capacities: torch.Tensor) -> torch.Tensor:
    """Greedy capacity-constrained assignment, most-confident neurons first."""

    n = distances.shape[0]
    top2 = distances.topk(2, dim=1, largest=False).values
    best = distances.argmin(dim=1)
    regret = top2[:, 1] - top2[:, 0]
    order = torch.argsort(regret, descending=True)

    labels = torch.zeros(n, dtype=torch.long)
    remaining = capacities.clone()
    for index in order.tolist():
        choice = int(best[index])
        if remaining[choice] > 0:
            labels[index] = choice
            remaining[choice] -= 1
            continue
        for candidate in torch.argsort(distances[index]).tolist():
            if remaining[candidate] > 0:
                labels[index] = candidate
                remaining[candidate] -= 1
                break
    return labels


def balanced_kmeans(
    features: torch.Tensor,
    n_clusters: int,
    *,
    iters: int = 12,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Cluster rows of ``features`` into ``n_clusters`` near-equal groups.

    ``features`` must be ``[n_neurons, n_tokens]``: one row per intermediate
    neuron, holding its activation profile across calibration tokens.
    """

    if features.numel() == 0:
        raise ValueError("no calibration activations collected")
    n = features.shape[0]
    k = min(n_clusters, n)
    x = features - features.mean(dim=0, keepdim=True)
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-6)

    generator = torch.Generator().manual_seed(seed)
    centroids = x[torch.randperm(n, generator=generator)[:k]].clone()
    base, extra = divmod(n, k)
    capacities = torch.tensor([base + (1 if i < extra else 0) for i in range(k)])

    labels = torch.zeros(n, dtype=torch.long)
    for _ in range(iters):
        distances = torch.cdist(x, centroids)
        labels = _balanced_assign(distances, capacities)
        for c in range(k):
            members = labels == c
            if members.any():
                centroids[c] = x[members].mean(dim=0)

    return [torch.where(labels == c)[0].sort().values for c in range(k)]


def clustering_stats(
    layer: str,
    activations: torch.Tensor,
    clusters: list[torch.Tensor],
    *,
    seed: int = 0,
) -> ClusterStats:
    activations = _neuron_profiles(activations)
    x = activations - activations.mean(dim=0, keepdim=True)
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-6)
    generator = torch.Generator().manual_seed(seed)

    def mean_cosine(assignment: list[torch.Tensor]) -> float:
        values = []
        for members in assignment:
            if members.numel() < 2:
                continue
            centroid = x[members].mean(dim=0)
            centroid = centroid / centroid.norm().clamp_min(1e-6)
            values.append(float((x[members] @ centroid).mean()))
        return sum(values) / len(values) if values else 0.0

    random_assignment = [
        chunk
        for chunk in torch.tensor_split(
            torch.randperm(x.shape[0], generator=generator), len(clusters)
        )
    ]
    sizes = torch.tensor([len(c) for c in clusters], dtype=torch.float32)
    balance = float(sizes.min() / sizes.max()) if sizes.numel() else 0.0
    return ClusterStats(
        layer=layer,
        n_neurons=int(x.shape[0]),
        n_clusters=len(clusters),
        cohesion=mean_cosine(clusters),
        baseline=mean_cosine(random_assignment),
        balance=balance,
    )


def contiguous_indices(intermediate: int, n_shards: int) -> list[torch.Tensor]:
    edges = [round(i * intermediate / n_shards) for i in range(n_shards + 1)]
    return [torch.arange(edges[i], edges[i + 1]) for i in range(n_shards)]


def _neuron_profiles(activations: torch.Tensor) -> torch.Tensor:
    """[tokens, intermediate] -> [intermediate, tokens]."""

    return activations.transpose(0, 1).contiguous()


def index_sets_for(
    activations: torch.Tensor | None,
    intermediate: int,
    n_shards: int,
    *,
    iters: int = 12,
    seed: int = 0,
) -> list[torch.Tensor]:
    if activations is None or activations.numel() == 0:
        return contiguous_indices(intermediate, n_shards)
    return balanced_kmeans(_neuron_profiles(activations), n_shards, iters=iters, seed=seed)
