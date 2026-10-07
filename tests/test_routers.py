"""Unit tests verifying shape consistency, gradient flow, and dual updates."""

import torch
from routers import make_router


def test_router_forward_shapes():
    batch_size, seq_len, latent_dim, num_experts, k = 4, 16, 64, 8, 2
    x = torch.randn(batch_size * seq_len, latent_dim)

    for variant in ["SLC", "DeepSeekBias"]:
        router = make_router(variant, latent_dim=latent_dim, num_experts=num_experts)
        gates, indices, expected_counts = router(x, k=k, return_aux=True)

        assert gates.shape == (batch_size * seq_len, k)
        assert indices.shape == (batch_size * seq_len, k)
        assert expected_counts.shape == (num_experts,)
        assert torch.all(indices >= 0) and torch.all(indices < num_experts)


def test_router_centroid_normalization():
    latent_dim, num_experts = 32, 4
    router = make_router("SLC", latent_dim=latent_dim, num_experts=num_experts)

    # Perform a forward pass and an update step to trigger unit-sphere projection
    x = torch.randn(32, latent_dim)
    gates, indices, expected_counts = router(x, k=2, return_aux=True)
    counts = torch.bincount(indices.view(-1), minlength=num_experts)

    router.update(
        counts=counts,
        expected_counts=expected_counts,
        n_tokens=32,
        top_k=2,
    )

    # Check that centroids are projected onto the unit sphere post-update
    norms_post = torch.norm(router.expert_centroids, dim=-1)
    assert torch.allclose(norms_post, torch.ones_like(norms_post), atol=1e-6)


def test_metrics_computation():
    latent_dim, num_experts = 32, 4
    router = make_router("DeepSeekBias", latent_dim=latent_dim, num_experts=num_experts)

    x = torch.randn(16, latent_dim)
    router(x, k=2, return_aux=True)
    m = router.metrics()

    assert "primal" in m
    assert "dual" in m
    assert "abs_gap" in m
    assert isinstance(m["primal"], float)
