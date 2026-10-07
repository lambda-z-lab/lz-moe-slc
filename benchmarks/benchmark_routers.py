"""Benchmark script reproducing stochastic load convergence under synthetic load skew."""

import time
import torch
from routers import make_router


def run_benchmark(variant: str, steps: int = 200) -> None:
    latent_dim = 128
    num_experts = 16
    k = 2
    batch_size = 32
    seq_len = 64
    n_tokens = batch_size * seq_len

    kwargs = {"latent_dim": latent_dim, "num_experts": num_experts}
    if variant == "SLC":
        kwargs["warmup_steps"] = 10

    router = make_router(variant, **kwargs)
    optimizer = torch.optim.Adam(router.parameters(), lr=1e-3)

    print(f"\n--- Benchmarking Router Variant: {variant} ---")
    start_time = time.time()

    for step in range(steps):
        # Inject artificial skew into token distribution to test control recovery
        x = torch.randn(n_tokens, latent_dim)
        if step < steps // 2:
            x[:, :4] += 2.0  # Skew toward first few experts initially

        optimizer.zero_grad()
        gates, indices, expected_counts = router(x, k=k, return_aux=True)

        # Dummy primal loss favoring gate magnitudes
        loss = -(gates * router.affinities.gather(1, indices)).mean()
        loss.backward()
        optimizer.step()

        counts = torch.bincount(indices.view(-1), minlength=num_experts).float()
        router.update(
            counts=counts,
            expected_counts=expected_counts,
            n_tokens=n_tokens,
            top_k=k,
        )

        if (step + 1) % 50 == 0:
            m = router.metrics()
            print(f"Step [{step+1}/{steps}] | Primal: {m['primal']:.4f} | Dual: {m['dual']:.4f} | Gap: {m['abs_gap']:.4f}")

    elapsed = time.time() - start_time
    print(f"Completed {steps} steps in {elapsed:.4f} seconds.")


if __name__ == "__main__":
    torch.manual_seed(42)
    run_benchmark("SLC", steps=150)
    run_benchmark("DeepSeekBias", steps=150)
