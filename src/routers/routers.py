"""MoE routers: Lagrangian and DeepSeek Bias variants.

Companion resource for: "Stochastic Lagrangian Control and Proportional
Bias Routing for Mixture-of-Experts".
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryRouter(nn.Module):
    """Shared base for geometry-based MoE routers."""

    variant_name: str = "base"

    def __init__(
        self,
        latent_dim: int,
        num_experts: int,
        *,
        t_sel: float = 0.1,
        t_ec: float = 0.1,
        load_error_ema_beta: float = 0.99,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.latent_dim = latent_dim
        self.t_sel = float(t_sel)
        self.t_ec = float(t_ec)
        self.load_error_ema_beta = float(load_error_ema_beta)

        self.expert_centroids = nn.Parameter(
            torch.randn(num_experts, latent_dim) * 0.05
        )
        self.register_buffer("load_error_ema", torch.zeros(num_experts))
        self.register_buffer("update_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("primal_ema", torch.zeros(()))
        self.register_buffer(
            "_primal_ema_ready", torch.zeros((), dtype=torch.bool)
        )

        self.affinities: torch.Tensor | None = None
        self.topk_indices: torch.Tensor | None = None
        self.expected_counts: torch.Tensor | None = None
        self.expected_probs: torch.Tensor | None = None

    def _logit_modifier(self) -> torch.Tensor:
        return torch.zeros(self.num_experts, device=self.expert_centroids.device)

    @torch.no_grad()
    def _do_update(
        self,
        *,
        counts: torch.Tensor,
        expected_counts: torch.Tensor,
        n_tokens: int,
        top_k: int,
    ) -> None:
        pass

    def _gate_from_affinities(self, sel_aff: torch.Tensor, k: int) -> torch.Tensor:
        return torch.softmax(sel_aff.to(torch.float32), dim=-1).to(sel_aff.dtype)

    @torch.no_grad()
    def _dual_value(self, primal_ema: torch.Tensor) -> torch.Tensor:
        return primal_ema.clone()

    def forward(
        self,
        mla_latent: torch.Tensor,
        k: int = 2,
        return_aux: bool = False,
    ):
        aff = torch.matmul(mla_latent, self.expert_centroids.t())
        aff = aff / (mla_latent.size(-1) ** 0.5)
        self.affinities = aff

        biased = aff + self._logit_modifier()
        sel_logits = biased / self.t_sel
        topk_indices = torch.topk(sel_logits, k=k, dim=-1).indices
        sel_aff = aff.gather(1, topk_indices)
        gates = self._gate_from_affinities(sel_aff, k).to(mla_latent.dtype)

        self.topk_indices = topk_indices

        if not return_aux:
            return gates, topk_indices

        with torch.no_grad():
            p_proxy = torch.softmax(sel_logits.to(torch.float32) / self.t_ec, dim=-1)
            self.expected_counts = p_proxy.sum(dim=0) * float(k)
            self.expected_probs = p_proxy

        return gates, topk_indices, self.expected_counts

    @torch.no_grad()
    def update(
        self,
        *,
        counts: torch.Tensor,
        expected_counts: torch.Tensor,
        n_tokens: int,
        top_k: int,
    ) -> None:
        self.update_count.add_(1)
        target = float(n_tokens) * float(top_k) / self.num_experts
        raw_violation = (expected_counts - target) / (target + 1e-6)
        self.load_error_ema.lerp_(raw_violation, 1.0 - self.load_error_ema_beta)

        self._do_update(
            counts=counts,
            expected_counts=expected_counts,
            n_tokens=n_tokens,
            top_k=top_k,
        )

        self.expert_centroids.data.copy_(
            F.normalize(self.expert_centroids.data, dim=-1)
        )

    @torch.no_grad()
    def _primal_value(self) -> torch.Tensor:
        chosen = torch.gather(self.affinities, 1, self.topk_indices)
        return chosen.sum() / self.affinities.size(0)

    @torch.no_grad()
    def metrics(self) -> dict[str, float]:
        primal = self._primal_value()
        if not bool(self._primal_ema_ready):
            self.primal_ema.copy_(primal)
            self._primal_ema_ready.fill_(True)
        else:
            self.primal_ema.lerp_(primal.to(torch.float32), 1.0 - self.load_error_ema_beta)
        dual = self._dual_value(self.primal_ema)
        abs_gap = torch.abs(self.primal_ema - dual)
        rel = abs_gap / (torch.abs(self.primal_ema) + 1e-8)
        return {
            "primal": self.primal_ema.item(),
            "dual": dual.item(),
            "abs_gap": abs_gap.item(),
            "relative_gap_percent": rel.item() * 100,
        }


class LagrangianRouter(GeometryRouter):
    """Stochastic Augmented Lagrangian Control (SLC) Router."""

    variant_name = "SLC"

    def __init__(
        self,
        latent_dim: int,
        num_experts: int,
        *,
        learning_rate_dual: float = 0.3,
        momentum_dual: float = 0.98,
        rho: float = 0.15,
        rho_max: float = 1.0,
        rho_growth: float = 1.0,
        warmup_steps: int = 1000,
        clamp_abs: float = 10.0,
        slack: float = 0.005,
        capacity_factor: float = 1.0,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(latent_dim, num_experts, **base_kwargs)
        self.target_lr_dual = float(learning_rate_dual)
        self.lr_dual = 0.0
        self.momentum = float(momentum_dual)
        self.warmup_steps = int(warmup_steps)
        self.clamp_abs = float(clamp_abs)
        self.slack = float(slack)
        self.capacity_factor = float(capacity_factor)

        self.register_buffer("v_dual", torch.zeros(num_experts))
        self.register_buffer("v_velocity", torch.zeros(num_experts))
        self.register_buffer("rho", torch.tensor(float(rho)))
        self.rho_max = float(rho_max)
        self.rho_growth = float(rho_growth)

    def _logit_modifier(self) -> torch.Tensor:
        return -(self.v_dual + self.rho * self.load_error_ema)

    def _warmup_lr(self) -> None:
        if self.update_count < self.warmup_steps:
            progress = self.update_count.float() / self.warmup_steps
            self.lr_dual = progress.item() * self.target_lr_dual
        else:
            self.lr_dual = self.target_lr_dual

    @torch.no_grad()
    def _do_update(
        self,
        *,
        counts: torch.Tensor,
        expected_counts: torch.Tensor,
        n_tokens: int,
        top_k: int,
    ) -> None:
        self._warmup_lr()
        if self.lr_dual == 0:
            return

        target = (
            float(n_tokens) * float(top_k) / self.num_experts
        ) * self.capacity_factor
        raw = (expected_counts - target) / (target + 1e-6)

        effective = torch.where(
            raw.abs() < self.slack,
            torch.zeros_like(raw),
            raw - self.slack * raw.sign(),
        )

        if self.momentum > 0:
            self.v_velocity.lerp_(effective, 1.0 - self.momentum)
            grad = self.v_velocity
        else:
            grad = effective

        self.v_dual.add_(grad, alpha=self.lr_dual)
        self.v_dual.sub_(self.v_dual.mean())
        if self.clamp_abs is not None:
            self.v_dual.clamp_(-self.clamp_abs, self.clamp_abs)

    @torch.no_grad()
    def _dual_value(self, primal_ema: torch.Tensor) -> torch.Tensor:
        return primal_ema - torch.dot(self.v_dual, self.load_error_ema)


class DeepSeekBiasRouter(GeometryRouter):
    """DeepSeek Proportional Bias Router."""

    variant_name = "DeepSeekBias"

    def __init__(
        self,
        latent_dim: int,
        num_experts: int,
        *,
        bias_update_gamma: float = 0.001,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(latent_dim, num_experts, **base_kwargs)
        self.register_buffer("router_bias", torch.zeros(num_experts))
        self.register_buffer(
            "bias_update_gamma", torch.tensor(float(bias_update_gamma))
        )

    def _logit_modifier(self) -> torch.Tensor:
        return self.router_bias

    def _gate_from_affinities(self, sel_aff: torch.Tensor, k: int) -> torch.Tensor:
        denom = sel_aff.sum(dim=1, keepdim=True)
        return torch.where(
            denom > 1e-9,
            sel_aff / (denom + 1e-9),
            torch.full_like(sel_aff, 1.0 / k),
        )

    @torch.no_grad()
    def _do_update(
        self,
        *,
        counts: torch.Tensor,
        expected_counts: torch.Tensor,
        n_tokens: int,
        top_k: int,
    ) -> None:
        avg = counts.sum() / self.num_experts
        violation = (counts - avg) / (avg + 1e-6)
        self.router_bias.sub_(self.bias_update_gamma * violation)

    @torch.no_grad()
    def _dual_value(self, primal_ema: torch.Tensor) -> torch.Tensor:
        return primal_ema - torch.dot(-self.router_bias, self.load_error_ema)


ROUTER_REGISTRY: dict[str, type[GeometryRouter]] = {
    LagrangianRouter.variant_name: LagrangianRouter,
    DeepSeekBiasRouter.variant_name: DeepSeekBiasRouter,
}


def make_router(variant: str, **kwargs: Any) -> GeometryRouter:
    if variant not in ROUTER_REGISTRY:
        raise ValueError(f"Unknown router variant {variant!r}. Registered: {list(ROUTER_REGISTRY)}")
    return ROUTER_REGISTRY[variant](**kwargs)