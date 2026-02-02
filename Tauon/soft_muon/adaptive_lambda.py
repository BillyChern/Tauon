"""
Adaptive λ strategies for SoftMuon optimizer.

This module provides various strategies for setting the regularization
parameter λ, from fixed values to adaptive schemes based on gradient
statistics.
"""

import torch
from abc import ABC, abstractmethod
from typing import Dict, Optional
import math


class AdaptiveLambdaStrategy(ABC):
    """Base class for λ adaptation strategies."""

    @abstractmethod
    def get_lambda(
        self,
        grad: torch.Tensor,
        param: torch.Tensor,
        state: Dict,
        base_lambda: float,
    ) -> float:
        """
        Compute effective λ for this parameter.

        Args:
            grad: Current gradient
            param: Parameter tensor
            state: Optimizer state for this parameter
            base_lambda: Base λ from config

        Returns:
            Effective λ to use
        """
        pass

    def update_state(self, state: Dict, grad: torch.Tensor) -> None:
        """Update internal state after step (optional)."""
        pass


class FixedLambda(AdaptiveLambdaStrategy):
    """Fixed λ - always returns base_lambda."""

    def get_lambda(
        self,
        grad: torch.Tensor,
        param: torch.Tensor,
        state: Dict,
        base_lambda: float,
    ) -> float:
        return base_lambda


class ScheduledLambda(AdaptiveLambdaStrategy):
    """
    Scheduled λ decay over training.

    Starts high (more regularization for stability) and decays to allow
    more aggressive updates later in training.
    """

    def __init__(
        self,
        warmup_steps: int = 100,
        start_multiplier: float = 10.0,
        end_multiplier: float = 0.1,
        schedule: str = "cosine",
    ):
        """
        Args:
            warmup_steps: Number of steps for warmup phase
            start_multiplier: Multiply base_lambda by this at start
            end_multiplier: Multiply base_lambda by this at end
            schedule: 'cosine', 'linear', or 'exponential'
        """
        self.warmup_steps = warmup_steps
        self.start_multiplier = start_multiplier
        self.end_multiplier = end_multiplier
        self.schedule = schedule

    def get_lambda(
        self,
        grad: torch.Tensor,
        param: torch.Tensor,
        state: Dict,
        base_lambda: float,
    ) -> float:
        step = state.get("step", 0)

        if step < self.warmup_steps:
            # During warmup: decay from start_multiplier to 1.0
            progress = step / self.warmup_steps
            if self.schedule == "cosine":
                multiplier = self.start_multiplier + (1.0 - self.start_multiplier) * (
                    1 - math.cos(progress * math.pi)
                ) / 2
            elif self.schedule == "linear":
                multiplier = self.start_multiplier + (1.0 - self.start_multiplier) * progress
            else:  # exponential
                multiplier = self.start_multiplier * (1.0 / self.start_multiplier) ** progress
        else:
            # After warmup: decay from 1.0 to end_multiplier
            # Use a slower decay rate
            steps_after_warmup = step - self.warmup_steps
            decay_rate = 0.001  # Slow decay
            multiplier = 1.0 + (self.end_multiplier - 1.0) * (
                1 - math.exp(-decay_rate * steps_after_warmup)
            )

        return base_lambda * multiplier


class GradientAdaptiveLambda(AdaptiveLambdaStrategy):
    """
    Adaptive λ based on gradient spectral statistics.

    The idea: if the gradient is well-conditioned, use smaller λ (closer to Muon).
    If ill-conditioned, use larger λ for stability.
    """

    def __init__(
        self,
        min_lambda: float = 0.01,
        max_lambda: float = 10.0,
        ema_decay: float = 0.99,
        use_condition_proxy: bool = True,
    ):
        """
        Args:
            min_lambda: Minimum allowed λ
            max_lambda: Maximum allowed λ
            ema_decay: EMA decay for smoothing estimates
            use_condition_proxy: If True, use cheap proxy for condition number
        """
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.ema_decay = ema_decay
        self.use_condition_proxy = use_condition_proxy

    def get_lambda(
        self,
        grad: torch.Tensor,
        param: torch.Tensor,
        state: Dict,
        base_lambda: float,
    ) -> float:
        # Compute condition proxy
        if self.use_condition_proxy:
            condition_proxy = self._cheap_condition_proxy(grad)
        else:
            condition_proxy = self._svd_condition(grad)

        # EMA smoothing
        if "condition_ema" not in state:
            state["condition_ema"] = condition_proxy
        else:
            state["condition_ema"] = (
                self.ema_decay * state["condition_ema"]
                + (1 - self.ema_decay) * condition_proxy
            )

        # Map condition number to λ
        # Higher condition → higher λ (more regularization needed)
        # condition_proxy in [0, 1] where 1 = very ill-conditioned
        lambda_range = self.max_lambda - self.min_lambda
        effective_lambda = self.min_lambda + lambda_range * state["condition_ema"]

        return effective_lambda

    def _cheap_condition_proxy(self, grad: torch.Tensor) -> float:
        """
        Cheap proxy for condition number without full SVD.

        Uses ratio of trace to max diagonal element of G^T G.
        """
        if grad.dim() != 2:
            return 0.5  # Default for non-matrix

        # Compute G^T G diagonal cheaply
        GtG_diag = (grad * grad).sum(dim=0)  # Sum of squared columns
        trace = GtG_diag.sum()
        max_diag = GtG_diag.max()
        n = grad.shape[1]

        # Ratio: trace / (n * max_diag)
        # = 1 if all diag elements equal (well-conditioned)
        # < 1 if one dominates (ill-conditioned)
        ratio = trace / (n * max_diag + 1e-8)

        # Convert to ill-conditioning measure (1 - ratio)
        # Clamp to [0, 1]
        condition_proxy = torch.clamp(1.0 - ratio, 0.0, 1.0).item()

        return condition_proxy

    def _svd_condition(self, grad: torch.Tensor) -> float:
        """Exact condition number via SVD (expensive)."""
        if grad.dim() != 2:
            return 0.5

        S = torch.linalg.svdvals(grad)
        condition = (S[0] / (S[-1] + 1e-8)).item()

        # Normalize to [0, 1] range
        # condition = 1 → 0, condition ≥ 100 → 1
        normalized = min(1.0, math.log10(condition + 1) / 2)
        return normalized


class PerLayerLambda(AdaptiveLambdaStrategy):
    """
    Per-layer λ values, optionally learned or preset.

    Different layers may benefit from different regularization levels.
    """

    def __init__(
        self,
        layer_lambdas: Optional[Dict[str, float]] = None,
        default_lambda: float = 0.1,
        learn_lambdas: bool = False,
    ):
        """
        Args:
            layer_lambdas: Dict mapping param names to λ values
            default_lambda: Default λ for unlisted parameters
            learn_lambdas: If True, adapt lambdas based on training dynamics
        """
        self.layer_lambdas = layer_lambdas or {}
        self.default_lambda = default_lambda
        self.learn_lambdas = learn_lambdas
        self._param_to_name: Dict[int, str] = {}

    def register_param(self, param: torch.Tensor, name: str) -> None:
        """Register parameter name for lookup."""
        self._param_to_name[id(param)] = name

    def get_lambda(
        self,
        grad: torch.Tensor,
        param: torch.Tensor,
        state: Dict,
        base_lambda: float,
    ) -> float:
        param_id = id(param)
        name = self._param_to_name.get(param_id, "")

        # Check if we have a specific λ for this layer
        for key, lam in self.layer_lambdas.items():
            if key in name:
                return lam

        return self.default_lambda

    def update_state(self, state: Dict, grad: torch.Tensor) -> None:
        """Update λ based on training dynamics if learn_lambdas is True."""
        if not self.learn_lambdas:
            return

        # Track gradient norm variance
        grad_norm = torch.norm(grad).item()
        if "grad_norm_history" not in state:
            state["grad_norm_history"] = []

        state["grad_norm_history"].append(grad_norm)

        # Keep only recent history
        if len(state["grad_norm_history"]) > 100:
            state["grad_norm_history"] = state["grad_norm_history"][-100:]


class WarmupThenAdaptiveLambda(AdaptiveLambdaStrategy):
    """
    Combines warmup schedule with adaptive λ.

    During warmup: use scheduled high λ for stability.
    After warmup: switch to adaptive λ based on gradient statistics.
    """

    def __init__(
        self,
        warmup_steps: int = 100,
        warmup_lambda_multiplier: float = 10.0,
        min_lambda: float = 0.01,
        max_lambda: float = 10.0,
    ):
        self.warmup_steps = warmup_steps
        self.warmup_lambda_multiplier = warmup_lambda_multiplier
        self.scheduled = ScheduledLambda(
            warmup_steps=warmup_steps,
            start_multiplier=warmup_lambda_multiplier,
            end_multiplier=1.0,
        )
        self.adaptive = GradientAdaptiveLambda(
            min_lambda=min_lambda,
            max_lambda=max_lambda,
        )

    def get_lambda(
        self,
        grad: torch.Tensor,
        param: torch.Tensor,
        state: Dict,
        base_lambda: float,
    ) -> float:
        step = state.get("step", 0)

        if step < self.warmup_steps:
            return self.scheduled.get_lambda(grad, param, state, base_lambda)
        else:
            return self.adaptive.get_lambda(grad, param, state, base_lambda)
