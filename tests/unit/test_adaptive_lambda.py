"""
Unit tests for adaptive λ strategies.
"""

import pytest
import torch
from soft_muon.adaptive_lambda import (
    FixedLambda,
    ScheduledLambda,
    GradientAdaptiveLambda,
    PerLayerLambda,
    WarmupThenAdaptiveLambda,
)


class TestFixedLambda:
    """Test fixed λ strategy."""

    def test_returns_base_lambda(self):
        """Should always return base λ."""
        strategy = FixedLambda()
        grad = torch.randn(10, 5)
        param = torch.randn(10, 5)
        state = {"step": 100}
        base_lambda = 0.5

        result = strategy.get_lambda(grad, param, state, base_lambda)
        assert result == base_lambda

    def test_unchanged_over_steps(self):
        """Should not change with step count."""
        strategy = FixedLambda()
        grad = torch.randn(10, 5)
        param = torch.randn(10, 5)
        base_lambda = 0.1

        for step in [0, 10, 100, 1000]:
            state = {"step": step}
            result = strategy.get_lambda(grad, param, state, base_lambda)
            assert result == base_lambda


class TestScheduledLambda:
    """Test scheduled λ strategy."""

    def test_warmup_decay(self):
        """λ should decay during warmup."""
        strategy = ScheduledLambda(
            warmup_steps=100,
            start_multiplier=10.0,
            end_multiplier=0.1,
            schedule="linear",
        )
        grad = torch.randn(10, 5)
        param = torch.randn(10, 5)
        base_lambda = 0.1

        lambdas = []
        for step in [0, 25, 50, 75, 100]:
            state = {"step": step}
            lam = strategy.get_lambda(grad, param, state, base_lambda)
            lambdas.append(lam)

        # Should decrease during warmup
        for i in range(1, len(lambdas)):
            assert lambdas[i] <= lambdas[i - 1]

    def test_start_multiplier(self):
        """Should start at base_lambda * start_multiplier."""
        strategy = ScheduledLambda(
            warmup_steps=100,
            start_multiplier=10.0,
        )
        grad = torch.randn(10, 5)
        param = torch.randn(10, 5)
        base_lambda = 0.1
        state = {"step": 0}

        lam = strategy.get_lambda(grad, param, state, base_lambda)
        assert abs(lam - base_lambda * 10.0) < 1e-5

    def test_cosine_schedule(self):
        """Test cosine schedule."""
        strategy = ScheduledLambda(
            warmup_steps=100,
            start_multiplier=10.0,
            schedule="cosine",
        )
        grad = torch.randn(10, 5)
        param = torch.randn(10, 5)
        base_lambda = 0.1

        lambdas = []
        for step in range(0, 101, 10):
            state = {"step": step}
            lam = strategy.get_lambda(grad, param, state, base_lambda)
            lambdas.append(lam)

        # Cosine should be smooth
        assert len(lambdas) == 11


class TestGradientAdaptiveLambda:
    """Test gradient-adaptive λ strategy."""

    def test_well_conditioned_gives_low_lambda(self):
        """Well-conditioned gradient should result in lower λ."""
        strategy = GradientAdaptiveLambda(min_lambda=0.01, max_lambda=10.0)

        # Well-conditioned: all columns have similar norm
        grad = torch.randn(10, 5)
        grad = grad / grad.norm(dim=0, keepdim=True)  # Normalize columns

        param = torch.randn(10, 5)
        state = {}
        base_lambda = 1.0

        lam = strategy.get_lambda(grad, param, state, base_lambda)

        # Should be toward the lower end
        assert lam < 5.0  # Less than midpoint

    def test_ill_conditioned_gives_high_lambda(self):
        """Ill-conditioned gradient should result in higher λ."""
        strategy = GradientAdaptiveLambda(min_lambda=0.01, max_lambda=10.0)

        # Ill-conditioned: one column dominates
        grad = torch.zeros(10, 5)
        grad[:, 0] = 100.0  # One column very large

        param = torch.randn(10, 5)
        state = {}
        base_lambda = 1.0

        lam = strategy.get_lambda(grad, param, state, base_lambda)

        # Should be toward the higher end
        assert lam > 5.0  # More than midpoint

    def test_ema_smoothing(self):
        """EMA should smooth the λ estimates."""
        strategy = GradientAdaptiveLambda(min_lambda=0.01, max_lambda=10.0, ema_decay=0.9)

        param = torch.randn(10, 5)
        base_lambda = 1.0
        state = {}

        lambdas = []
        for _ in range(10):
            # Alternating gradients
            grad = torch.randn(10, 5)
            lam = strategy.get_lambda(grad, param, state, base_lambda)
            lambdas.append(lam)

        # With EMA, should be smoothed (not jumping around too much)
        diffs = [abs(lambdas[i + 1] - lambdas[i]) for i in range(len(lambdas) - 1)]
        avg_diff = sum(diffs) / len(diffs)

        # Average change should be bounded
        assert avg_diff < 5.0


class TestPerLayerLambda:
    """Test per-layer λ strategy."""

    def test_specific_layer_lambda(self):
        """Specific layers should use their assigned λ."""
        layer_lambdas = {"conv1": 0.01, "conv2": 0.1, "fc": 1.0}
        strategy = PerLayerLambda(layer_lambdas=layer_lambdas, default_lambda=0.5)

        grad = torch.randn(10, 5)
        state = {}
        base_lambda = 1.0

        # Register params
        param_conv1 = torch.randn(10, 5)
        param_fc = torch.randn(10, 5)
        param_other = torch.randn(10, 5)

        strategy.register_param(param_conv1, "model.conv1.weight")
        strategy.register_param(param_fc, "model.fc.weight")
        strategy.register_param(param_other, "model.other.weight")

        assert strategy.get_lambda(grad, param_conv1, state, base_lambda) == 0.01
        assert strategy.get_lambda(grad, param_fc, state, base_lambda) == 1.0
        assert strategy.get_lambda(grad, param_other, state, base_lambda) == 0.5

    def test_default_lambda(self):
        """Unspecified layers should use default."""
        strategy = PerLayerLambda(layer_lambdas={}, default_lambda=0.3)

        grad = torch.randn(10, 5)
        param = torch.randn(10, 5)
        state = {}
        base_lambda = 1.0

        lam = strategy.get_lambda(grad, param, state, base_lambda)
        assert lam == 0.3


class TestWarmupThenAdaptive:
    """Test warmup + adaptive λ strategy."""

    def test_uses_scheduled_during_warmup(self):
        """Should use scheduled λ during warmup."""
        strategy = WarmupThenAdaptiveLambda(
            warmup_steps=100,
            warmup_lambda_multiplier=10.0,
        )

        grad = torch.randn(10, 5)
        param = torch.randn(10, 5)
        base_lambda = 0.1

        # At step 0, should be high (warmup)
        state = {"step": 0}
        lam_start = strategy.get_lambda(grad, param, state, base_lambda)

        # Should be around base_lambda * 10
        assert lam_start > base_lambda * 5

    def test_uses_adaptive_after_warmup(self):
        """Should use adaptive λ after warmup."""
        strategy = WarmupThenAdaptiveLambda(warmup_steps=100)

        grad = torch.randn(10, 5)
        param = torch.randn(10, 5)
        base_lambda = 0.1

        # After warmup
        state = {"step": 150}
        lam = strategy.get_lambda(grad, param, state, base_lambda)

        # Should be in adaptive range
        assert strategy.adaptive.min_lambda <= lam <= strategy.adaptive.max_lambda
