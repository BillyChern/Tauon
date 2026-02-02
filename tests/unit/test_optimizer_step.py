"""
Unit tests for SoftMuon optimizer step behavior.
"""

import pytest
import torch
import torch.nn as nn
from soft_muon import SoftMuon, SoftMuonConfig
from soft_muon.optimizer import create_optimizer_groups, CombinedOptimizer


class TestSoftMuonBasics:
    """Test basic optimizer functionality."""

    def test_initialization(self):
        """Test optimizer initializes correctly."""
        model = nn.Linear(10, 5)
        config = SoftMuonConfig(lr=0.01, lambda_reg=0.1)
        optimizer = SoftMuon(model.parameters(), config)

        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]["lr"] == 0.01
        assert optimizer.param_groups[0]["lambda_reg"] == 0.1

    def test_step_updates_params(self):
        """Test that step updates parameters."""
        torch.manual_seed(42)
        model = nn.Linear(10, 5, bias=False)
        initial_weight = model.weight.clone()

        config = SoftMuonConfig(lr=0.01, lambda_reg=0.1)
        optimizer = SoftMuon(model.parameters(), config)

        # Forward and backward
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()

        optimizer.step()

        # Weight should have changed
        assert not torch.allclose(model.weight, initial_weight)

    def test_zero_grad(self):
        """Test zero_grad clears gradients."""
        model = nn.Linear(10, 5)
        config = SoftMuonConfig()
        optimizer = SoftMuon(model.parameters(), config)

        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()

        assert model.weight.grad is not None

        optimizer.zero_grad()

        assert model.weight.grad is None or model.weight.grad.abs().sum() == 0

    def test_no_grad_params_unchanged(self):
        """Parameters without gradients should not change."""
        model = nn.Linear(10, 5, bias=False)
        config = SoftMuonConfig()
        optimizer = SoftMuon(model.parameters(), config)

        initial_weight = model.weight.clone()

        # Step without backward (no gradients)
        optimizer.step()

        assert torch.allclose(model.weight, initial_weight)

    def test_closure_support(self):
        """Test that closure is supported."""
        model = nn.Linear(10, 5)
        config = SoftMuonConfig()
        optimizer = SoftMuon(model.parameters(), config)

        x = torch.randn(4, 10)

        def closure():
            optimizer.zero_grad()
            loss = model(x).sum()
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        assert loss is not None


class TestMomentum:
    """Test momentum behavior."""

    def test_momentum_accumulates(self):
        """Test that momentum accumulates across steps."""
        torch.manual_seed(42)
        model = nn.Linear(10, 5, bias=False)

        config = SoftMuonConfig(lr=0.01, momentum=0.9, lambda_reg=0.1)
        optimizer = SoftMuon(model.parameters(), config)

        x = torch.randn(4, 10)

        # Multiple steps with same input
        weight_changes = []
        for _ in range(5):
            initial = model.weight.clone()
            optimizer.zero_grad()
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            weight_changes.append((model.weight - initial).norm().item())

        # With momentum, later updates should be larger (momentum builds up)
        # Actually depends on Nesterov vs standard, but changes should differ
        assert not all(abs(w - weight_changes[0]) < 1e-10 for w in weight_changes)

    def test_nesterov_vs_standard(self):
        """Test difference between Nesterov and standard momentum."""
        torch.manual_seed(42)

        model_nesterov = nn.Linear(10, 5, bias=False)
        model_standard = nn.Linear(10, 5, bias=False)

        # Same initialization
        model_standard.weight.data = model_nesterov.weight.data.clone()

        config_nesterov = SoftMuonConfig(lr=0.01, lambda_reg=0.1, use_nesterov=True)
        config_standard = SoftMuonConfig(lr=0.01, lambda_reg=0.1, use_nesterov=False)

        opt_nesterov = SoftMuon(model_nesterov.parameters(), config_nesterov)
        opt_standard = SoftMuon(model_standard.parameters(), config_standard)

        # Use different random inputs each step to accumulate differences
        for i in range(10):
            x = torch.randn(4, 10)
            target = torch.randn(4, 5)

            opt_nesterov.zero_grad()
            opt_standard.zero_grad()

            loss_n = ((model_nesterov(x) - target) ** 2).sum()
            loss_s = ((model_standard(x) - target) ** 2).sum()

            loss_n.backward()
            loss_s.backward()

            opt_nesterov.step()
            opt_standard.step()

        # After multiple steps, weights should differ due to Nesterov lookahead
        # Use a tolerance to check they're not exactly equal
        diff = (model_nesterov.weight - model_standard.weight).abs().max()
        assert diff > 1e-6, f"Expected difference, got max diff {diff}"


class TestLambdaModes:
    """Test different λ modes."""

    def test_fixed_lambda(self):
        """Test fixed λ mode."""
        model = nn.Linear(10, 5, bias=False)
        config = SoftMuonConfig(lambda_reg=0.5, lambda_mode="fixed")
        optimizer = SoftMuon(model.parameters(), config)

        x = torch.randn(4, 10)
        for _ in range(5):
            optimizer.zero_grad()
            model(x).sum().backward()
            optimizer.step()

        # Just verify no errors occur
        assert True

    def test_scheduled_lambda(self):
        """Test scheduled λ mode."""
        model = nn.Linear(10, 5, bias=False)
        config = SoftMuonConfig(
            lambda_reg=0.1,
            lambda_mode="scheduled",
            lambda_warmup_steps=10,
        )
        optimizer = SoftMuon(model.parameters(), config)

        x = torch.randn(4, 10)
        for _ in range(20):
            optimizer.zero_grad()
            model(x).sum().backward()
            optimizer.step()

        assert True

    def test_adaptive_lambda(self):
        """Test adaptive λ mode."""
        model = nn.Linear(10, 5, bias=False)
        config = SoftMuonConfig(lambda_reg=0.1, lambda_mode="adaptive")
        optimizer = SoftMuon(model.parameters(), config)

        x = torch.randn(4, 10)
        for _ in range(10):
            optimizer.zero_grad()
            model(x).sum().backward()
            optimizer.step()

        assert True


class TestWeightDecay:
    """Test weight decay behavior."""

    def test_weight_decay_shrinks_weights(self):
        """Weight decay should shrink weights."""
        torch.manual_seed(42)

        model_decay = nn.Linear(10, 5, bias=False)
        model_no_decay = nn.Linear(10, 5, bias=False)
        model_no_decay.weight.data = model_decay.weight.data.clone()

        config_decay = SoftMuonConfig(lr=0.01, weight_decay=0.1)
        config_no_decay = SoftMuonConfig(lr=0.01, weight_decay=0.0)

        opt_decay = SoftMuon(model_decay.parameters(), config_decay)
        opt_no_decay = SoftMuon(model_no_decay.parameters(), config_no_decay)

        x = torch.randn(4, 10)

        for _ in range(10):
            opt_decay.zero_grad()
            opt_no_decay.zero_grad()

            model_decay(x).sum().backward()
            model_no_decay(x).sum().backward()

            opt_decay.step()
            opt_no_decay.step()

        # Decayed model should have smaller weight norm
        assert model_decay.weight.norm() < model_no_decay.weight.norm()


class TestExclusions:
    """Test parameter exclusion behavior."""

    def test_1d_params_not_orthogonalized(self):
        """1D parameters (biases) should not have polar applied."""
        model = nn.Linear(10, 5)  # Has bias
        config = SoftMuonConfig()
        optimizer = SoftMuon(model.parameters(), config)

        # Bias is 1D, should not use regularized polar
        assert model.bias.dim() == 1
        # This test mainly verifies no error occurs
        x = torch.randn(4, 10)
        optimizer.zero_grad()
        model(x).sum().backward()
        optimizer.step()

    def test_excluded_names(self):
        """Parameters with excluded names should not use polar."""

        class ModelWithEmbed(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 32)
                self.linear = nn.Linear(32, 10)

            def forward(self, x):
                return self.linear(self.embed(x))

        model = ModelWithEmbed()
        config = SoftMuonConfig(exclude_names=("embed",))
        optimizer = SoftMuon(model.parameters(), config)
        optimizer.register_param_names(model.named_parameters())

        # Check that embed is excluded
        for p in model.parameters():
            name = optimizer._get_param_name(p)
            if "embed" in name:
                assert not optimizer._should_orthogonalize(p, optimizer.param_groups[0])


class TestCombinedOptimizer:
    """Test combined optimizer utility."""

    def test_creates_both_optimizers(self):
        """Test that both SoftMuon and AdamW are created."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                self.ln = nn.LayerNorm(5)

            def forward(self, x):
                return self.ln(self.linear(x))

        model = SimpleModel()
        config = SoftMuonConfig(exclude_names=("ln",))

        combined = CombinedOptimizer(model, config)

        assert combined.soft_opt is not None
        assert combined.adamw_opt is not None

    def test_combined_step(self):
        """Test combined optimizer step."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5, bias=False)  # No bias to ensure gradient flows
                self.ln = nn.LayerNorm(5)

            def forward(self, x):
                return self.ln(self.fc(x))

        torch.manual_seed(42)
        model = SimpleModel()
        config = SoftMuonConfig(lr=0.02, lambda_reg=0.1)

        combined = CombinedOptimizer(model, config)

        initial_fc = model.fc.weight.clone()
        initial_ln = model.ln.weight.clone()

        # Use multiple steps to ensure optimization happens
        for _ in range(3):
            x = torch.randn(4, 10)
            target = torch.randn(4, 5)
            combined.zero_grad()
            output = model(x)
            loss = ((output - target) ** 2).sum()
            loss.backward()
            combined.step()

        # Both should have changed (fc via SoftMuon, ln via AdamW)
        assert not torch.allclose(model.fc.weight, initial_fc)
        assert not torch.allclose(model.ln.weight, initial_ln)


class TestNumericalStability:
    """Test numerical stability."""

    def test_no_nan_inf(self):
        """Optimizer should not produce NaN or Inf."""
        torch.manual_seed(42)
        model = nn.Linear(100, 50, bias=False)
        config = SoftMuonConfig(lr=0.02, lambda_reg=0.1)  # Reasonable LR
        optimizer = SoftMuon(model.parameters(), config)

        x = torch.randn(32, 100)

        for _ in range(50):
            optimizer.zero_grad()
            loss = model(x).sum()
            loss.backward()
            optimizer.step()

            assert not torch.isnan(model.weight).any()
            assert not torch.isinf(model.weight).any()

    def test_large_gradients(self):
        """Should handle large gradients without exploding."""
        torch.manual_seed(42)
        model = nn.Linear(20, 10, bias=False)
        config = SoftMuonConfig(lr=0.01, lambda_reg=0.1)
        optimizer = SoftMuon(model.parameters(), config)

        # Create artificially large gradient
        model.weight.grad = torch.randn_like(model.weight) * 1000

        optimizer.step()

        assert not torch.isnan(model.weight).any()
        assert not torch.isinf(model.weight).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_training(self):
        """Test on GPU."""
        model = nn.Linear(100, 50, bias=False).cuda()
        config = SoftMuonConfig()
        optimizer = SoftMuon(model.parameters(), config)

        x = torch.randn(32, 100).cuda()

        for _ in range(10):
            optimizer.zero_grad()
            model(x).sum().backward()
            optimizer.step()

        assert model.weight.device.type == "cuda"
        assert not torch.isnan(model.weight).any()
