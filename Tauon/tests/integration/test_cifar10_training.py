"""
Integration tests for CIFAR-10 training.
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from soft_muon import SoftMuon, SoftMuonConfig
from soft_muon.optimizer import CombinedOptimizer


class SimpleCNN(nn.Module):
    """Simple CNN for testing."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


class TestCIFAR10Integration:
    """Integration tests with CIFAR-10-like data."""

    @pytest.fixture
    def model(self):
        return SimpleCNN()

    @pytest.fixture
    def fake_data(self):
        """Generate fake CIFAR-10-like data."""
        # 100 samples of 32x32 RGB images
        images = torch.randn(100, 3, 32, 32)
        labels = torch.randint(0, 10, (100,))
        return images, labels

    def test_softmuon_training_converges(self, model, fake_data):
        """SoftMuon should decrease loss over training."""
        images, labels = fake_data
        config = SoftMuonConfig(lr=0.01, lambda_reg=0.1)
        optimizer = SoftMuon(model.parameters(), config)

        initial_loss = None
        for epoch in range(5):
            optimizer.zero_grad()
            output = model(images)
            loss = nn.functional.cross_entropy(output, labels)
            loss.backward()
            optimizer.step()

            if initial_loss is None:
                initial_loss = loss.item()

        final_loss = loss.item()
        assert final_loss < initial_loss, "Loss should decrease"

    def test_combined_optimizer_converges(self, model, fake_data):
        """CombinedOptimizer should decrease loss."""
        images, labels = fake_data
        config = SoftMuonConfig(lr=0.01, lambda_reg=0.1)
        optimizer = CombinedOptimizer(model, config, adamw_lr=1e-3)

        initial_loss = None
        for epoch in range(5):
            optimizer.zero_grad()
            output = model(images)
            loss = nn.functional.cross_entropy(output, labels)
            loss.backward()
            optimizer.step()

            if initial_loss is None:
                initial_loss = loss.item()

        final_loss = loss.item()
        assert final_loss < initial_loss

    def test_training_accuracy_improves(self, model, fake_data):
        """Training accuracy should improve."""
        images, labels = fake_data
        config = SoftMuonConfig(lr=0.01, lambda_reg=0.1)
        optimizer = CombinedOptimizer(model, config)

        # Initial accuracy
        model.eval()
        with torch.no_grad():
            pred = model(images).argmax(dim=1)
            initial_acc = (pred == labels).float().mean().item()
        model.train()

        # Train
        for _ in range(20):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()

        # Final accuracy
        model.eval()
        with torch.no_grad():
            pred = model(images).argmax(dim=1)
            final_acc = (pred == labels).float().mean().item()

        assert final_acc > initial_acc

    @pytest.mark.parametrize("lambda_reg", [0.01, 0.1, 1.0])
    def test_different_lambdas_train(self, model, fake_data, lambda_reg):
        """Different λ values should all train without errors."""
        images, labels = fake_data
        config = SoftMuonConfig(lr=0.01, lambda_reg=lambda_reg)
        optimizer = SoftMuon(model.parameters(), config)

        for _ in range(5):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()

            assert not torch.isnan(loss)


class TestNumericalStability:
    """Test numerical stability in various conditions."""

    def test_no_nan_after_many_steps(self):
        """Should not produce NaN after many steps."""
        torch.manual_seed(42)
        model = SimpleCNN()
        config = SoftMuonConfig(lr=0.02, lambda_reg=0.1)
        optimizer = CombinedOptimizer(model, config)

        for step in range(100):
            images = torch.randn(16, 3, 32, 32)
            labels = torch.randint(0, 10, (16,))

            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()

            # Check for NaN
            for p in model.parameters():
                assert not torch.isnan(p).any(), f"NaN at step {step}"
                assert not torch.isinf(p).any(), f"Inf at step {step}"

    def test_gradient_clipping_compatibility(self):
        """Should work with gradient clipping."""
        model = SimpleCNN()
        config = SoftMuonConfig(lr=0.02, lambda_reg=0.1)
        optimizer = CombinedOptimizer(model, config)

        images = torch.randn(16, 3, 32, 32)
        labels = torch.randint(0, 10, (16,))

        for _ in range(10):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(images), labels)
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        # Should complete without error

    def test_zero_gradients_handled(self):
        """Should handle zero gradients gracefully."""
        model = nn.Linear(10, 5, bias=False)
        config = SoftMuonConfig(lr=0.01, lambda_reg=0.1)
        optimizer = SoftMuon(model.parameters(), config)

        # Set gradient to zero
        model.weight.grad = torch.zeros_like(model.weight)

        # Should not crash
        optimizer.step()


class TestGradientCorrectness:
    """Test that gradients are used correctly."""

    def test_gradients_flow_through(self):
        """Gradients should be used, not ignored."""
        torch.manual_seed(42)
        model = nn.Linear(10, 5, bias=False)
        config = SoftMuonConfig(lr=0.1, lambda_reg=0.1)
        optimizer = SoftMuon(model.parameters(), config)

        x = torch.randn(4, 10, requires_grad=False)
        target = torch.randn(4, 5)

        initial_weight = model.weight.clone()

        optimizer.zero_grad()
        loss = ((model(x) - target) ** 2).sum()
        loss.backward()
        optimizer.step()

        # Weights should have changed
        assert not torch.allclose(model.weight, initial_weight)

        # Change should be in direction of negative gradient (roughly)
        # Not exact due to regularized polar transformation

    def test_different_gradients_different_updates(self):
        """Different gradients should produce different updates."""
        torch.manual_seed(42)

        # Two identical models
        model1 = nn.Linear(10, 5, bias=False)
        model2 = nn.Linear(10, 5, bias=False)
        model2.weight.data = model1.weight.data.clone()

        config = SoftMuonConfig(lr=0.1, lambda_reg=0.1)
        opt1 = SoftMuon(model1.parameters(), config)
        opt2 = SoftMuon(model2.parameters(), config)

        # Different inputs
        x1 = torch.randn(4, 10)
        x2 = torch.randn(4, 10)  # Different!

        target = torch.randn(4, 5)

        opt1.zero_grad()
        ((model1(x1) - target) ** 2).sum().backward()
        opt1.step()

        opt2.zero_grad()
        ((model2(x2) - target) ** 2).sum().backward()
        opt2.step()

        # Updates should be different
        assert not torch.allclose(model1.weight, model2.weight)
