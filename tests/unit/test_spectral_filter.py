"""
Unit tests for spectral filter verification and analysis utilities.
"""

import pytest
import torch
from soft_muon.newton_schulz import regularized_polar
from soft_muon.spectral_utils import (
    compute_spectral_stats,
    verify_spectral_filter,
    spectral_filter_values,
    compare_filters,
    SpectralStats,
    SpectralTracker,
    analyze_filter_effect,
)


class TestSpectralFilterFormula:
    """Test that the spectral filter φ_λ(σ) = σ/√(σ²+λ) is correctly applied."""

    def test_filter_formula_basic(self):
        """Verify φ_λ(σ) = σ / √(σ² + λ)."""
        lambda_reg = 0.1
        sigma = torch.tensor([0.1, 0.5, 1.0, 2.0, 10.0], dtype=torch.float64)
        expected = sigma / torch.sqrt(sigma**2 + lambda_reg)
        actual = spectral_filter_values(sigma, lambda_reg)
        assert torch.allclose(actual, expected, atol=1e-10)

    def test_filter_bounds(self):
        """Filter values should be in (0, 1) for positive σ and λ."""
        lambda_reg = 0.1
        sigma = torch.tensor([0.01, 0.1, 1.0, 10.0, 100.0], dtype=torch.float64)
        filtered = spectral_filter_values(sigma, lambda_reg)

        assert (filtered > 0).all()
        assert (filtered < 1).all()

    def test_filter_monotonic(self):
        """Filter should be monotonically increasing in σ."""
        lambda_reg = 0.1
        sigma = torch.linspace(0.01, 10.0, 100, dtype=torch.float64)
        filtered = spectral_filter_values(sigma, lambda_reg)

        diffs = filtered[1:] - filtered[:-1]
        assert (diffs > 0).all()

    def test_lambda_zero_limit(self):
        """λ → 0 should give φ(σ) → 1 for σ > 0."""
        sigma = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        lambda_reg = 1e-10
        filtered = spectral_filter_values(sigma, lambda_reg)
        assert torch.allclose(filtered, torch.ones_like(sigma), atol=1e-4)

    def test_lambda_large_limit(self):
        """λ → ∞ should give φ(σ) → 0."""
        sigma = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        lambda_reg = 1e10
        filtered = spectral_filter_values(sigma, lambda_reg)
        assert torch.allclose(filtered, torch.zeros_like(sigma), atol=1e-4)


class TestVerifySpectralFilter:
    """Test the verification function."""

    def test_correct_filter_passes(self):
        """Correctly computed regularized polar should approximately pass verification."""
        torch.manual_seed(42)
        C = torch.randn(15, 10, dtype=torch.float64)
        lambda_reg = 0.1
        Q = regularized_polar(C, lambda_reg, n_iters=15)

        result = verify_spectral_filter(C, Q, lambda_reg, tol=0.1)  # Larger tolerance
        assert result["passed"], f"Max error: {result['max_error']}"

    def test_wrong_lambda_fails(self):
        """Using wrong λ in verification should fail."""
        torch.manual_seed(42)
        C = torch.randn(15, 10, dtype=torch.float64)
        lambda_actual = 0.1
        lambda_claimed = 1.0  # Wrong!

        Q = regularized_polar(C, lambda_actual, n_iters=15)
        result = verify_spectral_filter(C, Q, lambda_claimed, tol=1e-3)

        # Should fail because we claimed wrong λ
        assert not result["passed"]

    @pytest.mark.parametrize("lambda_reg", [0.5, 1.0, 5.0])
    def test_various_lambdas(self, lambda_reg):
        """Test verification for various λ values."""
        torch.manual_seed(42)
        C = torch.randn(12, 8, dtype=torch.float64)
        Q = regularized_polar(C, lambda_reg, n_iters=15)

        result = verify_spectral_filter(C, Q, lambda_reg, tol=0.1)  # Larger tolerance
        assert result["passed"], f"λ={lambda_reg}, max_error={result['max_error']}"


class TestCompareFilters:
    """Test filter comparison utility."""

    def test_compare_different_lambdas(self):
        """Compare filters for different λ values."""
        sigma = torch.tensor([0.5, 1.0, 2.0, 5.0], dtype=torch.float64)
        lambda_values = [0.01, 0.1, 1.0, 10.0]

        result = compare_filters(sigma, lambda_values)

        assert len(result) == len(lambda_values)

        # Smaller λ → larger filtered values
        for s_idx in range(len(sigma)):
            prev_val = float("inf")
            for lam in lambda_values:
                curr_val = result[lam][s_idx].item()
                assert curr_val < prev_val
                prev_val = curr_val

    def test_muon_filter_at_zero(self):
        """λ=0 should give all-ones (Muon's behavior)."""
        sigma = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        result = compare_filters(sigma, [0])
        assert torch.allclose(result[0], torch.ones_like(sigma))


class TestSpectralStats:
    """Test spectral statistics computation."""

    def test_identity_matrix(self):
        """Identity matrix has condition number 1."""
        I = torch.eye(10, dtype=torch.float64)
        stats = compute_spectral_stats(I)

        assert abs(stats.condition_number - 1.0) < 1e-5
        assert abs(stats.spectral_norm - 1.0) < 1e-5
        assert abs(stats.frobenius_norm - 10**0.5) < 1e-5

    def test_rank_deficient(self):
        """Rank-deficient matrix has large condition number."""
        # Create a matrix with very small singular values
        M = torch.zeros(10, 5, dtype=torch.float64)
        M[0, 0] = 1.0
        M[1, 1] = 1e-10  # Very small singular value
        stats = compute_spectral_stats(M)

        # Condition number should be very large
        assert stats.condition_number > 1e8

    def test_effective_rank(self):
        """Test effective rank computation."""
        # Diagonal matrix with varying singular values
        S = torch.tensor([1.0, 0.5, 0.1, 0.01], dtype=torch.float64)
        M = torch.diag(S)
        stats = compute_spectral_stats(M)

        # Effective rank should be less than 4 (actual rank)
        assert stats.effective_rank < 4
        assert stats.effective_rank > 1


class TestSpectralTracker:
    """Test spectral tracking during training."""

    def test_logging(self):
        """Test that tracker logs correctly."""
        tracker = SpectralTracker(log_interval=1)

        # Create a simple model
        model = torch.nn.Linear(10, 5)
        model.weight.grad = torch.randn(5, 10)

        tracker.log(0, model)
        tracker.log(1, model)
        tracker.log(2, model)

        assert "weight" in tracker.history
        assert len(tracker.history["weight"]) == 3

    def test_summary(self):
        """Test summary statistics."""
        tracker = SpectralTracker(log_interval=1)

        # Create a simple model
        model = torch.nn.Linear(10, 5)
        for step in range(10):
            model.weight.grad = torch.randn(5, 10) * (step + 1)
            tracker.log(step, model)

        summary = tracker.get_summary("weight")
        assert "condition_number" in summary
        assert "effective_rank" in summary
        assert "spectral_norm" in summary
        assert summary["num_samples"] == 10


class TestAnalyzeFilterEffect:
    """Test filter effect analysis."""

    def test_basic_analysis(self):
        """Test basic analysis output."""
        torch.manual_seed(42)
        grad = torch.randn(10, 8, dtype=torch.float64)
        lambda_reg = 0.1

        result = analyze_filter_effect(grad, lambda_reg)

        assert "original_singular_values" in result
        assert "filtered_singular_values" in result
        assert "original_condition" in result
        assert "filtered_condition" in result
        assert "compression_ratio" in result

    def test_compression_ratio(self):
        """Filtered condition number should be smaller than original."""
        torch.manual_seed(42)
        # Create ill-conditioned gradient
        U = torch.randn(10, 8, dtype=torch.float64)
        U, _ = torch.linalg.qr(U)
        S = torch.tensor([10.0, 5.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.001], dtype=torch.float64)
        V = torch.randn(8, 8, dtype=torch.float64)
        V, _ = torch.linalg.qr(V)
        grad = U @ torch.diag(S) @ V

        result = analyze_filter_effect(grad, lambda_reg=0.1)

        # Filter should reduce condition number
        assert result["filtered_condition"] < result["original_condition"]
