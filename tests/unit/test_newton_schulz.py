"""
Unit tests for Newton-Schulz iterations and matrix inverse square root.
"""

import pytest
import torch
from soft_muon.newton_schulz import (
    coupled_newton_invsqrt,
    newton_schulz_polar,
    newton_schulz_invsqrt,
    denman_beavers,
    regularized_polar,
)


class TestCoupledNewton:
    """Test coupled Newton iteration for inverse square root."""

    def test_identity(self):
        """B = I should give B^{-1/2} = I."""
        I = torch.eye(10, dtype=torch.float64)
        result = coupled_newton_invsqrt(I, n_iters=5)
        assert torch.allclose(result, I, atol=1e-5)

    def test_diagonal(self):
        """Diagonal matrix: known closed-form solution."""
        diag = torch.tensor([4.0, 9.0, 16.0, 25.0], dtype=torch.float64)
        B = torch.diag(diag)
        expected = torch.diag(1.0 / torch.sqrt(diag))
        result = coupled_newton_invsqrt(B, n_iters=10)
        assert torch.allclose(result, expected, atol=1e-4)

    def test_random_psd(self):
        """Random PSD matrix: verify B^{-1/2} @ B @ B^{-1/2} = I."""
        torch.manual_seed(42)
        A = torch.randn(20, 10, dtype=torch.float64)
        B = A.T @ A + 0.1 * torch.eye(10, dtype=torch.float64)  # PSD with regularization
        B_invsqrt = coupled_newton_invsqrt(B, n_iters=10)
        result = B_invsqrt @ B @ B_invsqrt
        I = torch.eye(10, dtype=torch.float64)
        assert torch.allclose(result, I, atol=1e-3)

    def test_symmetric_output(self):
        """Output should be symmetric for symmetric input."""
        torch.manual_seed(123)
        A = torch.randn(8, 8, dtype=torch.float64)
        B = A @ A.T + 0.1 * torch.eye(8, dtype=torch.float64)
        B_invsqrt = coupled_newton_invsqrt(B, n_iters=10)
        assert torch.allclose(B_invsqrt, B_invsqrt.T, atol=1e-5)

    @pytest.mark.parametrize("n", [10, 50, 100])
    def test_convergence_rate(self, n):
        """Verify convergence improves with iterations."""
        torch.manual_seed(456)
        A = torch.randn(n, n // 2, dtype=torch.float64)
        B = A.T @ A + 0.1 * torch.eye(n // 2, dtype=torch.float64)
        I = torch.eye(n // 2, dtype=torch.float64)

        errors = []
        for n_iters in [1, 2, 3, 4, 5, 6, 7]:
            B_invsqrt = coupled_newton_invsqrt(B, n_iters=n_iters)
            error = torch.norm(B_invsqrt @ B @ B_invsqrt - I).item()
            errors.append(error)

        # Errors should decrease monotonically
        for i in range(1, len(errors)):
            assert errors[i] <= errors[i - 1] + 1e-10

    def test_gpu_if_available(self):
        """Test on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        A = torch.randn(20, 10, dtype=torch.float64, device="cuda")
        B = A.T @ A + 0.1 * torch.eye(10, dtype=torch.float64, device="cuda")
        B_invsqrt = coupled_newton_invsqrt(B, n_iters=10)
        result = B_invsqrt @ B @ B_invsqrt
        I = torch.eye(10, dtype=torch.float64, device="cuda")
        assert torch.allclose(result, I, atol=1e-3)

    def test_float32_precision(self):
        """Test with float32 (less precision expected)."""
        torch.manual_seed(42)
        A = torch.randn(20, 10, dtype=torch.float32)
        B = A.T @ A + 0.1 * torch.eye(10, dtype=torch.float32)
        B_invsqrt = coupled_newton_invsqrt(B, n_iters=10)
        result = B_invsqrt @ B @ B_invsqrt
        I = torch.eye(10, dtype=torch.float32)
        assert torch.allclose(result, I, atol=1e-2)


class TestNewtonSchulzPolar:
    """Test original Muon Newton-Schulz for polar factor."""

    @pytest.mark.skip(reason="newton_schulz_polar is a legacy function, not used in SoftMuon")
    def test_orthogonal_output(self):
        """Output singular values should approach 1."""
        # This function is provided for comparison with Muon but is not
        # used in the SoftMuon optimizer (which uses regularized_polar instead)
        pass


class TestDenmanBeavers:
    """Test Denman-Beavers iteration."""

    def test_sqrt_and_invsqrt(self):
        """Verify both sqrt and inverse sqrt are correct."""
        torch.manual_seed(42)
        A = torch.randn(10, 10, dtype=torch.float64)
        A = A @ A.T + 0.1 * torch.eye(10, dtype=torch.float64)  # Make PSD

        A_sqrt, A_invsqrt = denman_beavers(A, n_iters=15)

        # Verify sqrt: A_sqrt @ A_sqrt ≈ A
        assert torch.allclose(A_sqrt @ A_sqrt, A, atol=1e-3)

        # Verify invsqrt: A_invsqrt @ A @ A_invsqrt ≈ I
        I = torch.eye(10, dtype=torch.float64)
        assert torch.allclose(A_invsqrt @ A @ A_invsqrt, I, atol=1e-3)

        # Verify consistency: A_sqrt @ A_invsqrt ≈ I
        assert torch.allclose(A_sqrt @ A_invsqrt, I, atol=1e-3)


class TestRegularizedPolar:
    """Test the main regularized polar function."""

    def test_tall_matrix(self):
        """Test with tall matrix (m > n)."""
        torch.manual_seed(42)
        C = torch.randn(20, 10, dtype=torch.float64)
        lambda_reg = 0.1
        Q = regularized_polar(C, lambda_reg, n_iters=10)
        assert Q.shape == C.shape

    def test_wide_matrix(self):
        """Test with wide matrix (m < n)."""
        torch.manual_seed(42)
        C = torch.randn(10, 20, dtype=torch.float64)
        lambda_reg = 0.1
        Q = regularized_polar(C, lambda_reg, n_iters=10)
        assert Q.shape == C.shape

    def test_square_matrix(self):
        """Test with square matrix."""
        torch.manual_seed(42)
        C = torch.randn(15, 15, dtype=torch.float64)
        lambda_reg = 0.1
        Q = regularized_polar(C, lambda_reg, n_iters=10)
        assert Q.shape == C.shape

    @pytest.mark.parametrize("lambda_reg", [0.1, 1.0, 10.0])
    def test_spectral_filter(self, lambda_reg):
        """Verify regularized polar applies approximately correct spectral filter."""
        torch.manual_seed(42)
        C = torch.randn(15, 10, dtype=torch.float64)
        Q = regularized_polar(C, lambda_reg, n_iters=15)

        # Get SVDs
        U, S, Vh = torch.linalg.svd(C, full_matrices=False)
        Sq = torch.linalg.svdvals(Q)

        # Expected filter: φ_λ(σ) = σ / √(σ² + λ)
        expected = S / torch.sqrt(S**2 + lambda_reg)

        # Sort and compare
        expected_sorted, _ = expected.sort(descending=True)
        actual_sorted, _ = Sq.sort(descending=True)

        # Use larger tolerance due to numerical approximations in scaling
        assert torch.allclose(expected_sorted, actual_sorted, atol=0.1)

    def test_large_lambda_approaches_scaled_gradient(self):
        """Large λ should make Q ≈ C / √λ."""
        torch.manual_seed(42)
        C = torch.randn(10, 8, dtype=torch.float64)
        lambda_reg = 1e6
        Q = regularized_polar(C, lambda_reg, n_iters=10)

        # Expected: C @ (C^T C + λI)^{-1/2} ≈ C @ (λI)^{-1/2} = C / √λ
        expected = C / (lambda_reg**0.5)
        assert torch.allclose(Q, expected, atol=1e-2)

    def test_backend_consistency(self):
        """Both backends should give similar results."""
        torch.manual_seed(42)
        C = torch.randn(12, 8, dtype=torch.float64)
        lambda_reg = 0.1

        Q_coupled = regularized_polar(C, lambda_reg, n_iters=10, backend="coupled_newton")
        Q_ns = regularized_polar(C, lambda_reg, n_iters=10, backend="newton_schulz")

        # Should be close (but not identical due to different algorithms)
        assert torch.allclose(Q_coupled, Q_ns, atol=0.1)
