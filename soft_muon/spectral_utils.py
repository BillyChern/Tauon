"""
Spectral analysis utilities for SoftMuon optimizer.

This module provides tools for analyzing the spectral properties of gradients
and updates, useful for debugging, validation, and experiment analysis.
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class SpectralStats:
    """Spectral statistics for a single matrix."""

    singular_values: torch.Tensor
    condition_number: float
    effective_rank: float
    frobenius_norm: float
    spectral_norm: float
    nuclear_norm: float = 0.0
    stable_rank: float = 0.0

    def __post_init__(self):
        """Compute derived statistics."""
        S = self.singular_values
        if len(S) > 0:
            self.nuclear_norm = S.sum().item()
            self.stable_rank = (self.frobenius_norm**2) / (self.spectral_norm**2 + 1e-10)


def compute_spectral_stats(M: torch.Tensor) -> SpectralStats:
    """
    Compute comprehensive spectral statistics of a matrix.

    Args:
        M: Input matrix of shape (m, n)

    Returns:
        SpectralStats containing singular values and derived metrics
    """
    S = torch.linalg.svdvals(M)

    # Condition number (ratio of largest to smallest singular value)
    if S[-1] > 1e-10:
        condition_number = (S[0] / S[-1]).item()
    else:
        condition_number = float("inf")

    # Effective rank: (sum of σ)² / (sum of σ²)
    # Measures how many singular values contribute significantly
    sum_s = S.sum()
    sum_s2 = (S**2).sum()
    effective_rank = ((sum_s**2) / (sum_s2 + 1e-10)).item()

    frobenius_norm = torch.norm(M, "fro").item()
    spectral_norm = S[0].item()

    return SpectralStats(
        singular_values=S,
        condition_number=condition_number,
        effective_rank=effective_rank,
        frobenius_norm=frobenius_norm,
        spectral_norm=spectral_norm,
    )


def verify_spectral_filter(
    C: torch.Tensor,
    Q: torch.Tensor,
    lambda_reg: float,
    tol: float = 1e-4,
) -> Dict:
    """
    Verify that Q = C @ (C^T C + λI)^{-1/2} has correct spectral structure.

    The regularized polar should apply the filter φ_λ(σ) = σ/√(σ²+λ)
    to the singular values of C.

    Args:
        C: Original matrix
        Q: Computed regularized polar
        lambda_reg: Regularization parameter used
        tol: Tolerance for verification

    Returns:
        Dict containing:
        - expected_filter: φ_λ(σ) = σ/√(σ²+λ) applied to C's singular values
        - actual_filter: Singular values of Q
        - max_error: Maximum deviation between expected and actual
        - passed: Whether verification passed
        - original_singular_values: C's singular values
    """
    # Get SVD of original matrix
    U, S, Vh = torch.linalg.svd(C, full_matrices=False)

    # Expected singular values of Q
    expected_filter = S / torch.sqrt(S**2 + lambda_reg)

    # Actual singular values of Q
    Sq = torch.linalg.svdvals(Q)

    # Sort both for comparison (SVD may permute)
    expected_sorted, _ = expected_filter.sort(descending=True)
    actual_sorted, _ = Sq.sort(descending=True)

    # Handle potential size mismatch
    min_len = min(len(expected_sorted), len(actual_sorted))
    expected_sorted = expected_sorted[:min_len]
    actual_sorted = actual_sorted[:min_len]

    max_error = (expected_sorted - actual_sorted).abs().max().item()

    return {
        "expected_filter": expected_sorted,
        "actual_filter": actual_sorted,
        "max_error": max_error,
        "passed": max_error < tol,
        "original_singular_values": S,
        "filter_values": expected_filter,
    }


def spectral_filter_values(
    sigma: torch.Tensor,
    lambda_reg: float,
) -> torch.Tensor:
    """
    Compute spectral filter φ_λ(σ) = σ/√(σ²+λ).

    Args:
        sigma: Singular values
        lambda_reg: Regularization parameter

    Returns:
        Filtered singular values
    """
    return sigma / torch.sqrt(sigma**2 + lambda_reg)


def compare_filters(
    sigma: torch.Tensor,
    lambda_values: List[float],
) -> Dict[float, torch.Tensor]:
    """
    Compare spectral filters for different λ values.

    Args:
        sigma: Singular values to filter
        lambda_values: List of λ values to compare

    Returns:
        Dict mapping λ to filtered singular values
    """
    result = {}
    for lam in lambda_values:
        if lam == 0:
            # Muon's filter: all become 1
            result[lam] = torch.ones_like(sigma)
        else:
            result[lam] = spectral_filter_values(sigma, lam)
    return result


class SpectralTracker:
    """
    Track spectral statistics during training.

    Useful for analyzing how gradient spectra evolve and how different
    λ values affect the updates.
    """

    def __init__(
        self,
        log_interval: int = 50,
        track_layers: Optional[List[str]] = None,
        max_history: int = 1000,
    ):
        """
        Args:
            log_interval: Log every N steps
            track_layers: List of layer name patterns to track (None = all)
            max_history: Maximum history length per layer
        """
        self.log_interval = log_interval
        self.track_layers = track_layers
        self.max_history = max_history
        self.history: Dict[str, List[SpectralStats]] = {}
        self.step_history: Dict[str, List[int]] = {}
        self._current_step = 0

    def should_track(self, name: str) -> bool:
        """Check if this layer should be tracked."""
        if self.track_layers is None:
            return True
        return any(pattern in name for pattern in self.track_layers)

    def log(self, step: int, model: torch.nn.Module) -> None:
        """
        Log spectral statistics for model parameters.

        Args:
            step: Current training step
            model: Model to analyze
        """
        if step % self.log_interval != 0:
            return

        self._current_step = step

        for name, param in model.named_parameters():
            if param.grad is None or param.dim() != 2:
                continue
            if not self.should_track(name):
                continue

            stats = compute_spectral_stats(param.grad)

            if name not in self.history:
                self.history[name] = []
                self.step_history[name] = []

            self.history[name].append(stats)
            self.step_history[name].append(step)

            # Trim history if too long
            if len(self.history[name]) > self.max_history:
                self.history[name] = self.history[name][-self.max_history :]
                self.step_history[name] = self.step_history[name][-self.max_history :]

    def log_tensor(self, name: str, tensor: torch.Tensor, step: int) -> None:
        """
        Log spectral statistics for a specific tensor.

        Args:
            name: Name/identifier for this tensor
            tensor: The tensor to analyze
            step: Current step
        """
        if step % self.log_interval != 0:
            return
        if tensor.dim() != 2:
            return

        stats = compute_spectral_stats(tensor)

        if name not in self.history:
            self.history[name] = []
            self.step_history[name] = []

        self.history[name].append(stats)
        self.step_history[name].append(step)

    def get_summary(self, name: str) -> Dict:
        """
        Get summary statistics for a tracked layer.

        Args:
            name: Layer name

        Returns:
            Dict with summary statistics over training
        """
        if name not in self.history:
            return {}

        stats_list = self.history[name]
        if not stats_list:
            return {}

        condition_numbers = [s.condition_number for s in stats_list]
        effective_ranks = [s.effective_rank for s in stats_list]
        spectral_norms = [s.spectral_norm for s in stats_list]

        return {
            "condition_number": {
                "mean": sum(condition_numbers) / len(condition_numbers),
                "max": max(condition_numbers),
                "min": min(condition_numbers),
            },
            "effective_rank": {
                "mean": sum(effective_ranks) / len(effective_ranks),
                "max": max(effective_ranks),
                "min": min(effective_ranks),
            },
            "spectral_norm": {
                "mean": sum(spectral_norms) / len(spectral_norms),
                "max": max(spectral_norms),
                "min": min(spectral_norms),
            },
            "num_samples": len(stats_list),
        }

    def get_all_summaries(self) -> Dict[str, Dict]:
        """Get summaries for all tracked layers."""
        return {name: self.get_summary(name) for name in self.history}


def analyze_filter_effect(
    grad: torch.Tensor,
    lambda_reg: float,
) -> Dict:
    """
    Analyze the effect of the spectral filter on a gradient.

    Args:
        grad: Gradient matrix
        lambda_reg: Regularization parameter

    Returns:
        Dict with analysis results
    """
    S = torch.linalg.svdvals(grad)
    S_filtered = spectral_filter_values(S, lambda_reg)

    # Muon's filter for comparison
    S_muon = torch.ones_like(S)

    return {
        "original_singular_values": S,
        "filtered_singular_values": S_filtered,
        "muon_singular_values": S_muon,
        "original_condition": (S[0] / (S[-1] + 1e-10)).item(),
        "filtered_condition": (S_filtered[0] / (S_filtered[-1] + 1e-10)).item(),
        "compression_ratio": (S_filtered / S).mean().item(),
        "max_compression": (S_filtered / S).min().item(),
        "min_compression": (S_filtered / S).max().item(),
    }
