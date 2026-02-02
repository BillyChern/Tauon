"""
Soft Muon Optimizer - Regularized Polar Spectral Shaping.

A production-quality optimizer implementing the spectral filter φ_λ(σ) = σ/√(σ² + λ)
instead of Muon's hard φ(σ) = 1.
"""

from soft_muon.optimizer import SoftMuon, SoftMuonConfig
from soft_muon.newton_schulz import (
    coupled_newton_invsqrt,
    newton_schulz_polar,
    denman_beavers,
)
from soft_muon.spectral_utils import (
    SpectralStats,
    compute_spectral_stats,
    verify_spectral_filter,
    SpectralTracker,
)
from soft_muon.adaptive_lambda import (
    AdaptiveLambdaStrategy,
    FixedLambda,
    ScheduledLambda,
    GradientAdaptiveLambda,
    PerLayerLambda,
)

__version__ = "0.1.0"
__all__ = [
    "SoftMuon",
    "SoftMuonConfig",
    "coupled_newton_invsqrt",
    "newton_schulz_polar",
    "denman_beavers",
    "SpectralStats",
    "compute_spectral_stats",
    "verify_spectral_filter",
    "SpectralTracker",
    "AdaptiveLambdaStrategy",
    "FixedLambda",
    "ScheduledLambda",
    "GradientAdaptiveLambda",
    "PerLayerLambda",
]
