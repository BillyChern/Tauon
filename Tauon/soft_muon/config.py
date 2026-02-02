"""Configuration dataclasses for SoftMuon optimizer."""

from dataclasses import dataclass, field
from typing import Literal, Tuple, Optional


@dataclass
class SoftMuonConfig:
    """
    Configuration for SoftMuon optimizer.

    Attributes:
        lr: Learning rate. Default: 0.02
        momentum: Momentum coefficient. Default: 0.95
        lambda_reg: Regularization parameter λ for spectral filter φ_λ(σ) = σ/√(σ²+λ).
            Higher values = more regularization = closer to gradient descent.
            Lower values = closer to Muon's polar decomposition. Default: 0.1
        lambda_mode: How to determine λ per step/layer.
            - 'fixed': Use lambda_reg everywhere
            - 'adaptive': Estimate λ from gradient statistics
            - 'per_layer': Different λ per parameter (requires lambda_per_layer dict)
            - 'scheduled': Decay λ over training
            Default: 'fixed'
        ns_iters: Number of Newton-Schulz iterations for matrix inverse sqrt.
            More iterations = better accuracy but slower. Default: 5
        ns_coeffs: Coefficients for Newton-Schulz iteration (if using that backend).
            Default: (3.4445, -4.7750, 2.0315) from Muon paper
        backend: Which algorithm to use for computing (C^T C + λI)^{-1/2}.
            - 'coupled_newton': Coupled Newton iteration (recommended, stable)
            - 'newton_schulz': Original Muon-style iteration (for comparison)
            Default: 'coupled_newton'
        apply_to: Which parameters to apply regularized polar to.
            - 'all_2d': All 2D parameters (matrices)
            - 'mlp_only': Only MLP/feed-forward layers
            - 'attn_only': Only attention layers
            Default: 'all_2d'
        exclude_names: Tuple of substrings; parameters with these in their name
            will use standard gradient descent instead of regularized polar.
            Default: ('embed', 'head', 'ln', 'norm')
        eps: Small constant for numerical stability. Default: 1e-7
        use_nesterov: Whether to use Nesterov momentum. Default: True
        weight_decay: L2 weight decay coefficient. Default: 0.0
        foreach: Use foreach implementation for parameter groups. Default: True
    """

    lr: float = 0.02
    momentum: float = 0.95
    lambda_reg: float = 0.1
    lambda_mode: Literal["fixed", "adaptive", "per_layer", "scheduled"] = "fixed"
    ns_iters: int = 5
    ns_coeffs: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
    backend: Literal["coupled_newton", "newton_schulz"] = "coupled_newton"
    apply_to: Literal["all_2d", "mlp_only", "attn_only"] = "all_2d"
    exclude_names: Tuple[str, ...] = ("embed", "head", "ln", "norm")
    eps: float = 1e-7
    use_nesterov: bool = True
    weight_decay: float = 0.0
    foreach: bool = True

    # Scheduled lambda parameters
    lambda_warmup_steps: int = 100
    lambda_start_multiplier: float = 10.0
    lambda_end_multiplier: float = 0.1

    # Adaptive lambda parameters
    adaptive_lambda_min: float = 0.01
    adaptive_lambda_max: float = 10.0

    def __post_init__(self):
        """Validate configuration."""
        if self.lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.lr}")
        if not 0 <= self.momentum < 1:
            raise ValueError(f"Momentum must be in [0, 1), got {self.momentum}")
        if self.lambda_reg < 0:
            raise ValueError(f"lambda_reg must be non-negative, got {self.lambda_reg}")
        if self.ns_iters < 1:
            raise ValueError(f"ns_iters must be >= 1, got {self.ns_iters}")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
