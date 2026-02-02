"""
SoftMuon Optimizer - Regularized Polar Spectral Shaping.

This module implements the main SoftMuon optimizer class, which applies
the spectral filter φ_λ(σ) = σ/√(σ² + λ) to momentum updates via the
regularized polar decomposition Q_λ(C) = C @ (C^T C + λI)^{-1/2}.
"""

import torch
from torch.optim import Optimizer
from typing import Dict, List, Optional, Callable, Any, Iterable, Union
import math

from soft_muon.config import SoftMuonConfig
from soft_muon.newton_schulz import regularized_polar, coupled_newton_invsqrt
from soft_muon.adaptive_lambda import (
    AdaptiveLambdaStrategy,
    FixedLambda,
    ScheduledLambda,
    GradientAdaptiveLambda,
    PerLayerLambda,
)


class SoftMuon(Optimizer):
    """
    Soft Spectrum Shaping Optimizer (Regularized Polar).

    Applies spectral filter φ_λ(σ) = σ / √(σ² + λ) to momentum updates,
    computed via Q_λ(C) = C @ (C^T C + λI)^{-1/2}.

    This interpolates between:
    - λ → 0: Muon's polar decomposition (all singular values → 1)
    - λ → ∞: Scaled gradient descent (preserves relative singular values)

    The regularized polar provides a "soft" orthogonalization that:
    1. Reduces condition number of updates (like Muon)
    2. Preserves some gradient magnitude information (unlike Muon)
    3. Provides a tunable regularization-orthogonalization tradeoff

    Args:
        params: Model parameters or parameter groups
        config: SoftMuonConfig with all hyperparameters

    Example:
        >>> config = SoftMuonConfig(lr=0.02, lambda_reg=0.1)
        >>> optimizer = SoftMuon(model.parameters(), config)
        >>> for data, target in dataloader:
        ...     optimizer.zero_grad()
        ...     loss = criterion(model(data), target)
        ...     loss.backward()
        ...     optimizer.step()

    Note:
        For best results, use SoftMuon for 2D weight matrices (linear, conv)
        and a separate optimizer (e.g., AdamW) for 1D parameters (biases,
        layer norms, embeddings).
    """

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
        config: Optional[SoftMuonConfig] = None,
        **kwargs,
    ):
        """
        Initialize SoftMuon optimizer.

        Args:
            params: Parameters to optimize
            config: SoftMuonConfig object. If None, uses defaults.
            **kwargs: Override config values (e.g., lr=0.01)
        """
        if config is None:
            config = SoftMuonConfig(**kwargs)
        else:
            # Allow kwargs to override config
            config_dict = {k: v for k, v in config.__dict__.items()}
            config_dict.update(kwargs)
            config = SoftMuonConfig(**config_dict)

        self.config = config

        defaults = dict(
            lr=config.lr,
            momentum=config.momentum,
            lambda_reg=config.lambda_reg,
            weight_decay=config.weight_decay,
        )
        super().__init__(params, defaults)

        # Setup lambda strategy
        self._setup_lambda_strategy()

        # Build param name lookup
        self._param_to_name: Dict[int, str] = {}

    def _setup_lambda_strategy(self) -> None:
        """Initialize the λ adaptation strategy based on config."""
        if self.config.lambda_mode == "fixed":
            self._lambda_strategy = FixedLambda()
        elif self.config.lambda_mode == "scheduled":
            self._lambda_strategy = ScheduledLambda(
                warmup_steps=self.config.lambda_warmup_steps,
                start_multiplier=self.config.lambda_start_multiplier,
                end_multiplier=self.config.lambda_end_multiplier,
            )
        elif self.config.lambda_mode == "adaptive":
            self._lambda_strategy = GradientAdaptiveLambda(
                min_lambda=self.config.adaptive_lambda_min,
                max_lambda=self.config.adaptive_lambda_max,
            )
        elif self.config.lambda_mode == "per_layer":
            self._lambda_strategy = PerLayerLambda(
                default_lambda=self.config.lambda_reg,
            )
        else:
            raise ValueError(f"Unknown lambda_mode: {self.config.lambda_mode}")

    def register_param_names(self, named_params: Iterable[tuple]) -> None:
        """
        Register parameter names for exclusion checking and per-layer λ.

        Args:
            named_params: Iterator of (name, param) tuples from model.named_parameters()
        """
        for name, param in named_params:
            self._param_to_name[id(param)] = name
            if isinstance(self._lambda_strategy, PerLayerLambda):
                self._lambda_strategy.register_param(param, name)

    def _get_param_name(self, p: torch.Tensor) -> str:
        """Get registered name for a parameter."""
        return self._param_to_name.get(id(p), "")

    def _should_orthogonalize(self, p: torch.Tensor, group: Dict) -> bool:
        """
        Check if parameter should have regularized polar applied.

        Applies to 2D parameters (matrices) that are not in the exclusion list.
        """
        if p.dim() != 2:
            return False

        name = self._get_param_name(p)
        for excl in self.config.exclude_names:
            if excl.lower() in name.lower():
                return False

        return True

    def _get_lambda(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        state: Dict,
        group: Dict,
    ) -> float:
        """Get effective λ for this parameter."""
        return self._lambda_strategy.get_lambda(
            grad=grad,
            param=p,
            state=state,
            base_lambda=group["lambda_reg"],
        )

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        """
        Perform a single optimization step.

        Args:
            closure: Optional closure that reevaluates the model and returns loss

        Returns:
            Loss value if closure provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SoftMuon does not support sparse gradients")

                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p)

                state["step"] += 1
                buf = state["momentum_buffer"]

                # Weight decay (decoupled, like AdamW)
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                # Momentum update
                if self.config.use_nesterov:
                    # Nesterov momentum: look ahead
                    buf.mul_(group["momentum"]).add_(grad)
                    update = grad + group["momentum"] * buf
                else:
                    # Standard momentum
                    buf.mul_(group["momentum"]).add_(grad)
                    update = buf.clone()

                # Apply regularized polar to 2D params
                if self._should_orthogonalize(p, group):
                    lambda_eff = self._get_lambda(p, grad, state, group)
                    update = self._apply_regularized_polar(update, lambda_eff)

                # Parameter update
                p.add_(update, alpha=-group["lr"])

                # Update lambda strategy state
                self._lambda_strategy.update_state(state, grad)

        return loss

    def _apply_regularized_polar(
        self,
        C: torch.Tensor,
        lambda_reg: float,
    ) -> torch.Tensor:
        """
        Apply regularized polar: Q_λ(C) = C @ (C^T C + λI)^{-1/2}

        For m×n matrix C:
        - If m < n (wide): Uses (CC^T + λI)^{-1/2} @ C
        - If m >= n (tall): Uses C @ (C^T C + λI)^{-1/2}

        Args:
            C: Update matrix
            lambda_reg: Regularization parameter

        Returns:
            Regularized polar factor
        """
        if lambda_reg <= 0:
            # Fall back to standard scaling when λ=0
            # (true polar would require separate implementation)
            return C

        return regularized_polar(
            C,
            lambda_reg=lambda_reg,
            n_iters=self.config.ns_iters,
            eps=self.config.eps,
            backend=self.config.backend,
        )


def create_optimizer_groups(
    model: torch.nn.Module,
    config: SoftMuonConfig,
    adamw_lr: float = 1e-3,
    adamw_weight_decay: float = 0.0,
) -> tuple:
    """
    Create separate optimizer groups for SoftMuon and AdamW.

    SoftMuon is applied to 2D weight matrices, AdamW to everything else.

    Args:
        model: The model to optimize
        config: SoftMuonConfig
        adamw_lr: Learning rate for AdamW parameters
        adamw_weight_decay: Weight decay for AdamW parameters

    Returns:
        Tuple of (SoftMuon optimizer, AdamW optimizer)
    """
    softmuon_params = []
    adamw_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Check if this should use SoftMuon
        use_softmuon = p.dim() == 2
        for excl in config.exclude_names:
            if excl.lower() in name.lower():
                use_softmuon = False
                break

        if use_softmuon:
            softmuon_params.append(p)
        else:
            adamw_params.append(p)

    # Create optimizers
    soft_opt = SoftMuon(softmuon_params, config)
    soft_opt.register_param_names(
        [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    )

    adamw_opt = torch.optim.AdamW(
        adamw_params,
        lr=adamw_lr,
        weight_decay=adamw_weight_decay,
    ) if adamw_params else None

    return soft_opt, adamw_opt


class CombinedOptimizer:
    """
    Combined optimizer that wraps SoftMuon + AdamW for convenience.

    Applies SoftMuon to 2D matrices and AdamW to other parameters.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: SoftMuonConfig,
        adamw_lr: float = 1e-3,
        adamw_weight_decay: float = 0.0,
    ):
        """
        Args:
            model: Model to optimize
            config: SoftMuonConfig
            adamw_lr: Learning rate for non-SoftMuon parameters
            adamw_weight_decay: Weight decay for non-SoftMuon parameters
        """
        self.soft_opt, self.adamw_opt = create_optimizer_groups(
            model, config, adamw_lr, adamw_weight_decay
        )

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients for all parameters."""
        self.soft_opt.zero_grad(set_to_none=set_to_none)
        if self.adamw_opt is not None:
            self.adamw_opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        """Perform optimization step."""
        loss = self.soft_opt.step(closure)
        if self.adamw_opt is not None:
            self.adamw_opt.step()
        return loss

    def state_dict(self) -> Dict:
        """Get combined state dict."""
        return {
            "soft_opt": self.soft_opt.state_dict(),
            "adamw_opt": self.adamw_opt.state_dict() if self.adamw_opt else None,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load combined state dict."""
        self.soft_opt.load_state_dict(state_dict["soft_opt"])
        if self.adamw_opt is not None and state_dict["adamw_opt"] is not None:
            self.adamw_opt.load_state_dict(state_dict["adamw_opt"])
