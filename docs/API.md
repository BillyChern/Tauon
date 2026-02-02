# API Reference

## Core Classes

### SoftMuon

```python
class SoftMuon(torch.optim.Optimizer):
    """
    Soft Spectrum Shaping Optimizer (Regularized Polar).

    Applies spectral filter φ_λ(σ) = σ / √(σ² + λ) to momentum updates.
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
            **kwargs: Override config values
        """

    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        """Perform a single optimization step."""

    def register_param_names(self, named_params: Iterable[tuple]) -> None:
        """Register parameter names for exclusion checking."""
```

### SoftMuonConfig

```python
@dataclass
class SoftMuonConfig:
    """Configuration for SoftMuon optimizer."""

    # Core parameters
    lr: float = 0.02                    # Learning rate
    momentum: float = 0.95              # Momentum coefficient
    lambda_reg: float = 0.1             # Regularization parameter λ
    lambda_mode: str = 'fixed'          # 'fixed', 'adaptive', 'scheduled', 'per_layer'

    # Algorithm settings
    ns_iters: int = 5                   # Newton-Schulz iterations
    backend: str = 'coupled_newton'     # 'coupled_newton' or 'newton_schulz'

    # Parameter selection
    apply_to: str = 'all_2d'            # 'all_2d', 'mlp_only', 'attn_only'
    exclude_names: tuple = ('embed', 'head', 'ln', 'norm')

    # Other options
    eps: float = 1e-7                   # Numerical stability
    use_nesterov: bool = True           # Nesterov momentum
    weight_decay: float = 0.0           # L2 regularization

    # Scheduled lambda parameters
    lambda_warmup_steps: int = 100
    lambda_start_multiplier: float = 10.0
    lambda_end_multiplier: float = 0.1

    # Adaptive lambda parameters
    adaptive_lambda_min: float = 0.01
    adaptive_lambda_max: float = 10.0
```

### CombinedOptimizer

```python
class CombinedOptimizer:
    """Combined optimizer wrapping SoftMuon + AdamW."""

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
            config: SoftMuonConfig for matrix parameters
            adamw_lr: Learning rate for non-matrix parameters
            adamw_weight_decay: Weight decay for non-matrix parameters
        """

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients for all parameters."""

    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        """Perform optimization step."""

    def state_dict(self) -> Dict:
        """Get combined state dict."""

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load combined state dict."""
```

## Newton-Schulz Functions

### coupled_newton_invsqrt

```python
def coupled_newton_invsqrt(
    B: torch.Tensor,
    n_iters: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Compute B^{-1/2} via coupled Newton iteration.

    Args:
        B: Symmetric positive definite matrix (n, n)
        n_iters: Number of iterations
        eps: Numerical stability constant

    Returns:
        B^{-1/2} of shape (n, n)
    """
```

### regularized_polar

```python
def regularized_polar(
    C: torch.Tensor,
    lambda_reg: float,
    n_iters: int = 5,
    eps: float = 1e-7,
    backend: str = "coupled_newton",
) -> torch.Tensor:
    """
    Compute regularized polar factor Q_λ(C) = C @ (C^T C + λI)^{-1/2}.

    Args:
        C: Input matrix (m, n)
        lambda_reg: Regularization parameter
        n_iters: Iterations for inverse sqrt
        eps: Numerical stability
        backend: 'coupled_newton' or 'newton_schulz'

    Returns:
        Q_λ(C) of shape (m, n)
    """
```

## Spectral Utilities

### SpectralStats

```python
@dataclass
class SpectralStats:
    """Spectral statistics for a matrix."""
    singular_values: torch.Tensor
    condition_number: float
    effective_rank: float
    frobenius_norm: float
    spectral_norm: float
    nuclear_norm: float
    stable_rank: float
```

### compute_spectral_stats

```python
def compute_spectral_stats(M: torch.Tensor) -> SpectralStats:
    """Compute spectral statistics of a matrix."""
```

### verify_spectral_filter

```python
def verify_spectral_filter(
    C: torch.Tensor,
    Q: torch.Tensor,
    lambda_reg: float,
    tol: float = 1e-4,
) -> Dict:
    """
    Verify that Q has correct spectral structure.

    Returns:
        Dict with expected_filter, actual_filter, max_error, passed
    """
```

### SpectralTracker

```python
class SpectralTracker:
    """Track spectral statistics during training."""

    def __init__(
        self,
        log_interval: int = 50,
        track_layers: Optional[List[str]] = None,
        max_history: int = 1000,
    ):
        """Initialize tracker."""

    def log(self, step: int, model: torch.nn.Module) -> None:
        """Log spectral statistics."""

    def get_summary(self, name: str) -> Dict:
        """Get summary statistics for a layer."""
```

## Adaptive Lambda Strategies

### FixedLambda

```python
class FixedLambda(AdaptiveLambdaStrategy):
    """Always returns base lambda."""
```

### ScheduledLambda

```python
class ScheduledLambda(AdaptiveLambdaStrategy):
    """Scheduled λ decay over training."""

    def __init__(
        self,
        warmup_steps: int = 100,
        start_multiplier: float = 10.0,
        end_multiplier: float = 0.1,
        schedule: str = "cosine",  # 'cosine', 'linear', 'exponential'
    ):
```

### GradientAdaptiveLambda

```python
class GradientAdaptiveLambda(AdaptiveLambdaStrategy):
    """Adaptive λ based on gradient spectral statistics."""

    def __init__(
        self,
        min_lambda: float = 0.01,
        max_lambda: float = 10.0,
        ema_decay: float = 0.99,
    ):
```

### PerLayerLambda

```python
class PerLayerLambda(AdaptiveLambdaStrategy):
    """Per-layer λ values."""

    def __init__(
        self,
        layer_lambdas: Optional[Dict[str, float]] = None,
        default_lambda: float = 0.1,
    ):

    def register_param(self, param: torch.Tensor, name: str) -> None:
        """Register parameter name."""
```
