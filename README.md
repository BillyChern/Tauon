# Soft Muon Optimizer

**Regularized Polar Spectral Shaping for Deep Learning**

Soft Muon is a production-quality optimizer implementing the spectral filter φ_λ(σ) = σ/√(σ² + λ) instead of Muon's hard φ(σ) = 1. This provides a smooth interpolation between gradient descent and Muon's polar decomposition.

## Key Features

- **Soft orthogonalization**: Reduces condition number while preserving gradient magnitude information
- **Tunable regularization**: λ parameter controls the regularization-orthogonalization tradeoff
- **Multiple λ modes**: Fixed, adaptive, scheduled, and per-layer strategies
- **Production-ready**: Comprehensive tests, type hints, and documentation

## Installation

```bash
# From source
pip install -e .

# With all dependencies
pip install -e ".[all]"
```

## Quick Start

```python
import torch
from soft_muon import SoftMuon, SoftMuonConfig

# Create model
model = torch.nn.Linear(100, 10)

# Configure optimizer
config = SoftMuonConfig(
    lr=0.02,
    momentum=0.95,
    lambda_reg=0.1,  # Key parameter: regularization strength
)

# Create optimizer
optimizer = SoftMuon(model.parameters(), config)

# Training loop
for data, target in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(data), target)
    loss.backward()
    optimizer.step()
```

## Theory

### Muon's Approach
Muon applies the polar decomposition to momentum updates, effectively mapping all singular values to 1:
```
φ(σ) = 1  (for all σ > 0)
```

### Soft Muon's Approach
Soft Muon uses a regularized polar decomposition with spectral filter:
```
φ_λ(σ) = σ / √(σ² + λ)
```

This is computed via:
```
Q_λ(C) = C @ (C^T C + λI)^{-1/2}
```

**Properties:**
- **λ → 0**: Approaches Muon (φ → 1)
- **λ → ∞**: Approaches scaled gradient descent (φ → 0)
- **Intermediate λ**: Soft orthogonalization that reduces condition number while preserving relative singular value magnitudes

## Configuration

```python
from soft_muon import SoftMuonConfig

config = SoftMuonConfig(
    # Core parameters
    lr=0.02,                    # Learning rate
    momentum=0.95,              # Momentum coefficient
    lambda_reg=0.1,             # Regularization parameter

    # Lambda adaptation
    lambda_mode='fixed',        # 'fixed', 'adaptive', 'scheduled', 'per_layer'

    # Algorithm settings
    ns_iters=5,                 # Newton-Schulz iterations
    backend='coupled_newton',   # 'coupled_newton' or 'newton_schulz'

    # Parameter selection
    exclude_names=('embed', 'head', 'ln', 'norm'),  # Don't apply to these

    # Other options
    use_nesterov=True,
    weight_decay=0.0,
)
```

## Benchmarks

### CIFAR-10 Speedrun

```bash
# Run training
python benchmarks/cifar10/train_softmuon.py --lambda-reg 0.1

# Run hyperparameter sweep
python experiments/scripts/run_sweep.py --config experiments/configs/lambda_sweep.yaml
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/test_newton_schulz.py -v

# Run with coverage
pytest tests/ --cov=soft_muon --cov-report=html
```

## Project Structure

```
soft_muon/
├── __init__.py              # Package exports
├── optimizer.py             # Main SoftMuon class
├── newton_schulz.py         # Matrix inverse sqrt algorithms
├── spectral_utils.py        # Spectral analysis tools
├── adaptive_lambda.py       # Lambda adaptation strategies
└── config.py                # Configuration dataclass

benchmarks/
└── cifar10/                 # CIFAR-10 speedrun benchmark

experiments/
├── configs/                 # Experiment configurations
├── scripts/                 # Experiment runners
└── results/                 # Output directory

tests/
├── unit/                    # Unit tests
├── integration/             # Integration tests
└── benchmarks/              # Performance tests
```

## Algorithm Details

### Coupled Newton Iteration

We use the coupled Newton iteration (Higham & Guo) to compute (C^T C + λI)^{-1/2}:

```
X₀ = I, M₀ = B_scaled
X_{k+1} = 0.5 * X_k @ (3I - M_k)
M_{k+1} = 0.5 * (3I - M_k) @ M_k
```

This converges quadratically and is more stable than naive Newton-Schulz for the inverse square root.

### Wide vs Tall Matrices

For efficiency, we use different formulations based on matrix shape:
- **Tall matrix (m ≥ n)**: Q_λ = C @ (C^T C + λI)^{-1/2}
- **Wide matrix (m < n)**: Q_λ = (CC^T + λI)^{-1/2} @ C

This ensures we always work with the smaller dimension.

## Citation

If you use Soft Muon in your research, please cite:

```bibtex
@software{soft_muon,
  title={Soft Muon: Regularized Polar Spectral Shaping},
  year={2024},
}
```

## License

MIT License
