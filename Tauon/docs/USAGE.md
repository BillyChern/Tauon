# Usage Guide

## Basic Usage

### Simple Training

```python
import torch
import torch.nn as nn
from soft_muon import SoftMuon, SoftMuonConfig

# Create model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

# Configure optimizer
config = SoftMuonConfig(lr=0.02, lambda_reg=0.1)
optimizer = SoftMuon(model.parameters(), config)

# Training loop
for epoch in range(10):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
```

### Combined Optimizer (Recommended)

For models with embeddings, layer norms, and other non-matrix parameters:

```python
from soft_muon.optimizer import CombinedOptimizer

# CombinedOptimizer uses SoftMuon for 2D weights, AdamW for the rest
config = SoftMuonConfig(lr=0.02, lambda_reg=0.1)
optimizer = CombinedOptimizer(
    model,
    config,
    adamw_lr=1e-3,
    adamw_weight_decay=0.01,
)

# Training
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## Choosing Lambda (λ)

The λ parameter is the most important hyperparameter:

| λ Value | Effect |
|---------|--------|
| 0.001 - 0.01 | Almost like Muon (aggressive orthogonalization) |
| 0.1 | Balanced (good starting point) |
| 1.0 - 10.0 | More like gradient descent (conservative) |

**Tuning Strategy:**
1. Start with λ=0.1
2. If training is unstable, increase λ
3. If training is slow, decrease λ
4. Use the lambda sweep configs to find optimal value

## Lambda Modes

### Fixed Lambda (Default)

```python
config = SoftMuonConfig(
    lambda_reg=0.1,
    lambda_mode='fixed',
)
```

### Scheduled Lambda

Start high (for stability) and decay:

```python
config = SoftMuonConfig(
    lambda_reg=0.1,
    lambda_mode='scheduled',
    lambda_warmup_steps=100,
    lambda_start_multiplier=10.0,  # Start at 1.0
    lambda_end_multiplier=0.1,     # End at 0.01
)
```

### Adaptive Lambda

Automatically adjust based on gradient condition number:

```python
config = SoftMuonConfig(
    lambda_reg=0.1,
    lambda_mode='adaptive',
    adaptive_lambda_min=0.01,
    adaptive_lambda_max=10.0,
)
```

## Parameter Exclusion

By default, SoftMuon excludes certain parameters from regularized polar:

```python
config = SoftMuonConfig(
    exclude_names=('embed', 'head', 'ln', 'norm', 'bias'),
)
```

To customize which parameters use SoftMuon:

```python
# Apply only to MLP layers
config = SoftMuonConfig(
    apply_to='mlp_only',
)

# Or manually select
softmuon_params = []
other_params = []
for name, p in model.named_parameters():
    if 'mlp' in name and p.dim() == 2:
        softmuon_params.append(p)
    else:
        other_params.append(p)

opt_soft = SoftMuon(softmuon_params, config)
opt_adamw = torch.optim.AdamW(other_params, lr=1e-3)
```

## Learning Rate Schedule

```python
config = SoftMuonConfig(lr=0.02)
optimizer = SoftMuon(model.parameters(), config)

# Cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
)

# Or with warmup
def lr_lambda(step):
    warmup_steps = 100
    if step < warmup_steps:
        return step / warmup_steps
    return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

## Mixed Precision Training

```python
scaler = torch.amp.GradScaler('cuda')

for data, target in train_loader:
    optimizer.zero_grad()

    with torch.amp.autocast('cuda'):
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Debugging and Analysis

### Track Spectral Statistics

```python
from soft_muon import SpectralTracker

tracker = SpectralTracker(log_interval=100)

for step, (data, target) in enumerate(train_loader):
    # Training step...

    tracker.log(step, model)

# After training
for name in tracker.history:
    summary = tracker.get_summary(name)
    print(f"{name}: condition={summary['condition_number']['mean']:.2f}")
```

### Verify Spectral Filter

```python
from soft_muon import verify_spectral_filter
from soft_muon.newton_schulz import regularized_polar

# Get a gradient
grad = model.fc.weight.grad

# Apply regularized polar
Q = regularized_polar(grad, lambda_reg=0.1)

# Verify
result = verify_spectral_filter(grad, Q, lambda_reg=0.1)
print(f"Filter verification passed: {result['passed']}")
print(f"Max error: {result['max_error']:.6f}")
```

## Common Issues

### Training Instability

**Symptoms:** Loss explodes, NaN values

**Solutions:**
1. Increase λ (e.g., from 0.1 to 1.0)
2. Reduce learning rate
3. Use scheduled λ with high initial value
4. Increase ns_iters for better numerical accuracy

### Slow Convergence

**Symptoms:** Training is slow compared to Muon

**Solutions:**
1. Decrease λ (e.g., from 0.1 to 0.01)
2. Increase learning rate
3. Check that matrix parameters are actually using SoftMuon

### Memory Issues

**Solutions:**
1. SoftMuon has same memory footprint as Muon (momentum buffers only)
2. For very large matrices, consider gradient checkpointing
3. Reduce batch size if needed
