# Experiment Reproduction Guide

## Setup

1. Install dependencies:
```bash
pip install -e ".[all]"
```

2. Download CIFAR-10 (happens automatically on first run):
```bash
python -c "import torchvision; torchvision.datasets.CIFAR10('./data', download=True)"
```

## Running Experiments

### Quick Smoke Test

Verify everything works:

```bash
python benchmarks/cifar10/train_softmuon.py --smoke-test
```

### Lambda Sweep

Find optimal λ value:

```bash
# Run sweep
python experiments/scripts/run_sweep.py --config experiments/configs/lambda_sweep.yaml

# Analyze results
python experiments/scripts/analyze_results.py --results experiments/results/lambda_sweep
```

### Learning Rate × Lambda Grid

```bash
python experiments/scripts/run_sweep.py --config experiments/configs/lr_lambda_grid.yaml
```

### Ablation Studies

```bash
python experiments/scripts/run_sweep.py --config experiments/configs/ablations.yaml
```

### Baseline Comparisons

```bash
python experiments/scripts/run_sweep.py --config experiments/configs/baselines.yaml
```

## Unit Tests

```bash
# All tests
pytest tests/

# Specific test files
pytest tests/unit/test_newton_schulz.py -v
pytest tests/unit/test_spectral_filter.py -v
pytest tests/unit/test_optimizer_step.py -v

# With coverage
pytest tests/ --cov=soft_muon --cov-report=html
```

## Expected Results

### Lambda Sweep (Approximate)

| λ | Accuracy | Notes |
|---|----------|-------|
| 0.001 | ~93-94% | Near-Muon behavior |
| 0.01 | ~93-94% | |
| 0.1 | ~93-94% | Good balance |
| 1.0 | ~92-93% | More conservative |
| 10.0 | ~90-91% | Near gradient descent |

### Time Targets

- CIFAR-10 to 94%: Target < 5 seconds on A100
- Muon baseline: ~2.6 seconds

## Reproducing Specific Results

### Best SoftMuon Configuration

```bash
python benchmarks/cifar10/train_softmuon.py \
    --lr 0.02 \
    --momentum 0.95 \
    --lambda-reg 0.1 \
    --epochs 10 \
    --batch-size 512
```

### Comparing with Muon

To approximate Muon behavior, use very small λ:

```bash
python benchmarks/cifar10/train_softmuon.py --lambda-reg 0.001
```

## Output Format

Each experiment produces:

```
experiments/results/<sweep_name>/
├── <config>_seed42/
│   ├── config.json      # Experiment configuration
│   └── results.json     # Training results
├── <config>_seed123/
│   └── ...
└── sweep_summary.json   # Aggregate summary
```

Results JSON format:

```json
{
    "best_accuracy": 94.2,
    "final_accuracy": 94.0,
    "total_time": 45.3,
    "time_to_94": 38.1,
    "epochs": 10,
    "status": "success",
    "config": {...}
}
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python benchmarks/cifar10/train_softmuon.py --batch-size 256
```

### Slow Data Loading

Use fast loader (default) or reduce workers:
```bash
python benchmarks/cifar10/train_softmuon.py --num-workers 2
```

### Resume Incomplete Sweep

```bash
python experiments/scripts/run_sweep.py --config configs/lambda_sweep.yaml --resume
```
