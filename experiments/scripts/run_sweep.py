"""
Hyperparameter sweep runner for SoftMuon experiments.

Usage:
    python run_sweep.py --config configs/lambda_sweep.yaml
"""

import argparse
import itertools
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_sweep_configs(config: Dict) -> List[Dict]:
    """Generate all configurations from sweep specification."""
    base_config = config.get("base_config", {})
    sweep_spec = config.get("sweep", {})
    seeds = config.get("seeds", [42])

    # Generate all parameter combinations
    param_names = list(sweep_spec.keys())
    param_values = [sweep_spec[name]["values"] for name in param_names]

    configs = []
    for values in itertools.product(*param_values):
        for seed in seeds:
            run_config = base_config.copy()
            for name, value in zip(param_names, values):
                run_config[name] = value
            run_config["seed"] = seed
            configs.append(run_config)

    return configs


def run_single_experiment(
    run_config: Dict,
    benchmark: str,
    output_dir: str,
    dry_run: bool = False,
) -> Dict:
    """Run a single experiment with given configuration."""
    # Create unique run name
    param_str = "_".join(f"{k}={v}" for k, v in sorted(run_config.items()) if k != "seed")
    run_name = f"{param_str}_seed{run_config['seed']}"
    run_dir = Path(output_dir) / run_name

    if dry_run:
        print(f"[DRY RUN] Would run: {run_name}")
        print(f"  Config: {run_config}")
        return {"status": "dry_run", "config": run_config}

    # Create output directory
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"Running: {run_name}")

    if benchmark == "cifar10":
        from benchmarks.cifar10.train_softmuon import train, TrainConfig

        # Convert dict to TrainConfig
        train_config = TrainConfig(**{k: v for k, v in run_config.items() if hasattr(TrainConfig, k)})

        try:
            results = train(train_config)
            results["status"] = "success"
        except Exception as e:
            results = {"status": "failed", "error": str(e)}
            print(f"  Failed: {e}")
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    # Save results
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_sweep(config_path: str, dry_run: bool = False, resume: bool = False):
    """Run full hyperparameter sweep."""
    config = load_config(config_path)

    sweep_name = config.get("sweep_name", "unnamed_sweep")
    benchmark = config.get("benchmark", "cifar10")
    output_dir = config.get("output_dir", f"experiments/results/{sweep_name}")

    print(f"Sweep: {sweep_name}")
    print(f"Benchmark: {benchmark}")
    print(f"Output: {output_dir}")

    # Generate all configs
    run_configs = generate_sweep_configs(config)
    print(f"Total runs: {len(run_configs)}")

    # Check for completed runs if resuming
    completed = set()
    if resume and Path(output_dir).exists():
        for run_dir in Path(output_dir).iterdir():
            if run_dir.is_dir() and (run_dir / "results.json").exists():
                completed.add(run_dir.name)
        print(f"Resuming: {len(completed)} runs already completed")

    # Run experiments
    all_results = []
    for i, run_config in enumerate(run_configs):
        param_str = "_".join(f"{k}={v}" for k, v in sorted(run_config.items()) if k != "seed")
        run_name = f"{param_str}_seed{run_config['seed']}"

        if run_name in completed:
            print(f"Skipping completed: {run_name}")
            continue

        print(f"\n[{i + 1}/{len(run_configs)}] Running experiment...")
        results = run_single_experiment(
            run_config,
            benchmark,
            output_dir,
            dry_run=dry_run,
        )
        results["run_name"] = run_name
        results["config"] = run_config
        all_results.append(results)

    # Save sweep summary
    summary = {
        "sweep_name": sweep_name,
        "benchmark": benchmark,
        "timestamp": datetime.now().isoformat(),
        "total_runs": len(run_configs),
        "completed_runs": len(all_results),
        "results": all_results,
    }

    summary_path = Path(output_dir) / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSweep complete. Summary saved to {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep")
    parser.add_argument("--config", type=str, required=True, help="Path to sweep config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Print configs without running")
    parser.add_argument("--resume", action="store_true", help="Resume incomplete sweep")

    args = parser.parse_args()

    run_sweep(args.config, dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()
