"""
Analyze experiment results and generate plots.

Usage:
    python analyze_results.py --results experiments/results/lambda_sweep
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def load_results(results_dir: str) -> List[Dict]:
    """Load all results from a sweep directory."""
    results = []
    results_path = Path(results_dir)

    for run_dir in results_path.iterdir():
        if run_dir.is_dir():
            results_file = run_dir / "results.json"
            config_file = run_dir / "config.json"

            if results_file.exists():
                with open(results_file, "r") as f:
                    result = json.load(f)
                if config_file.exists():
                    with open(config_file, "r") as f:
                        result["config"] = json.load(f)
                result["run_name"] = run_dir.name
                results.append(result)

    return results


def aggregate_by_param(
    results: List[Dict],
    param_name: str,
    metric: str = "best_accuracy",
) -> Dict:
    """Aggregate results by a parameter value."""
    aggregated = {}

    for result in results:
        if result.get("status") != "success":
            continue

        config = result.get("config", {})
        param_value = config.get(param_name)

        if param_value is None:
            continue

        if param_value not in aggregated:
            aggregated[param_value] = []

        metric_value = result.get(metric)
        if metric_value is not None:
            aggregated[param_value].append(metric_value)

    # Compute statistics
    stats = {}
    for param_value, values in aggregated.items():
        if values:
            stats[param_value] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "n": len(values),
            }

    return stats


def print_lambda_analysis(results: List[Dict]):
    """Print analysis of lambda sweep results."""
    print("\n" + "=" * 60)
    print("LAMBDA SWEEP ANALYSIS")
    print("=" * 60)

    # Aggregate by lambda
    acc_stats = aggregate_by_param(results, "lambda_reg", "best_accuracy")
    time_stats = aggregate_by_param(results, "lambda_reg", "total_time")
    time_94_stats = aggregate_by_param(results, "lambda_reg", "time_to_94")

    # Sort by lambda value
    lambda_values = sorted(acc_stats.keys())

    print("\nAccuracy by λ:")
    print("-" * 50)
    print(f"{'λ':>10} {'Mean Acc':>12} {'Std':>10} {'N':>5}")
    print("-" * 50)

    for lam in lambda_values:
        stats = acc_stats[lam]
        print(f"{lam:>10.4f} {stats['mean']:>12.2f}% {stats['std']:>10.2f} {stats['n']:>5}")

    # Find best lambda
    best_lam = max(lambda_values, key=lambda x: acc_stats[x]["mean"])
    print(f"\nBest λ: {best_lam} (accuracy: {acc_stats[best_lam]['mean']:.2f}%)")

    if time_94_stats:
        print("\nTime to 94% by λ:")
        print("-" * 50)
        for lam in lambda_values:
            if lam in time_94_stats:
                stats = time_94_stats[lam]
                print(f"λ={lam:.4f}: {stats['mean']:.2f}s ± {stats['std']:.2f}s")


def print_grid_analysis(results: List[Dict]):
    """Print analysis of grid search results."""
    print("\n" + "=" * 60)
    print("GRID SEARCH ANALYSIS")
    print("=" * 60)

    # Get unique values
    lr_values = sorted(set(r["config"].get("lr") for r in results if r.get("status") == "success"))
    lambda_values = sorted(
        set(r["config"].get("lambda_reg") for r in results if r.get("status") == "success")
    )

    # Build accuracy matrix
    print("\nAccuracy Matrix (LR x Lambda):")
    print("-" * 60)

    header = "LR \\ λ   " + "  ".join(f"{lam:>8.3f}" for lam in lambda_values)
    print(header)
    print("-" * 60)

    for lr in lr_values:
        row = f"{lr:>8.4f}"
        for lam in lambda_values:
            # Find matching results
            matching = [
                r
                for r in results
                if r.get("status") == "success"
                and r["config"].get("lr") == lr
                and r["config"].get("lambda_reg") == lam
            ]
            if matching:
                mean_acc = np.mean([r["best_accuracy"] for r in matching])
                row += f"  {mean_acc:>8.2f}"
            else:
                row += f"  {'N/A':>8}"
        print(row)


def generate_summary_report(results: List[Dict], output_path: str):
    """Generate a summary report."""
    successful = [r for r in results if r.get("status") == "success"]

    report = {
        "total_runs": len(results),
        "successful_runs": len(successful),
        "failed_runs": len(results) - len(successful),
    }

    if successful:
        accuracies = [r["best_accuracy"] for r in successful]
        report["best_accuracy"] = max(accuracies)
        report["mean_accuracy"] = np.mean(accuracies)
        report["std_accuracy"] = np.std(accuracies)

        times_to_94 = [r["time_to_94"] for r in successful if r.get("time_to_94")]
        if times_to_94:
            report["fastest_to_94"] = min(times_to_94)
            report["mean_time_to_94"] = np.mean(times_to_94)

        # Find best config
        best_result = max(successful, key=lambda r: r["best_accuracy"])
        report["best_config"] = best_result.get("config")

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSummary report saved to {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--results", type=str, required=True, help="Path to results directory")
    parser.add_argument("--output", type=str, default=None, help="Output path for summary report")
    parser.add_argument(
        "--type", type=str, default="lambda", choices=["lambda", "grid", "all"], help="Analysis type"
    )

    args = parser.parse_args()

    results = load_results(args.results)
    print(f"Loaded {len(results)} results from {args.results}")

    if args.type == "lambda" or args.type == "all":
        print_lambda_analysis(results)

    if args.type == "grid" or args.type == "all":
        print_grid_analysis(results)

    if args.output:
        generate_summary_report(results, args.output)
    else:
        # Default output path
        output_path = Path(args.results) / "analysis_summary.json"
        generate_summary_report(results, str(output_path))


if __name__ == "__main__":
    main()
