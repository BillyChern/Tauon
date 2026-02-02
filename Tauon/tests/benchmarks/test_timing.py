"""
Performance benchmark tests for SoftMuon optimizer.
"""

import pytest
import torch
import torch.nn as nn
import time
from typing import Type

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from soft_muon import SoftMuon, SoftMuonConfig
from soft_muon.newton_schulz import coupled_newton_invsqrt, regularized_polar


def benchmark_optimizer(
    optimizer_class: Type,
    model: nn.Module,
    steps: int = 100,
    batch_size: int = 32,
    input_size: tuple = (3, 32, 32),
    warmup_steps: int = 10,
    **opt_kwargs,
) -> float:
    """
    Benchmark an optimizer.

    Returns time per step in milliseconds.
    """
    device = next(model.parameters()).device

    if optimizer_class == SoftMuon:
        config = SoftMuonConfig(**opt_kwargs)
        optimizer = optimizer_class(model.parameters(), config)
    else:
        optimizer = optimizer_class(model.parameters(), **opt_kwargs)

    # Warmup
    for _ in range(warmup_steps):
        x = torch.randn(batch_size, *input_size, device=device)
        optimizer.zero_grad()
        model(x).sum().backward()
        optimizer.step()

    # Benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start_time = time.perf_counter()

    for _ in range(steps):
        x = torch.randn(batch_size, *input_size, device=device)
        optimizer.zero_grad()
        model(x).sum().backward()
        optimizer.step()

    if device.type == "cuda":
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
    else:
        elapsed_ms = (time.perf_counter() - start_time) * 1000

    return elapsed_ms / steps


class SimpleMLP(nn.Module):
    """Simple MLP for benchmarking."""

    def __init__(self, input_size=784, hidden_size=256, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class TestOptimizationOverhead:
    """Test optimization overhead compared to baselines."""

    @pytest.fixture
    def mlp_model(self):
        return SimpleMLP()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for timing")
    def test_softmuon_vs_adamw(self, mlp_model):
        """Compare SoftMuon timing to AdamW."""
        device = torch.device("cuda")
        model_soft = SimpleMLP().to(device)
        model_adamw = SimpleMLP().to(device)

        time_soft = benchmark_optimizer(
            SoftMuon,
            model_soft,
            steps=50,
            input_size=(1, 28, 28),
            lr=0.01,
            lambda_reg=0.1,
        )

        time_adamw = benchmark_optimizer(
            torch.optim.AdamW,
            model_adamw,
            steps=50,
            input_size=(1, 28, 28),
            lr=0.001,
        )

        print(f"\nSoftMuon: {time_soft:.2f} ms/step")
        print(f"AdamW: {time_adamw:.2f} ms/step")
        print(f"Overhead: {(time_soft / time_adamw - 1) * 100:.1f}%")

        # SoftMuon will be slower due to matrix operations,
        # but should be within reasonable bounds
        # (exact threshold depends on matrix sizes)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_scaling_with_matrix_size(self):
        """Test how timing scales with matrix size."""
        device = torch.device("cuda")
        sizes = [64, 128, 256, 512]
        times = []

        for size in sizes:
            model = nn.Linear(size, size).to(device)
            config = SoftMuonConfig(lr=0.01, lambda_reg=0.1)
            optimizer = SoftMuon(model.parameters(), config)

            # Warmup
            for _ in range(10):
                x = torch.randn(32, size, device=device)
                optimizer.zero_grad()
                model(x).sum().backward()
                optimizer.step()

            # Time
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            for _ in range(50):
                x = torch.randn(32, size, device=device)
                optimizer.zero_grad()
                model(x).sum().backward()
                optimizer.step()

            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / 50)

        print("\nTiming by matrix size:")
        for size, t in zip(sizes, times):
            print(f"  {size}x{size}: {t:.2f} ms/step")


class TestNewtonSchulzTiming:
    """Benchmark Newton-Schulz iterations."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_coupled_newton_timing(self):
        """Benchmark coupled Newton iteration."""
        device = torch.device("cuda")
        sizes = [32, 64, 128, 256]

        print("\nCoupled Newton timing:")
        for n in sizes:
            # Create PSD matrix
            A = torch.randn(n, n // 2, device=device)
            B = A.T @ A + 0.1 * torch.eye(n // 2, device=device)

            # Warmup
            for _ in range(10):
                _ = coupled_newton_invsqrt(B, n_iters=5)

            # Time
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            for _ in range(100):
                _ = coupled_newton_invsqrt(B, n_iters=5)

            end.record()
            torch.cuda.synchronize()

            time_per_call = start.elapsed_time(end) / 100
            print(f"  {n // 2}x{n // 2}: {time_per_call:.3f} ms")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_regularized_polar_timing(self):
        """Benchmark regularized polar computation."""
        device = torch.device("cuda")
        shapes = [(64, 32), (128, 64), (256, 128), (512, 256)]

        print("\nRegularized polar timing:")
        for m, n in shapes:
            C = torch.randn(m, n, device=device)

            # Warmup
            for _ in range(10):
                _ = regularized_polar(C, lambda_reg=0.1, n_iters=5)

            # Time
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            for _ in range(100):
                _ = regularized_polar(C, lambda_reg=0.1, n_iters=5)

            end.record()
            torch.cuda.synchronize()

            time_per_call = start.elapsed_time(end) / 100
            print(f"  {m}x{n}: {time_per_call:.3f} ms")


class TestMemoryUsage:
    """Test memory usage."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_memory_per_parameter(self):
        """Measure memory overhead per parameter."""
        device = torch.device("cuda")

        torch.cuda.reset_peak_memory_stats()

        model = nn.Linear(1024, 1024, bias=False).to(device)
        param_bytes = model.weight.numel() * 4  # float32

        # Baseline memory
        torch.cuda.synchronize()
        baseline_mem = torch.cuda.max_memory_allocated()

        # Create optimizer
        config = SoftMuonConfig(lr=0.01, lambda_reg=0.1)
        optimizer = SoftMuon(model.parameters(), config)

        # Do a step to allocate buffers
        x = torch.randn(32, 1024, device=device)
        optimizer.zero_grad()
        model(x).sum().backward()
        optimizer.step()

        torch.cuda.synchronize()
        after_step_mem = torch.cuda.max_memory_allocated()

        overhead = after_step_mem - baseline_mem
        overhead_per_param = overhead / model.weight.numel()

        print(f"\nMemory overhead:")
        print(f"  Parameter size: {param_bytes / 1024:.1f} KB")
        print(f"  Total overhead: {overhead / 1024:.1f} KB")
        print(f"  Overhead per param: {overhead_per_param:.1f} bytes")

        # Should be reasonable (momentum buffer + some temporary)
        # Roughly 2x parameter size for momentum
        assert overhead < 4 * param_bytes, "Memory overhead too high"
