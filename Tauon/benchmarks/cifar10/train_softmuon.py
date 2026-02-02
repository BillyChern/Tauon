"""
CIFAR-10 Speedrun with SoftMuon optimizer.

Target: 94% accuracy
Hardware: Single A100/H100
Baseline: 2.59 seconds (Muon)
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from soft_muon import SoftMuon, SoftMuonConfig
from soft_muon.optimizer import CombinedOptimizer
from benchmarks.cifar10.model import make_net, count_parameters
from benchmarks.cifar10.data import get_cifar10_loaders, FastCIFAR10


@dataclass
class TrainConfig:
    """Training configuration."""

    # Model
    arch: str = "speedrun"
    num_classes: int = 10

    # Optimizer - SoftMuon
    lr: float = 0.02
    momentum: float = 0.95
    lambda_reg: float = 0.1
    lambda_mode: str = "fixed"
    ns_iters: int = 5
    weight_decay: float = 0.0

    # Optimizer - AdamW for non-softmuon params
    adamw_lr: float = 1e-3
    adamw_weight_decay: float = 0.0

    # Training
    epochs: int = 10
    batch_size: int = 512
    warmup_epochs: float = 0.5

    # Learning rate schedule
    lr_schedule: str = "cosine"  # 'cosine', 'linear', 'constant'

    # Data
    num_workers: int = 4
    data_dir: str = "./data"
    use_fast_loader: bool = True

    # Misc
    seed: int = 42
    device: str = "cuda"
    mixed_precision: bool = True
    compile_model: bool = False

    # Logging
    log_interval: int = 10
    save_checkpoint: bool = False
    checkpoint_dir: str = "./checkpoints"


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr_schedule(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler."""
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = int(config.warmup_epochs * steps_per_epoch)

    if config.lr_schedule == "cosine":

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif config.lr_schedule == "linear":

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 1 - progress

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    else:  # constant
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader,
    device: str,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels, reduction="sum")
        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / total

    model.train()
    return {"accuracy": accuracy, "loss": avg_loss}


def train(config: TrainConfig) -> Dict[str, Any]:
    """
    Train CIFAR-10 with SoftMuon optimizer.

    Args:
        config: Training configuration

    Returns:
        Dict with training results
    """
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # Model
    model = make_net(arch=config.arch, num_classes=config.num_classes).to(device)
    print(f"Model: {config.arch}, Parameters: {count_parameters(model):,}")

    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    # Data
    if config.use_fast_loader:
        data = FastCIFAR10(
            data_dir=config.data_dir,
            device=str(device),
            batch_size=config.batch_size,
        )
        steps_per_epoch = data.n_train // config.batch_size
        val_loader = list(data.get_val_batches())
    else:
        train_loader, val_loader = get_cifar10_loaders(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            data_dir=config.data_dir,
        )
        steps_per_epoch = len(train_loader)

    # Optimizer
    soft_config = SoftMuonConfig(
        lr=config.lr,
        momentum=config.momentum,
        lambda_reg=config.lambda_reg,
        lambda_mode=config.lambda_mode,
        ns_iters=config.ns_iters,
        weight_decay=config.weight_decay,
    )

    optimizer = CombinedOptimizer(
        model,
        soft_config,
        adamw_lr=config.adamw_lr,
        adamw_weight_decay=config.adamw_weight_decay,
    )

    # LR scheduler
    scheduler = get_lr_schedule(optimizer.soft_opt, config, steps_per_epoch)

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda") if config.mixed_precision else None

    # Training loop
    model.train()
    global_step = 0
    best_accuracy = 0.0
    time_to_94 = None

    # Timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    training_start = time.time()

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        if config.use_fast_loader:
            iterator = range(steps_per_epoch)
        else:
            iterator = train_loader

        for batch_idx, batch_data in enumerate(iterator):
            if config.use_fast_loader:
                images, labels = data.get_train_batch(augment=True)
            else:
                images, labels = batch_data
                images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            if config.mixed_precision:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = F.cross_entropy(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer.soft_opt)
                if optimizer.adamw_opt:
                    scaler.step(optimizer.adamw_opt)
                scaler.update()
            else:
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()

            scheduler.step()

            # Tracking
            epoch_loss += loss.item()
            pred = outputs.argmax(dim=1)
            epoch_correct += pred.eq(labels).sum().item()
            epoch_total += labels.size(0)
            global_step += 1

            if global_step % config.log_interval == 0:
                train_acc = 100.0 * epoch_correct / epoch_total
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"Step {global_step}, Loss: {loss.item():.4f}, "
                    f"Train Acc: {train_acc:.2f}%, LR: {current_lr:.6f}"
                )

        # End of epoch evaluation
        if config.use_fast_loader:
            val_metrics = evaluate(model, val_loader, str(device))
        else:
            val_metrics = evaluate(model, val_loader, str(device))

        print(
            f"Epoch {epoch + 1}/{config.epochs}, "
            f"Val Acc: {val_metrics['accuracy']:.2f}%, "
            f"Val Loss: {val_metrics['loss']:.4f}"
        )

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]

        if val_metrics["accuracy"] >= 94.0 and time_to_94 is None:
            end_event.record()
            torch.cuda.synchronize()
            time_to_94 = start_event.elapsed_time(end_event) / 1000  # seconds

    # Final timing
    end_event.record()
    torch.cuda.synchronize()
    total_time = start_event.elapsed_time(end_event) / 1000  # seconds

    results = {
        "best_accuracy": best_accuracy,
        "final_accuracy": val_metrics["accuracy"],
        "total_time": total_time,
        "time_to_94": time_to_94,
        "epochs": config.epochs,
        "config": asdict(config),
    }

    print(f"\nTraining complete!")
    print(f"Best accuracy: {best_accuracy:.2f}%")
    print(f"Total time: {total_time:.2f}s")
    if time_to_94:
        print(f"Time to 94%: {time_to_94:.2f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 with SoftMuon")

    # Model
    parser.add_argument("--arch", type=str, default="speedrun")

    # SoftMuon hyperparameters
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--lambda-reg", type=float, default=0.1)
    parser.add_argument("--lambda-mode", type=str, default="fixed")
    parser.add_argument("--ns-iters", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--warmup-epochs", type=float, default=0.5)
    parser.add_argument("--lr-schedule", type=str, default="cosine")

    # Data
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-fast-loader", action="store_true")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")

    # Output
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    # Build config
    config = TrainConfig(
        arch=args.arch,
        lr=args.lr,
        momentum=args.momentum,
        lambda_reg=args.lambda_reg,
        lambda_mode=args.lambda_mode,
        ns_iters=args.ns_iters,
        weight_decay=args.weight_decay,
        epochs=args.epochs if not args.smoke_test else 1,
        batch_size=args.batch_size,
        warmup_epochs=args.warmup_epochs,
        lr_schedule=args.lr_schedule,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        use_fast_loader=not args.no_fast_loader,
        seed=args.seed,
        mixed_precision=not args.no_mixed_precision,
        compile_model=args.compile,
    )

    # Run training
    results = train(config)

    # Save results if requested
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
