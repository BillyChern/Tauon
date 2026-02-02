"""
Model architectures for CIFAR-10 speedrun benchmarking.

Based on cifar10-airbench architectures optimized for fast training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConvBN(nn.Module):
    """Convolution + BatchNorm + optional activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        activation: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = F.gelu(x)
        return x


class ResBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBN(channels, channels)
        self.conv2 = ConvBN(channels, channels, activation=False)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        x = F.gelu(x)
        return x


class SpeedrunNet(nn.Module):
    """
    Fast CIFAR-10 network architecture.

    Based on cifar10-airbench speedrun models, designed for:
    - Fast forward/backward passes
    - Good accuracy with minimal epochs
    - Efficient memory usage

    Architecture:
    - Initial conv: 3 -> 64 channels
    - Stage 1: 64 channels, 2 res blocks
    - Stage 2: 128 channels, 2 res blocks (with stride-2 downsample)
    - Stage 3: 256 channels, 2 res blocks (with stride-2 downsample)
    - Global average pool -> linear classifier
    """

    def __init__(
        self,
        num_classes: int = 10,
        base_channels: int = 64,
        num_blocks: tuple = (2, 2, 2),
    ):
        super().__init__()

        # Initial convolution
        self.stem = ConvBN(3, base_channels, kernel_size=3, padding=1)

        # Build stages
        channels = [base_channels, base_channels * 2, base_channels * 4]

        self.stage1 = self._make_stage(channels[0], channels[0], num_blocks[0], stride=1)
        self.stage2 = self._make_stage(channels[0], channels[1], num_blocks[1], stride=2)
        self.stage3 = self._make_stage(channels[1], channels[2], num_blocks[2], stride=2)

        # Classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels[2], num_classes)

        # Initialize weights
        self._init_weights()

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Create a stage with optional downsampling."""
        layers = []

        # Downsample if needed
        if stride > 1 or in_channels != out_channels:
            layers.append(
                ConvBN(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            )
        else:
            layers.append(ConvBN(in_channels, out_channels))

        # Residual blocks
        for _ in range(num_blocks):
            layers.append(ResBlock(out_channels))

        return nn.Sequential(*layers)

    def _init_weights(self):
        """Initialize weights for fast convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


class MiniNet(nn.Module):
    """
    Minimal network for fast testing.

    Much smaller than SpeedrunNet, useful for quick iteration
    during development and debugging.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBN(3, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            ConvBN(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            ConvBN(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


def make_net(
    arch: str = "speedrun",
    num_classes: int = 10,
    **kwargs,
) -> nn.Module:
    """
    Factory function for creating networks.

    Args:
        arch: Architecture name ('speedrun', 'mini')
        num_classes: Number of output classes
        **kwargs: Additional arguments for the architecture

    Returns:
        Network module
    """
    if arch == "speedrun":
        return SpeedrunNet(num_classes=num_classes, **kwargs)
    elif arch == "mini":
        return MiniNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float = 0.0,
) -> list:
    """
    Get parameter groups with optional weight decay exclusion for norms/biases.

    Args:
        model: The model
        weight_decay: Weight decay value for applicable parameters

    Returns:
        List of parameter group dicts
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bn" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
