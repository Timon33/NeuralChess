"""
Convolutional neural network for chess position evaluation.

Architecture inspired by LC0's residual tower and Stockfish NNUE design principles:
- Full spatial resolution (no pooling) to preserve piece-square relationships
- Residual blocks for gradient flow through deeper networks
- Global average pooling instead of flattening to force channel-level feature learning
- Small FC head for final evaluation
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn

from neuralchess.models.base import ChessModel


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + identity
        return self.relu(out)


@dataclass
class CNNConfig:
    input_channels: int = 14
    entry_channels: int = 128
    residual_channels: int = 128
    residual_blocks: int = 4
    fc_hidden: tuple[int, ...] = (256,)
    kernel_size: int = 3


class NeuralChessNet(ChessModel):
    def __init__(self, config: Optional[CNNConfig] = None) -> None:
        super().__init__()
        self.config = config or CNNConfig()
        self._expected_input_shape = (self.config.input_channels, 8, 8)

        entry_layers: list[nn.Module] = [
            nn.Conv2d(
                self.config.input_channels,
                self.config.entry_channels,
                kernel_size=self.config.kernel_size,
                padding=self.config.kernel_size // 2,
            ),
            nn.ReLU(),
        ]
        self.entry = nn.Sequential(*entry_layers)

        if self.config.entry_channels != self.config.residual_channels:
            self.bridge = nn.Conv2d(
                self.config.entry_channels,
                self.config.residual_channels,
                kernel_size=1,
            )
        else:
            self.bridge = nn.Identity()

        self.residual_tower = nn.Sequential(
            *[
                ResidualBlock(self.config.residual_channels)
                for _ in range(self.config.residual_blocks)
            ]
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        fc_layers: list[nn.Module] = []
        prev = self.config.residual_channels
        for hidden in self.config.fc_hidden:
            fc_layers.append(nn.Linear(prev, hidden))
            fc_layers.append(nn.ReLU())
            prev = hidden
        fc_layers.append(nn.Linear(prev, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        x = self.entry(position)
        x = self.bridge(x)
        x = self.residual_tower(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        return torch.tanh(self.fc(x))

    @property
    def expected_input_shape(self) -> tuple[int, ...]:
        return self._expected_input_shape
