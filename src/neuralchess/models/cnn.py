"""
Convolutional neural network for chess position evaluation.

Architecture inspired by LC0's residual tower and Stockfish NNUE design principles:
- Full spatial resolution (no pooling) to preserve piece-square relationships
- Residual blocks for gradient flow through deeper networks
- Global average pooling instead of flattening to force channel-level feature learning
- Small FC head for final evaluation
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn

from neuralchess.encoders import get_encoder
from neuralchess.models import ChessModel
from neuralchess.models.base import ChessModel, ModelConfig


class ResidualBlock(nn.Module):
    """Residual block with batch normalization for stable training."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


@dataclass
class CNNConfig(ModelConfig):
    encoder_name: str = "bitboard"
    input_channels: int = 14
    entry_channels: int = 64
    residual_channels: int = 64
    residual_blocks: int = 6
    fc_hidden: tuple[int, ...] = (256, 128)
    kernel_size: int = 3


class NeuralChessNet(ChessModel):
    def __init__(self, config: Optional[CNNConfig] = None) -> None:
        super().__init__()
        self._config = config or CNNConfig()
        self._encoder = get_encoder(self._config.encoder_name)
        self._device = torch.device("cpu")

        entry_layers: list[nn.Module] = [
            nn.Conv2d(
                self._config.input_channels,
                self._config.entry_channels,
                kernel_size=self._config.kernel_size,
                padding=self._config.kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(self._config.entry_channels),
            nn.ReLU(),
        ]
        self.entry = nn.Sequential(*entry_layers)

        if self._config.entry_channels != self._config.residual_channels:
            self.bridge = nn.Conv2d(
                self._config.entry_channels,
                self._config.residual_channels,
                kernel_size=1,
            )
        else:
            self.bridge = nn.Identity()

        self.residual_tower = nn.Sequential(
            *[
                ResidualBlock(self._config.residual_channels)
                for _ in range(self._config.residual_blocks)
            ]
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        fc_layers: list[nn.Module] = []
        prev = self._config.residual_channels
        for hidden in self._config.fc_hidden:
            fc_layers.append(nn.Linear(prev, hidden))
            fc_layers.append(nn.ReLU())
            prev = hidden
        fc_layers.append(nn.Linear(prev, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.entry(x)
        x = self.bridge(x)
        x = self.residual_tower(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        return torch.sigmoid(self.fc(x))

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str, device: torch.device
    ) -> "NeuralChessNet":
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model_config = checkpoint.get("model_config", {})
        config = CNNConfig(**model_config) if model_config else CNNConfig()
        model = cls(config)

        state = checkpoint["model_state"]
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        model.to(device)
        return model

    @property
    def config(self) -> CNNConfig:
        return self._config

    @property
    def expected_input_shape(self) -> tuple[int, ...]:
        return self._encoder.output_shape
