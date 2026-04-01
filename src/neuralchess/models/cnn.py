"""
Convolutional neural network for chess position evaluation.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn

from neuralchess.models.base import ChessModel


@dataclass
class CNNConfig:
    input_channels: int = 14
    conv_channels: tuple[int, ...] = (64, 128, 128)
    kernel_size: int = 3
    pool_after_first: bool = True
    fc_hidden: tuple[int, ...] = (256, 64)
    dropout: float = 0.0


class NeuralChessNet(ChessModel):
    def __init__(self, config: Optional[CNNConfig] = None) -> None:
        super().__init__()
        self.config = config or CNNConfig()
        self._expected_input_shape = (self.config.input_channels, 8, 8)

        conv_layers: list[nn.Module] = []
        in_ch = self.config.input_channels
        for i, out_ch in enumerate(self.config.conv_channels):
            conv_layers.append(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=self.config.kernel_size,
                    padding=self.config.kernel_size // 2,
                )
            )
            conv_layers.append(nn.BatchNorm2d(out_ch))
            conv_layers.append(nn.ReLU())
            if i == 0 and self.config.pool_after_first:
                conv_layers.append(nn.MaxPool2d(2))
            in_ch = out_ch

        self.conv = nn.Sequential(*conv_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, *self._expected_input_shape)
            flat_dim = self.conv(dummy).view(1, -1).size(1)

        fc_layers: list[nn.Module] = []
        prev = flat_dim
        for hidden in self.config.fc_hidden:
            fc_layers.append(nn.Linear(prev, hidden))
            if self.config.dropout > 0.0:
                fc_layers.append(nn.Dropout(self.config.dropout))
            fc_layers.append(nn.ReLU())
            prev = hidden
        fc_layers.append(nn.Linear(prev, 1))

        self.fc = nn.Sequential(*fc_layers)

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        x = self.conv(position)
        x = x.view(x.size(0), -1)
        return torch.tanh(self.fc(x))

    @property
    def expected_input_shape(self) -> tuple[int, ...]:
        return self._expected_input_shape
