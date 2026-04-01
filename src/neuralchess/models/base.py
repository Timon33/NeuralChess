"""
Abstract base class for chess neural network models.
"""

from abc import ABC, abstractmethod

import torch
from torch import nn


class ChessModel(ABC, nn.Module):
    @abstractmethod
    def forward(self, position: torch.Tensor) -> torch.Tensor: ...

    @property
    @abstractmethod
    def expected_input_shape(self) -> tuple[int, ...]: ...
