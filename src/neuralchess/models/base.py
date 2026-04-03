"""
Abstract base class for chess evaluation models.
"""

from abc import abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    """Base configuration for all chess models."""

    encoder_name: str


class ChessModel(nn.Module):
    """Abstract base class for chess evaluation models."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def evaluate(self, fens: list[str]) -> list[float]:
        """Batched evaluation: list of FEN strings to list of scores.

        Models handle their own encoding internally.
        """
        raise NotImplementedError

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str, device: torch.device
    ) -> "ChessModel":
        """Reconstruct and load model from checkpoint file."""
        raise NotImplementedError

    @property
    @abstractmethod
    def config(self) -> ModelConfig: ...

    @property
    @abstractmethod
    def expected_input_shape(self) -> tuple[int, ...]: ...
