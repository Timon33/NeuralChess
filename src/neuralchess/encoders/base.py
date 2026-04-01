"""
Abstract base class for position encoders.
"""

from abc import ABC, abstractmethod

import torch


class PositionEncoder(ABC):
    @abstractmethod
    def encode(self, fen: str) -> torch.Tensor: ...

    @property
    @abstractmethod
    def output_shape(self) -> tuple[int, ...]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...
