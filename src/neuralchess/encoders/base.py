"""
Abstract base class for position encoders.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch


class PositionEncoder(ABC):
    @abstractmethod
    def encode_position(self, fen: str) -> torch.Tensor: ...

    def encode_batch(self, fens: list[str]) -> np.ndarray:
        return np.stack([self.encode_position(fen).numpy() for fen in fens])

    @property
    @abstractmethod
    def output_shape(self) -> tuple[int, ...]: ...
