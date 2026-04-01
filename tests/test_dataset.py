import os
import tempfile

import numpy as np
import pytest
import torch

from neuralchess.core.dataset import ChessDataset


class TestChessDataset:
    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        num_samples = 10
        tensors = np.random.rand(num_samples, 14, 8, 8).astype(np.float32)
        evals = np.random.rand(num_samples).astype(np.float32) * 2 - 1
        np.save(os.path.join(self.tmpdir, "tensors.npy"), tensors)
        np.save(os.path.join(self.tmpdir, "evals.npy"), evals)

    def test_length(self) -> None:
        dataset = ChessDataset(self.tmpdir)
        assert len(dataset) == 10

    def test_tensor_shape(self) -> None:
        dataset = ChessDataset(self.tmpdir)
        tensor, _ = dataset[0]
        assert tensor.shape == (14, 8, 8)

    def test_max_positions(self) -> None:
        dataset = ChessDataset(self.tmpdir, max_positions=5)
        assert len(dataset) == 5
