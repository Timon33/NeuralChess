"""
Architecture-agnostic dataset for loading precomputed position tensors and evaluations.
"""

import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    def __init__(self, data_dir: str, max_positions: Optional[int] = None) -> None:
        tensors_path = os.path.join(data_dir, "tensors.npy")
        evals_path = os.path.join(data_dir, "evals.npy")

        self.tensors = np.load(tensors_path, mmap_mode="r")
        self.evals = np.load(evals_path, mmap_mode="r")

        if max_positions is not None:
            self.tensors = self.tensors[:max_positions]
            self.evals = self.evals[:max_positions]

    def __len__(self) -> int:
        return len(self.evals)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tensor = torch.from_numpy(self.tensors[idx].copy())
        eval_val = torch.tensor(self.evals[idx], dtype=torch.float32)
        return tensor, eval_val
