"""
Handcrafted material-counting evaluation model.

Uses standard piece values to compute a simple evaluation,
matching the same interface as neural models for comparison.
"""

import torch

from neuralchess.models.base import ChessModel

PIECE_VALUES = [1, 3, 3, 5, 9, 0]


class MaterialModel(ChessModel):
    def __init__(self) -> None:
        super().__init__()
        self._expected_input_shape = (14, 8, 8)

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        batch_size = position.size(0)
        scores = []

        for i in range(batch_size):
            board = position[i]
            white_material = 0.0
            black_material = 0.0

            for ch, value in enumerate(PIECE_VALUES):
                white_material += board[ch].sum().item() * value
                black_material += board[ch + 6].sum().item() * value

            material_diff = white_material - black_material
            turn = board[12, 0, 0].item()
            if turn == 0.0:
                material_diff = -material_diff

            scaled = float(torch.tanh(torch.tensor(material_diff / 6.0)))
            scores.append(scaled)

        return torch.tensor(scores, dtype=torch.float32).view(-1, 1)

    @property
    def expected_input_shape(self) -> tuple[int, ...]:
        return self._expected_input_shape
