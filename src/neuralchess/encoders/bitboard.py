"""
CNN-specific bitboard encoder: FEN → (14, 8, 8) tensor.

Channels 0–5: White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
Channels 6–11: Black pieces (Pawn, Knight, Bishop, Rook, Queen, King)
Channel 12: Side to move (all 1s = white, all 0s = black)
Channel 13: Castling availability (1s on squares with castling rights)
"""

import re

import numpy as np
import torch

from neuralchess.encoders.base import PositionEncoder

PIECE_TO_CHANNEL: dict[str, int] = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}

CASTLING_SQUARES: dict[str, list[int]] = {
    "K": [60, 63],
    "Q": [60, 56],
    "k": [4, 7],
    "q": [4, 0],
}

_FEN_BOARD_RE = re.compile(r"^([1-8kqrbnpKQRBNP/]+)\s+([wb])\s+([KQkq-]+)\s")


class BitboardEncoder(PositionEncoder):
    def encode_position(self, fen: str) -> torch.Tensor:
        parts = fen.split()
        board_str = parts[0]
        turn = parts[1]
        castling = parts[2]

        tensor = torch.zeros(14, 8, 8, dtype=torch.float32)

        row, col = 0, 0
        for char in board_str:
            if char == "/":
                row += 1
                col = 0
            elif char.isdigit():
                col += int(char)
            else:
                channel = PIECE_TO_CHANNEL[char]
                tensor[channel, row, col] = 1.0
                col += 1

        if turn == "w":
            tensor[12, :, :] = 1.0

        for castle_char in castling:
            if castle_char in CASTLING_SQUARES:
                for sq in CASTLING_SQUARES[castle_char]:
                    tensor[13, sq // 8, sq % 8] = 1.0

        return tensor

    @property
    def output_shape(self) -> tuple[int, ...]:
        return 14, 8, 8
