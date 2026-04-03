import re

import chess
import numpy as np
import torch

from neuralchess.encoders.base import PositionEncoder

class TokenEncoder(PositionEncoder):

    def encode_position(self, fen: str) -> torch.Tensor:
        board = chess.Board(fen)
        tokens = np.zeros(70, dtype=np.int64)

        # 1. 64 squares for the board (0 to 12)
        for i in range(64):
            piece = board.piece_at(i)
            if piece:
                # piece.color is bool (White=True, Black=False)
                offset = 0 if piece.color else 6
                tokens[i] = piece.piece_type + offset  # piece_type is 1 to 6

        # 2. Turn (13 or 14)
        tokens[64] = 13 if board.turn else 14

        # 3. Castling Rights (15 or 16)
        tokens[65] = 15 if board.has_kingside_castling_rights(chess.WHITE) else 16
        tokens[66] = 15 if board.has_queenside_castling_rights(chess.WHITE) else 16
        tokens[67] = 15 if board.has_kingside_castling_rights(chess.BLACK) else 16
        tokens[68] = 15 if board.has_queenside_castling_rights(chess.BLACK) else 16

        # 4. En Passant (0 to 64, map to 17 to 81 to keep vocab unique, or just reuse indices)
        # Reusing indices is fine because the transformer knows the position via pos_emb
        tokens[69] = board.ep_square if board.ep_square is not None else 64

        return torch.from_numpy(tokens)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (70, )
