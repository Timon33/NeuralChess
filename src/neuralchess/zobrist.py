"""
Zobrist hashing for chess positions.
"""

import random

import chess


PIECE_TO_INDEX: dict[int, int] = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


class ZobristHasher:
    def __init__(self, seed: int = 42) -> None:
        rng = random.Random(seed)
        self._piece_table: list[list[int]] = [
            [rng.getrandbits(64) for _ in range(64)] for _ in range(12)
        ]
        self._castling_table: list[int] = [rng.getrandbits(64) for _ in range(16)]
        self._ep_table: list[int] = [rng.getrandbits(64) for _ in range(64)]
        self._side: int = rng.getrandbits(64)

    def hash_board(self, board: chess.Board) -> int:
        h = 0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is not None:
                color_idx = 0 if piece.color == chess.WHITE else 6
                piece_idx = PIECE_TO_INDEX[piece.piece_type]
                h ^= self._piece_table[color_idx + piece_idx][sq]

        castling_index = 0
        if board.has_kingside_castling_rights(chess.WHITE):
            castling_index |= 1
        if board.has_queenside_castling_rights(chess.WHITE):
            castling_index |= 2
        if board.has_kingside_castling_rights(chess.BLACK):
            castling_index |= 4
        if board.has_queenside_castling_rights(chess.BLACK):
            castling_index |= 8
        h ^= self._castling_table[castling_index]

        if board.ep_square is not None:
            h ^= self._ep_table[board.ep_square]

        if board.turn == chess.BLACK:
            h ^= self._side

        return h
