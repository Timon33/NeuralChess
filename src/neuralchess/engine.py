"""
Neural chess engine with alpha-beta search, transposition table, and batched inference.
"""

import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import chess
import torch

from neuralchess.encoders import get_encoder
from neuralchess.encoders.base import PositionEncoder
from neuralchess.models.base import ChessModel
from neuralchess.zobrist import ZobristHasher


class SearchTimeout(Exception):
    pass


class TTFlag(Enum):
    EXACT = 0
    ALPHA = 1
    BETA = 2


@dataclass
class TTEntry:
    depth: int
    score: float
    flag: TTFlag
    best_move: Optional[chess.Move]


class NeuralEngine:
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        max_tt_size: int = 1_000_000,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        self.model = self._build_model_from_checkpoint(checkpoint)
        self.model.eval()

        self.encoder: PositionEncoder = get_encoder(
            checkpoint.get("encoder_name", "bitboard")
        )

        self._zobrist = ZobristHasher()
        self.tt: OrderedDict[int, TTEntry] = OrderedDict()
        self.max_tt_size = max_tt_size

        self.nodes_searched: int = 0
        self._time_check_interval: int = 1024

    @staticmethod
    def _build_model_from_checkpoint(checkpoint: dict) -> ChessModel:
        from neuralchess.models import CNNConfig, NeuralChessNet

        model_type = checkpoint.get("model_type", "cnn")
        model_config = checkpoint.get("model_config", {})

        if model_type == "cnn":
            config = CNNConfig(**model_config) if model_config else CNNConfig()
            model: ChessModel = NeuralChessNet(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        state = checkpoint["model_state"]
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        return model

    def search(
        self,
        board: chess.Board,
        movetime_ms: Optional[int] = None,
        wtime_ms: Optional[int] = None,
        btime_ms: Optional[int] = None,
        max_depth: int = 64,
    ) -> tuple[Optional[chess.Move], float, list[chess.Move]]:
        if movetime_ms is not None:
            time_limit_ms = movetime_ms
        elif wtime_ms is not None and btime_ms is not None:
            remaining = wtime_ms if board.turn == chess.WHITE else btime_ms
            time_limit_ms = min(remaining, 30000) // 30
        else:
            time_limit_ms = 1000

        time_limit_ms = max(time_limit_ms, 10)

        start_time = time.time()
        best_move: Optional[chess.Move] = None
        best_score = 0.0
        best_pv: list[chess.Move] = []

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, 0.0, []

        if len(legal_moves) == 1:
            score = self._evaluate_position(board.fen())
            return legal_moves[0], score, [legal_moves[0]]

        self.tt.clear()
        self.nodes_searched = 0

        for depth in range(1, max_depth + 1):
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > time_limit_ms:
                break

            move, score, pv = self._search_root(
                board, depth, legal_moves, start_time, time_limit_ms
            )

            if move is not None:
                best_move = move
                best_score = score
                best_pv = pv

        return best_move, best_score, best_pv

    def _search_root(
        self,
        board: chess.Board,
        depth: int,
        legal_moves: list[chess.Move],
        start_time: float,
        time_limit_ms: int,
    ) -> tuple[Optional[chess.Move], float, list[chess.Move]]:
        best_move: Optional[chess.Move] = None
        best_score = -math.inf
        best_pv: list[chess.Move] = []

        ordered_moves = self._order_moves(board, legal_moves)

        for move in ordered_moves:
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > time_limit_ms:
                break

            board.push(move)
            try:
                score, pv = self._alphabeta(
                    board, depth - 1, -math.inf, math.inf, start_time, time_limit_ms
                )
            except SearchTimeout:
                board.pop()
                break
            score = -score
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
                best_pv = [move] + pv

        if best_move is None:
            return None, 0.0, []

        return best_move, best_score, best_pv

    def _alphabeta(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        start_time: float,
        time_limit_ms: int,
    ) -> tuple[float, list[chess.Move]]:
        self.nodes_searched += 1

        if self.nodes_searched % self._time_check_interval == 0:
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > time_limit_ms:
                raise SearchTimeout()

        if board.is_game_over():
            return self._game_over_score(board), []

        tt_entry = self.tt.get(self._zobrist.hash_board(board))
        if tt_entry is not None and tt_entry.depth >= depth:
            score = self._tt_score(tt_entry, alpha, beta)
            if score is not None:
                return score, self._tt_pv(tt_entry, board)

        if depth == 0:
            return self._evaluate_position(board.fen()), []

        legal_moves = list(board.legal_moves)

        tt_best = tt_entry.best_move if tt_entry else None
        ordered_moves = self._order_moves(board, legal_moves, tt_best)

        best_score = -math.inf
        best_move: Optional[chess.Move] = None
        best_pv: list[chess.Move] = []
        flag = TTFlag.ALPHA

        for move in ordered_moves:
            board.push(move)
            try:
                score, pv = self._alphabeta(
                    board, depth - 1, -beta, -alpha, start_time, time_limit_ms
                )
            except SearchTimeout:
                board.pop()
                raise
            score = -score
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
                best_pv = [move] + pv

                if score > alpha:
                    alpha = score
                    flag = TTFlag.EXACT

                    if score >= beta:
                        flag = TTFlag.BETA
                        break

        self._store_tt(
            self._zobrist.hash_board(board), depth, best_score, flag, best_move
        )

        return best_score, best_pv

    def _evaluate_position(self, fen: str) -> float:
        tensor = self.encoder.encode(fen).unsqueeze(0).to(self.device)
        with torch.no_grad():
            score = self.model(tensor).item()
        return float(score)

    def _order_moves(
        self,
        board: chess.Board,
        moves: list[chess.Move],
        tt_best: Optional[chess.Move] = None,
    ) -> list[chess.Move]:
        def move_score(move: chess.Move) -> int:
            if tt_best is not None and move == tt_best:
                return 100_000

            score = 0
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                victim_value = (
                    self._piece_value(victim)
                    if victim
                    else (1 if board.is_en_passant(move) else 0)
                )
                attacker_value = self._piece_value(attacker) if attacker else 0
                score = 10 * victim_value - attacker_value

            if move.promotion:
                score += 900

            return score

        return sorted(moves, key=move_score, reverse=True)

    @staticmethod
    def _piece_value(piece: Optional[chess.Piece]) -> int:
        if piece is None:
            return 0
        return {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0,
        }.get(piece.piece_type, 0)

    def _store_tt(
        self,
        zobrist: int,
        depth: int,
        score: float,
        flag: TTFlag,
        best_move: Optional[chess.Move],
    ) -> None:
        if len(self.tt) >= self.max_tt_size:
            self.tt.popitem(last=False)

        self.tt[zobrist] = TTEntry(
            depth=depth, score=score, flag=flag, best_move=best_move
        )

    @staticmethod
    def _tt_score(entry: TTEntry, alpha: float, beta: float) -> Optional[float]:
        if entry.flag == TTFlag.EXACT:
            return entry.score
        if entry.flag == TTFlag.ALPHA and entry.score <= alpha:
            return entry.score
        if entry.flag == TTFlag.BETA and entry.score >= beta:
            return entry.score
        return None

    @staticmethod
    def _tt_pv(entry: TTEntry, board: chess.Board) -> list[chess.Move]:
        if entry.best_move is not None and entry.best_move in board.legal_moves:
            return [entry.best_move]
        return []

    @staticmethod
    def _game_over_score(board: chess.Board) -> float:
        if board.is_checkmate():
            return -1.0
        return 0.0
