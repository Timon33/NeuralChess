"""
Neural chess engine with alpha-beta search and batched inference.
"""

import logging
from typing import Optional, List, Tuple

import chess
import torch

from neuralchess.models.base import ChessModel

logger = logging.getLogger(__name__)

class NeuralEngine:
    def __init__(
        self,
        model: ChessModel,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model
        self.model.eval()
        self.model.to(self.device)

    def evaluate_position(self, board: chess.Board) -> List[Tuple[float, chess.Move]]:
        logger.debug(f"Evaluating \n{board}")
        legal_moves = list(board.legal_moves)

        positions_to_eval = []
        moves_to_eval = []
        game_over_moves = []

        for move in legal_moves:
            board.push(move)
            if board.is_game_over():
                game_over_moves.append((self._game_over_score(board), move))
            else:
                positions_to_eval.append(board.fen())
                moves_to_eval.append(move)
            board.pop()

        eval_scores = self.model.evaluate(positions_to_eval)

        scored_moves = game_over_moves + list(zip(eval_scores, moves_to_eval))
        sorted_scored_scores = sorted(scored_moves, key=lambda x: x[0], reverse=board.turn)

        logger.debug(f"Evaluated {len(legal_moves)} legal moves")
        for (score, move) in sorted_scored_scores:
            logger.debug(f"{move}: {score}")

        return sorted_scored_scores

    @staticmethod
    def _game_over_score(board: chess.Board) -> float:
        if board.is_checkmate():
            return -1 if board.turn else 1
        return 0.0
