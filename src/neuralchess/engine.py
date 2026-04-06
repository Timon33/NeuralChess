"""
Neural chess engine with alpha-beta search and batched inference.
"""

import logging
from typing import Optional, List, Tuple

import chess
import torch

from neuralchess.models.base import ChessModel
from neuralchess.zobrist import ZobristHasher

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

        self._hasher = ZobristHasher()
        self.tt = {}

    def _store_position(self, board: chess.Board, score: float, visits: int, best_move: Optional[chess.Move]) -> None:
        self.tt[self._hasher.hash_board(board)] = (score, visits, best_move)

    def _get_position(self, board: chess.Board) -> tuple[float, int, Optional[chess.Move]]:
        return self.tt[self._hasher.hash_board(board)]

    @staticmethod
    def _game_over_score(board: chess.Board) -> float:
        if board.is_checkmate():
            return 0 if board.turn else 1
        return 0.5

    def _evaluate_leaves(
        self, board: chess.Board, moves: List[chess.Move]
    ) -> List[Tuple[float, chess.Move]]:
        """Batch evaluates moves and returns them sorted by score."""
        positions = []
        moves_to_eval = []
        game_over_moves = []

        logger.debug(f"Evaluating position\n{board}")

        # iterate all moves
        for move in moves:
            board.push(move)
            if board.is_game_over():
                score = self._game_over_score(board)
                game_over_moves.append((score, move))
                self._store_position(board, score, 1, chess.Move.null())
            else:
                positions.append(board.fen())
                moves_to_eval.append(move)
            board.pop()

        # evaluate non game over positions
        if positions:
            eval_scores = self.model.evaluate(positions)
        else:
            eval_scores = []

        # store evaluations
        for score, pos in zip(eval_scores, positions):
            self._store_position(chess.Board(pos), score, 1, None)

        scored_moves = game_over_moves + list(zip(eval_scores, moves_to_eval))
        scored_moves.sort(key=lambda x: x[0], reverse=board.turn)

        logger.debug(f"Evals")
        for s, m, pos in sorted(list(zip(eval_scores, moves_to_eval, positions)), key=lambda x: x[0], reverse=board.turn):
            logger.debug(f"{m} -> {pos}: {s}")

        return scored_moves

    def _explore_pv(self, board: chess.Board, depth: int) -> tuple[float, list[chess.Move]]:
        score, visits, best_move = self._get_position(board)
        side = 1 if board.turn else -1
        if best_move == chess.Move.null() or depth <= 0:
            return score, []
        if best_move is None:
            # this is the end of the pv, evaluate here
            scored_moves = self._evaluate_leaves(board, list(board.legal_moves))
            best_score = scored_moves[0][0]
            best_move = scored_moves[0][1]

            self._store_position(board, best_score, visits + 1, best_move)
            return best_score, [best_move]

        # if already explored go deeper
        board.push(best_move)
        logger.debug(f"Searching along move {best_move}")
        updated_score, pv = self._explore_pv(board, depth=depth - 1)
        board.pop()

        logger.debug(f"Score update for move {best_move}: {score * side} -> {updated_score * side}")
        # check if score worsen, if yes we might have to find the new best move
        if (side * updated_score) < (side * score):
            scored_moves = []
            for move in board.legal_moves:
                board.push(move)
                move_score, _, _ = self._get_position(board)
                scored_moves.append((move_score, move))
                board.pop()

            scored_moves.sort(key=lambda x: x[0], reverse=board.turn)
            best_score = scored_moves[0][0]
            best_move = scored_moves[0][1]
            self._store_position(board, best_score, visits + 1, best_move)
            logger.debug(f"Best move {best_move}: {best_score}, visits {visits}, [{pv}]")
            return best_score, [] + pv
        else:
            # still the best move, update score
            self._store_position(board, updated_score, visits + 1, best_move)
            logger.debug(f"Best move {best_move}: {updated_score}, visits {visits} [{pv}]")
            return updated_score, [best_move] + pv


    def evaluate_position(
        self, board: chess.Board, evals: int = 10
    ) -> Tuple[float, chess.Move]:
        """Root search: evaluates all legal moves using batched alpha-beta."""
        logger.debug(f"Starting eval: \n{board.fen()}")

        self.tt = {}

        try:
            self._get_position(board)
        except KeyError:
            logger.info(f"Position to yet evaluated, setting dummy entry")
            self._store_position(board, float("NaN"), 1, None)

        for i in range(evals):
            logger.debug("=====================")
            logger.debug(f"Starting eval: {i+1}")
            score, pv = self._explore_pv(board, depth=20)
            logger.info(f"{score}: {pv}")

        score, _, best_move = self._get_position(board)
        return score, best_move

def main():
    from neuralchess.models import load_model
    checkpoint = "./checkpoints/transformer_default.pt"
    logging.basicConfig(level=logging.DEBUG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, device)
    engine = NeuralEngine(model=model, device=device)

    postitions = [
        chess.Board("r1b1k2r/pp1n1ppp/3Q4/2q1p3/6n1/5N2/PP3PPP/R3R1K1 w kq - 0 16"),
        chess.Board("r1b1k2r/pp1n1ppp/3Q4/2q1p3/P5n1/5N2/1P3PPP/R3R1K1 b kq")
    ]

    for pos in postitions:
        engine._evaluate_leaves(pos, pos.legal_moves)

if __name__ == "__main__":
    main()
