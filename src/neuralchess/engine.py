"""
Neural chess engine with batched quiescence search.
"""

import logging
import time
from typing import Optional, List, Tuple

import chess
import torch

from neuralchess.models.base import ChessModel
from neuralchess.zobrist import ZobristHasher

logger = logging.getLogger(__name__)


MAX_BATCH_SIZE = 4096


class QNode:
    def __init__(self, board: chess.Board, depth: int):
        self.board = board
        self.depth = depth
        self.stand_pat_score: float = 0.5
        self.children: List[Tuple[chess.Move, "QNode"]] = []
        self.is_leaf = False
        self.is_game_over = False
        self.turn = board.turn


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

    @staticmethod
    def _game_over_score(board: chess.Board) -> float:
        if board.is_checkmate():
            return 0.0 if board.turn else 1.0
        return 0.5

    def _is_noisy(self, board: chess.Board, move: chess.Move) -> bool:
        return (
            board.is_capture(move)
            or move.promotion is not None
            or board.gives_check(move)
        )

    def batched_qsearch(
        self, root_board: chess.Board, max_depth: int = 10
    ) -> Tuple[float, Optional[chess.Move]]:
        start_time = time.time()

        root = QNode(root_board.copy(), 0)
        frontier = [root]

        stats = {
            "nodes_visited": 0,
            "unique_positions_evaluated": 0,
            "max_depth_reached": 0,
            "batches_processed": 0,
        }

        logger.debug(f"Starting batched qsearch with max_depth={max_depth}")

        while frontier:
            stats["batches_processed"] += 1
            stats["nodes_visited"] += len(frontier)

            # 1. Evaluate current frontier
            eval_nodes = []
            fens = []

            for node in frontier:
                stats["max_depth_reached"] = max(stats["max_depth_reached"], node.depth)

                if node.board.is_game_over():
                    node.is_game_over = True
                    node.stand_pat_score = self._game_over_score(node.board)
                    node.is_leaf = True
                else:
                    eval_nodes.append(node)
                    fens.append(node.board.fen())

            if fens:
                unique_fens = list(set(fens))
                stats["unique_positions_evaluated"] += len(unique_fens)
                logger.debug(
                    f"Depth {frontier[0].depth}: Evaluating batch of {len(unique_fens)} unique positions "
                    f"(from {len(fens)} nodes)"
                )

                scores = []
                for i in range(0, len(unique_fens), MAX_BATCH_SIZE):
                    batch = unique_fens[i : i + MAX_BATCH_SIZE]
                    scores.extend(self.model.evaluate(batch))
                fen_to_score = dict(zip(unique_fens, scores))

                for node, fen in zip(eval_nodes, fens):
                    node.stand_pat_score = fen_to_score[fen]

            # 2. Expand current frontier
            new_frontier = []
            for node in eval_nodes:
                if node.depth >= max_depth:
                    node.is_leaf = True
                    continue

                moves = list(node.board.legal_moves)
                if not moves:
                    node.is_leaf = True
                    continue

                expanded_any = False
                for move in moves:
                    if node.depth == 0 or self._is_noisy(node.board, move):
                        child_board = node.board.copy()
                        child_board.push(move)
                        child_node = QNode(child_board, node.depth + 1)
                        node.children.append((move, child_node))
                        new_frontier.append(child_node)
                        expanded_any = True

                if not expanded_any:
                    node.is_leaf = True

            frontier = new_frontier

        # 3. Minimax Backpropagation
        def minimax(node: QNode) -> float:
            if node.is_leaf or not node.children:
                return node.stand_pat_score

            best_score = -float("inf") if node.turn else float("inf")

            can_stand_pat = node.depth > 0 and not node.board.is_check()
            if can_stand_pat:
                best_score = node.stand_pat_score

            for _, child in node.children:
                score = minimax(child)
                if node.turn:
                    best_score = max(best_score, score)
                else:
                    best_score = min(best_score, score)

            return best_score

        if not root.children:
            return root.stand_pat_score, None

        best_move = None
        best_score = -float("inf") if root.turn else float("inf")

        for move, child in root.children:
            score = minimax(child)
            if root.turn:  # White maximizes
                if best_move is None or score > best_score:
                    best_score = score
                    best_move = move
            else:  # Black minimizes
                if best_move is None or score < best_score:
                    best_score = score
                    best_move = move

        elapsed_time = time.time() - start_time
        logger.info(
            f"QSearch finished in {elapsed_time:.3f}s | "
            f"Nodes visited: {stats['nodes_visited']} | "
            f"Unique Evals: {stats['unique_positions_evaluated']} | "
            f"Max Depth: {stats['max_depth_reached']} | "
            f"Batches: {stats['batches_processed']} | "
            f"Best Move: {best_move} | "
            f"Score: {best_score:.4f}"
        )

        return best_score, best_move

    def evaluate_position(
        self, board: chess.Board, max_depth: int = 10
    ) -> Tuple[float, chess.Move]:
        """Root search: evaluates position using batched quiescence search."""
        logger.debug(f"Starting eval: \n{board.fen()}")

        score, best_move = self.batched_qsearch(board, max_depth=max_depth)

        if best_move is None:
            best_move = chess.Move.null()

        return score, best_move
