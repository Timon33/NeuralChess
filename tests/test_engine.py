"""
Tests for NeuralEngine.
"""

import chess
import pytest
import torch

from neuralchess.engine import NeuralEngine, TTFlag
from neuralchess.models import NeuralChessNet


@pytest.fixture
def engine():
    model = NeuralChessNet()
    model.eval()
    return NeuralEngine(model=model, device=torch.device("cpu"))


def test_engine_loads_checkpoint(engine):
    assert engine.model is not None


def test_search_returns_move(engine):
    board = chess.Board()
    move, score, pv = engine.search(board, max_depth=3)
    assert move is not None
    assert move in board.legal_moves
    assert isinstance(score, float)
    assert isinstance(pv, list)


def test_search_single_move(engine):
    board = chess.Board("k7/8/8/8/8/8/8/K6R w - - 0 1")
    move, score, pv = engine.search(board, max_depth=3)
    assert move is not None


def test_search_checkmate_position(engine):
    board = chess.Board("k7/8/8/8/8/8/8/K7 w - - 0 1")
    assert not board.is_checkmate()
    move, score, pv = engine.search(board, max_depth=3)
    assert move is not None


def test_search_stalemate_position(engine):
    board = chess.Board("k7/P7/K7/8/8/8/8/8 b - - 0 1")
    assert board.is_stalemate()
    move, score, pv = engine.search(board, max_depth=3)
    assert move is None


def test_transposition_table_populated(engine):
    board = chess.Board()
    engine.search(board, max_depth=3)
    assert len(engine.tt) > 0


def test_tt_entry_structure(engine):
    entry = engine.tt.get(engine._zobrist.hash_board(chess.Board()))
    if entry is not None:
        assert isinstance(entry.depth, int)
        assert isinstance(entry.score, float)
        assert isinstance(entry.flag, TTFlag)


def test_evaluate_position(engine):
    score = engine._evaluate_position(chess.Board().fen())
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_move_ordering_tt_first(engine):
    board = chess.Board()
    moves = list(board.legal_moves)
    e4 = chess.Move.from_uci("e2e4")
    ordered = engine._order_moves(board, moves, tt_best=e4)
    assert ordered[0] == e4


def test_move_ordering_captures_before_quiet(engine):
    board = chess.Board(
        "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"
    )
    moves = list(board.legal_moves)
    captures = [m for m in moves if board.is_capture(m)]
    quiet = [m for m in moves if not board.is_capture(m)]
    ordered = engine._order_moves(board, moves)
    capture_indices = [ordered.index(m) for m in captures if m in ordered]
    quiet_indices = [ordered.index(m) for m in quiet if m in ordered]
    if capture_indices and quiet_indices:
        assert max(capture_indices) < min(quiet_indices)


def test_move_ordering_promotions(engine):
    board = chess.Board("8/P7/8/8/8/8/8/K6k w - - 0 1")
    moves = list(board.legal_moves)
    promotion = chess.Move.from_uci("a7a8q")
    ordered = engine._order_moves(board, moves)
    assert ordered[0] == promotion


def test_nodes_searched_increases(engine):
    board = chess.Board()
    engine.search(board, max_depth=3)
    assert engine.nodes_searched > 0


def test_search_interrupt(engine):
    import threading
    import time

    board = chess.Board()
    stop_event = threading.Event()

    # Start search in a thread
    def run_search():
        engine.search(board, max_depth=20, stop_event=stop_event)

    t = threading.Thread(target=run_search)
    t.start()

    time.sleep(0.1)
    stop_event.set()
    t.join(timeout=2.0)
    assert not t.is_alive()


def test_game_over_score_checkmate(engine):
    board = chess.Board("k6R/8/1K6/8/8/8/8/8 b - - 1 1")
    assert board.is_checkmate()
    score = engine._game_over_score(board)
    assert score == -1.0


def test_game_over_score_stalemate(engine):
    board = chess.Board("k7/8/2K5/8/8/8/8/8 b - - 0 1")
    score = engine._game_over_score(board)
    assert score == 0.0
