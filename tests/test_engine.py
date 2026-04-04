"""
Tests for NeuralEngine.
"""

import chess
import pytest
import torch

from neuralchess.engine import NeuralEngine
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
    score, move = engine.evaluate_position(board)[0]
    assert move is not None
    assert move in board.legal_moves
    assert isinstance(score, float)


def test_search_single_move(engine):
    board = chess.Board("7r/2k5/8/8/8/8/r7/7K w - - 0 1")
    score, move= engine.evaluate_position(board)[0]
    assert move == chess.Move.from_uci("h1g1")


def test_search_checkmate_position(engine):
    board = chess.Board("8/R7/8/8/8/k1K5/8/8 b - - 0 1")
    assert board.is_checkmate()
    assert len(engine.evaluate_position(board)) == 0


def test_search_stalemate_position(engine):
    board = chess.Board("k7/P7/K7/8/8/8/8/8 b - - 0 1")
    assert board.is_stalemate()
    assert len(engine.evaluate_position(board)) == 0


def test_game_over_score_checkmate(engine):
    board = chess.Board("k6R/8/1K6/8/8/8/8/8 b - - 1 1")
    assert board.is_checkmate()
    score = engine._game_over_score(board)
    assert score == 1.0

    board = chess.Board("K6r/8/1k6/8/8/8/8/8 w - - 1 1")
    assert board.is_checkmate()
    score = engine._game_over_score(board)
    assert score == 0.0


def test_game_over_score_stalemate(engine):
    board = chess.Board("k7/8/2K5/8/8/8/8/8 b - - 0 1")
    score = engine._game_over_score(board)
    assert score == 0.5
