"""
Tests for ZobristHasher.
"""

import chess
import pytest

from neuralchess.zobrist import ZobristHasher


@pytest.fixture
def hasher():
    return ZobristHasher(seed=42)


def test_starting_position_hash_is_deterministic(hasher):
    board1 = chess.Board()
    board2 = chess.Board()
    assert hasher.hash_board(board1) == hasher.hash_board(board2)


def test_different_positions_have_different_hashes(hasher):
    board1 = chess.Board()
    board2 = chess.Board()
    board2.push_san("e4")
    assert hasher.hash_board(board1) != hasher.hash_board(board2)


def test_same_position_after_move_and_undo(hasher):
    board = chess.Board()
    h1 = hasher.hash_board(board)
    board.push_san("e4")
    board.push_san("e5")
    board.pop()
    board.pop()
    h2 = hasher.hash_board(board)
    assert h1 == h2


def test_different_seeds_produce_different_hashes():
    h1 = ZobristHasher(seed=42)
    h2 = ZobristHasher(seed=123)
    board = chess.Board()
    assert h1.hash_board(board) != h2.hash_board(board)


def test_castling_rights_affect_hash(hasher):
    board1 = chess.Board()
    board2 = chess.Board()
    board2.push_san("e4")
    board2.push_san("e5")
    board2.push_san("Nf3")
    board2.push_san("Nc6")
    assert hasher.hash_board(board1) != hasher.hash_board(board2)


def test_side_to_move_affects_hash(hasher):
    board1 = chess.Board()
    board2 = chess.Board()
    board2.push_san("e4")
    assert hasher.hash_board(board1) != hasher.hash_board(board2)


def test_ep_square_affects_hash(hasher):
    board1 = chess.Board()
    board1.push_san("e4")
    h1 = hasher.hash_board(board1)
    board2 = chess.Board()
    board2.push_san("e4")
    board2.push_san("d5")
    h2 = hasher.hash_board(board2)
    assert h1 != h2


def test_transposition_same_hash(hasher):
    board1 = chess.Board()
    board1.push_san("e4")
    board1.push_san("e5")
    board1.push_san("Nf3")
    board1.push_san("Nc6")
    board1.push_san("Bb5")
    board1.push_san("a6")

    board2 = chess.Board()
    board2.push_san("e4")
    board2.push_san("e5")
    board2.push_san("Bb5")
    board2.push_san("a6")
    board2.push_san("Nf3")
    board2.push_san("Nc6")

    assert hasher.hash_board(board1) == hasher.hash_board(board2)
