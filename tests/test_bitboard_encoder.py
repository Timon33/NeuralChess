import pytest
import torch

from neuralchess.encoders.bitboard import BitboardEncoder


class TestBitboardEncoder:
    def setup_method(self) -> None:
        self.encoder = BitboardEncoder()

    def test_output_shape(self) -> None:
        tensor = self.encoder.encode_position(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )
        assert tensor.shape == (14, 8, 8)

    def test_output_shape_property(self) -> None:
        assert self.encoder.output_shape == (14, 8, 8)

    def test_name_property(self) -> None:
        assert self.encoder.name == "bitboard"

    def test_starting_position_pieces(self) -> None:
        tensor = self.encoder.encode_position(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )
        assert tensor[0, 6, :].sum() == 8
        assert tensor[6, 1, :].sum() == 8
        assert tensor[5, 7, 4] == 1.0
        assert tensor[11, 0, 4] == 1.0
        assert tensor[:12].sum() == 32

    def test_side_to_move(self) -> None:
        tensor_w = self.encoder.encode_position(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )
        tensor_b = self.encoder.encode_position(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
        )
        assert tensor_w[12].sum() == 64.0
        assert tensor_b[12].sum() == 0.0

    def test_castling(self) -> None:
        tensor_all = self.encoder.encode_position(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )
        tensor_none = self.encoder.encode_position(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"
        )
        assert tensor_all[13].sum() == 6.0
        assert tensor_none[13].sum() == 0.0
