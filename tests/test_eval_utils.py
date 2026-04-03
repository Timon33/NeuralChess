import pytest

from neuralchess.core.eval_utils import parse_eval, scale_eval


class TestParseEval:
    def test_positive_centipawn(self) -> None:
        assert parse_eval("+56") == 56.0

    def test_negative_centipawn(self) -> None:
        assert parse_eval("-120") == -120.0

    def test_zero(self) -> None:
        assert parse_eval("0") == 0.0

    def test_mate_white(self) -> None:
        assert parse_eval("#10") == 2000.0

    def test_mate_with_plus(self) -> None:
        assert parse_eval("#+6") == 2000.0

    def test_mate_black(self) -> None:
        assert parse_eval("#-5") == -2000.0

    def test_mate_one(self) -> None:
        assert parse_eval("#1") == 2000.0

    def test_int_input(self) -> None:
        assert parse_eval(340) == 340.0

    def test_float_input(self) -> None:
        assert parse_eval(340.5) == 340.5


class TestScaleEval:
    def test_zero(self) -> None:
        assert scale_eval(0.0) == 0.0

    def test_600(self) -> None:
        result = scale_eval(600.0)
        assert abs(result - 0.761594) < 1e-5

    def test_symmetry(self) -> None:
        assert abs(scale_eval(100.0) + scale_eval(-100.0)) < 1e-6

    def test_inverse_identity(self) -> None:
        cp = 350.0
        scaled = scale_eval(cp)
        result = scale_eval(scaled, inverse=True)
        assert abs(result - cp) < 1e-4

    def test_inverse_zero(self) -> None:
        assert scale_eval(0.0, inverse=True) == 0.0

    def test_inverse_clamping(self) -> None:
        result = scale_eval(1.0, inverse=True)
        assert result > 0
        result = scale_eval(-1.0, inverse=True)
        assert result < 0

    def test_inverse_symmetry(self) -> None:
        scaled = scale_eval(200.0)
        assert abs(scale_eval(scaled, inverse=True) - 200.0) < 1e-4
        scaled_neg = scale_eval(-200.0)
        assert abs(scale_eval(scaled_neg, inverse=True) - (-200.0)) < 1e-4
