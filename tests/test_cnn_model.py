import torch

from neuralchess.models import CNNConfig, NeuralChessNet


def test_cnn_output_shape() -> None:
    model = NeuralChessNet()
    batch = torch.zeros(32, *model.expected_input_shape)
    output = model(batch)
    assert output.shape == (32, 1)


def test_cnn_forward_pass() -> None:
    model = NeuralChessNet()
    batch = torch.randn(8, *model.expected_input_shape)
    output = model(batch)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    assert (output >= -1.0).all() and (output <= 1.0).all()


def test_cnn_param_count() -> None:
    model = NeuralChessNet()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameter count: {param_count:,}")
    assert param_count > 0


def test_cnn_custom_config() -> None:
    config = CNNConfig(conv_channels=(32, 64), fc_hidden=(128,))
    model = NeuralChessNet(config)
    batch = torch.randn(4, *model.expected_input_shape)
    output = model(batch)
    assert output.shape == (4, 1)
