"""
Model registry and factory functions for NeuralChess.
"""

from dataclasses import asdict
from typing import Any, Optional

import torch

from neuralchess.models.base import ChessModel, ModelConfig
from neuralchess.models.cnn import CNNConfig, NeuralChessNet

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "cnn": {"cls": NeuralChessNet, "config_cls": CNNConfig},
}

__all__ = [
    "ChessModel",
    "ModelConfig",
    "NeuralChessNet",
    "CNNConfig",
    "MODEL_REGISTRY",
    "create_model",
    "load_model",
]


def create_model(
    model_type: str,
    config: dict,
    device: torch.device,
) -> ChessModel:
    """Create a model from type name and config dict."""
    if model_type not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model type: {model_type}. Available: {available}")

    entry = MODEL_REGISTRY[model_type]
    config_cls = entry["config_cls"]
    model_cls = entry["cls"]

    model_config = config_cls(**config) if config else config_cls()
    model = model_cls(model_config)
    model.to(device)
    return model


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> ChessModel:
    """Load a model from checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_type = checkpoint.get("model_type")

    if model_type not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model type in checkpoint: {model_type}. Available: {available}"
        )

    model_cls = MODEL_REGISTRY[model_type]["cls"]
    model = model_cls.from_checkpoint(checkpoint_path, device)
    return model
