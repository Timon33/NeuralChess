"""
Transformer-based neural network for chess position evaluation.

Uses tokenized board representations processed through a transformer encoder
with global average pooling for position evaluation.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from neuralchess.encoders import get_encoder
from neuralchess.models import ChessModel
from neuralchess.models.base import ChessModel, ModelConfig


@dataclass
class TransformerConfig(ModelConfig):
    encoder_name: str = "tokenizer"
    vocab_size: int = 65
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 8
    seq_len: int = 70


class TransformerChessNet(ChessModel):
    def __init__(self, config: Optional[TransformerConfig] = None) -> None:
        super().__init__()
        self._config = config or TransformerConfig()
        self._encoder = get_encoder(self._config.encoder_name)
        self._device = torch.device("cpu")

        self.token_emb = nn.Embedding(self._config.vocab_size, self._config.d_model)
        self.pos_emb = nn.Embedding(self._config.seq_len, self._config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._config.d_model,
            nhead=self._config.nhead,
            dim_feedforward=self._config.d_model * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self._config.num_layers,
            norm=nn.LayerNorm(self._config.d_model),
            enable_nested_tensor=False,
        )

        self.head = nn.Linear(self._config.d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(self._config.seq_len, device=x.device).unsqueeze(0)
        x = self.token_emb(x.long()) + self.pos_emb(positions)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return torch.sigmoid(self.head(x))

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str, device: torch.device
    ) -> "TransformerChessNet":
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model_config = checkpoint.get("model_config", {})
        config = (
            TransformerConfig(**model_config) if model_config else TransformerConfig()
        )
        model = cls(config)

        state = checkpoint["model_state"]
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        model.to(device)
        return model

    @property
    def config(self) -> TransformerConfig:
        return self._config

    @property
    def expected_input_shape(self) -> tuple[int, ...]:
        return self._encoder.output_shape
