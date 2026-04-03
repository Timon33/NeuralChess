from neuralchess.encoders.base import PositionEncoder
from neuralchess.encoders.bitboard import BitboardEncoder
from neuralchess.encoders.tokenizer import TokenEncoder

ENCODER_REGISTRY: dict[str, type[PositionEncoder]] = {
    "bitboard": BitboardEncoder,
    "tokenizer": TokenEncoder,
}


def get_encoder(name: str) -> PositionEncoder:
    if name not in ENCODER_REGISTRY:
        available = ", ".join(sorted(ENCODER_REGISTRY.keys()))
        raise ValueError(f"Unknown encoder '{name}'. Available: {available}")
    return ENCODER_REGISTRY[name]()
