## Project Overview

NeuralChess is a lightweight neural chess engine supporting multiple model architectures (CNN, transformer, etc.) trained on Stockfish evaluations from Kaggle datasets. It uses python-chess for board logic.

## Architecture Reference

### Position Encoders

Encoders convert FEN strings to tensors. Each encoder has a `name` and `output_shape`.

#### Bitboard Encoder (CNN) — `encoders/bitboard.py`
- Output shape: `(14, 8, 8)`
- Channels 0–5: White pieces (Pawn, Knight, Bishop, Rook, Queen, King) — one-hot per channel
- Channels 6–11: Black pieces (Pawn, Knight, Bishop, Rook, Queen, King) — one-hot per channel
- Channel 12: Side to move (all 1s = white, all 0s = black)
- Channel 13: Castling availability (1s on squares with castling rights)

### Evaluation Scaling
- Centipawn scores from dataset are scaled via: `tanh(cp / 600)`
- This maps typical evals (±600cp) to roughly ±0.76, with extreme values asymptoting to ±1

## Code Conventions

- **Type hints** on all function signatures
- **Docstrings** on public classes and functions only
- **Naming:** snake_case for functions/variables, PascalCase for classes
- **Imports:** standard library → third-party → local, sorted alphabetically within groups
- **Error handling:** raise descriptive exceptions; do not silently fail

## Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `core/eval_utils.py` | `parse_eval()`, `scale_eval()` — architecture-agnostic |
| `core/dataset.py` | `ChessDataset` — loads precomputed `.npy` tensors/evals |
| `encoders/base.py` | `PositionEncoder` ABC — interface for FEN→tensor encoders |
| `encoders/bitboard.py` | `BitboardEncoder` — CNN-specific 14×8×8 bitboard encoding |
| `encoders/__init__.py` | `ENCODER_REGISTRY`, `get_encoder()` — encoder factory |
| `models/base.py` | `ChessModel` ABC — interface for neural architectures |
| `engine.py` | `NeuralEngine` move selection |
| `play.py` | Interactive CLI game loop, board display, move input |
| `uci.py` | Full UCI protocol implementation |

## Testing

- Use pytest
- Test files mirror source: `tests/test_<module>.py`
- Test encoder output shapes and correctness with known positions
- Test engine search against known tactical positions
- Test UCI protocol message handling

## Important Constraints

1. **Never commit** data files, model checkpoints, or CSV files (they are gitignored)
3. **Use `python-chess`** for all board logic — never reimplement chess rules
4. **Batch inference** when evaluating multiple positions in the engine — never call the model one position at a time in a loop
7. **Architecture-agnostic core** — new architectures should only add files in `encoders/` and `models/` without modifying `core/`

## Commands

```bash
# Install dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Download and preprocess data (CNN/bitboard)
uv run python scripts/download_data.py --download --architecture bitboard --max-rows 100000

# Train model
uv run neuralchess-train --epochs 20 --batch-size 4096
```
