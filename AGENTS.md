# AGENTS.md — Instructions for AI Coding Agents

This file contains project-specific instructions for AI coding agents (e.g., opencode) working on NeuralChess. Read this file at the start of every session.

## Project Overview

NeuralChess is a lightweight neural chess engine supporting multiple model architectures (CNN, transformer, etc.) trained on Stockfish evaluations from Kaggle datasets. It uses python-chess for board logic and implements alpha-beta search with a transposition table. Target strength: 1500–1800 Elo.

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

### Engine
- Alpha-beta search with negamax formulation
- Transposition table keyed by zobrist hash (from python-chess `board.zobrist_hash()`)
- Move ordering: captures first (MVV-LVA), then promotions, then other moves
- Batched inference at leaf nodes for performance
- Configurable depth via CLI arg or UCI `setoption`

## Code Conventions

- **No comments** in code unless explicitly requested
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
| `engine.py` | `NeuralEngine` class: alpha-beta search, transposition table, move selection |
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
2. **Never hardcode** search depth — always make it configurable via CLI arg or UCI option
3. **Use `python-chess`** for all board logic — never reimplement chess rules
4. **Batch inference** when evaluating multiple positions in the engine — never call the model one position at a time in a loop
5. **Keep models small** — target ~500K parameters for CPU compatibility
6. **CPU-compatible** — the engine must work on CPU even if trained on GPU
7. **Architecture-agnostic core** — new architectures should only add files in `encoders/` and `models/` without modifying `core/`

## Cloud Training (Modal)

NeuralChess supports cloud GPU training via Modal. Local workflows remain unchanged.

### Setup (one-time)

```bash
# Install modal CLI
pip install modal

# Authenticate
modal setup

# Create Kaggle API secret
modal secret create kaggle-creds KAGGLE_API_TOKEN=<your-token>

# Create persistent volume for data and checkpoints
modal volume create neuralchess-data
```

### Commands

```bash
# Preprocess data
modal run modal_app.py --action preprocess --max-rows 100000

# Train on cloud GPU (loads preprocessed data from volume)
modal run modal_app.py --action train --epochs 20 --batch-size 4096

# Preprocess + train in one run
modal run modal_app.py --action all --epochs 20

# Save checkpoint with custom name
modal run modal_app.py --action train --checkpoint-name best_20epochs.pt

# Download trained checkpoint from Modal volume to local
modal volume get neuralchess-data checkpoints/best.pt ./checkpoints/best.pt

# List files in Modal volume
modal volume ls neuralchess-data
```

### Data Handling

- **Modal Volume** (`neuralchess-data`) persists data and checkpoints across runs
- Preprocessed `.npy` files are stored at `data/bitboard/` in the volume
- Checkpoints are stored at `checkpoints/` in the volume
- Training auto-resumes from existing checkpoint if found
- Download checkpoints locally with `modal volume get` to use with `play.py` or `uci.py`
- 
## Commands

```bash
# Install dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Download and preprocess data (CNN/bitboard)
uv run python scripts/download_data.py --download --architecture bitboard --max-rows 100000

# Train model
uv run python scripts/train.py --epochs 20 --batch-size 4096

# Play CLI
uv run python scripts/play.py --checkpoint checkpoints/best.pt --depth 3

# Run UCI engine
uv run python scripts/uci.py --checkpoint checkpoints/best.pt
```
