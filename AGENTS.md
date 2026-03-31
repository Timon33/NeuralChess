# AGENTS.md — Instructions for AI Coding Agents

This file contains project-specific instructions for AI coding agents (e.g., opencode) working on NeuralChess. Read this file at the start of every session.

## Project Overview

NeuralChess is a lightweight neural chess engine using a 5-layer CNN trained on Stockfish evaluations from Kaggle datasets. It uses python-chess for board logic and implements alpha-beta search with a transposition table. Target strength: 1500–1800 Elo.

## Architecture Reference

### Input Tensor: 8×8×14
- Channels 0–5: White pieces (Pawn, Knight, Bishop, Rook, Queen, King) — one-hot per channel
- Channels 6–11: Black pieces (Pawn, Knight, Bishop, Rook, Queen, King) — one-hot per channel
- Channel 12: Side to move (all 1s = white, all 0s = black)
- Channel 13: Castling availability (1s on squares with castling rights)

### CNN Architecture
- 5 Conv layers: 3×3 filters, 64→64→128→128→128 channels, ReLU + BatchNorm
- Flatten → Linear(128×4×4, 256) → ReLU → Linear(256, 64) → ReLU → Linear(64, 1) → tanh
- Output: scalar in [-1, 1]

### Evaluation Scaling
- Centipawn scores from dataset are scaled via: `tanh(cp / 600)`
- This maps typical evals (±600cp) to roughly ±0.76, with extreme values asymptoting to ±1

### Engine
- Alpha-beta search with negamax formulation
- Transposition table keyed by zobrist hash (from python-chess `board.zobrist_hash()`)
- Move ordering: captures first (MVV-LVA), then promotions, then other moves
- Batched CNN inference at leaf nodes for performance
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
| `model.py` | NeuralChessNet class, forward pass, device handling |
| `data.py` | `fen_to_tensor()`, `ChessDataset`, data loading utilities |
| `train.py` | Training loop, validation, checkpoint save/load |
| `engine.py` | `NeuralEngine` class: alpha-beta search, transposition table, move selection |
| `play.py` | Interactive CLI game loop, board display, move input |
| `uci.py` | Full UCI protocol implementation |

## Testing

- Use pytest
- Test files mirror source: `tests/test_<module>.py`
- Test FEN→tensor conversion with known positions
- Test engine search against known tactical positions
- Test UCI protocol message handling

## Important Constraints

1. **Never commit** data files, model checkpoints, or CSV files (they are gitignored)
2. **Never hardcode** search depth — always make it configurable via CLI arg or UCI option
3. **Use `python-chess`** for all board logic — never reimplement chess rules
4. **Batch inference** when evaluating multiple positions in the engine — never call the model one position at a time in a loop
5. **Keep the model small** — do not add attention layers, transformers, or increase parameter count significantly
6. **CPU-compatible** — the engine must work on CPU even if trained on GPU

## Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Train model
python scripts/train.py --epochs 20 --batch-size 4096

# Play CLI
python scripts/play.py --checkpoint checkpoints/best.pt --depth 3

# Run UCI engine
python scripts/uci.py --checkpoint checkpoints/best.pt
```
