# NeuralChess

A lightweight neural chess engine built with a small Convolutional Neural Network (CNN) trained on Stockfish evaluations. Designed to run on a single consumer GPU or Google Colab.

## Overview

NeuralChess distills Stockfish's positional knowledge into a tiny, fast CNN (~500K parameters) that serves as the engine's intuition. It uses alpha-beta search with configurable depth and a transposition table for efficient lookups.

**Target strength:** 1500–1800 Elo  
**Training time:** A few hours on a free Google Colab T4 GPU  
**Inference:** Runs on CPU or GPU

## Architecture

- **Input:** 8×8×14 bitboard tensor (12 piece channels + turn + castling rights)
- **Network:** 5 Conv layers (3×3, 64→128 channels, ReLU, BatchNorm) → Flatten → 2 Dense layers → `tanh` output
- **Output:** Single value in [-1, 1] representing position evaluation
- **Search:** Alpha-beta pruning with configurable depth, transposition table, and move ordering

## Setup

### Prerequisites

- Python 3.13+
- pip or uv

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/NeuralChess.git
cd NeuralChess

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

## Quick Start

### 1. Download Training Data

```bash
python scripts/download_data.py
```

This downloads a Kaggle chess evaluations CSV (~5-20M positions with FEN + centipawn scores).

### 2. Train the Model

```bash
python scripts/train.py --epochs 20 --batch-size 4096 --lr 0.001
```

Training produces a checkpoint file in `checkpoints/`.

### 3. Play Against the Engine

**CLI mode:**
```bash
python scripts/play.py --checkpoint checkpoints/best.pt --depth 3
```

**UCI mode (for GUI integration):**
```bash
python scripts/uci.py --checkpoint checkpoints/best.pt
```

Connect to any UCI-compatible GUI (Arena, Cute Chess, ChessX).

## Project Structure

```
neuralchess/
├── src/neuralchess/
│   ├── model.py          # CNN architecture
│   ├── data.py           # FEN→tensor conversion, dataset loader
│   ├── train.py          # Training loop
│   ├── engine.py         # Alpha-beta search engine
│   ├── play.py           # CLI game interface
│   └── uci.py            # UCI protocol handler
├── scripts/
│   ├── download_data.py  # Dataset download helper
│   ├── train.py          # Training entry point
│   └── play.py           # CLI play entry point
├── tests/
├── checkpoints/          # Saved model weights (gitignored)
├── data/                 # Downloaded datasets (gitignored)
├── pyproject.toml
├── README.md
└── AGENTS.md             # Instructions for AI coding agents
```

## Configuration

### Engine Options (UCI)

| Option       | Default | Description                    |
| ------------ | ------- | ------------------------------ |
| `Depth`      | 3       | Search depth (plies)           |
| `ModelPath`  | —       | Path to model checkpoint       |
| `UseBook`    | false   | Use opening book if available  |

### Training Hyperparameters

| Parameter    | Default | Description                    |
| ------------ | ------- | ------------------------------ |
| `--epochs`   | 20      | Number of training epochs      |
| `--batch-size` | 4096  | Batch size                     |
| `--lr`       | 0.001   | Learning rate                  |
| `--val-split`| 0.1     | Validation split ratio         |

## License

MIT
