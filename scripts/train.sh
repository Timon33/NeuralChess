#!/usr/bin/env bash
# Generic training script for NeuralChess models on Modal GPU.
# Supports all model types (transformer, cnn) via --model-type.
# Accepts --config to load model architecture from a JSON file.
#
# Usage:
#   scripts/train.sh --model-type transformer --config configs/transformer_small.json
#   scripts/train.sh --model-type cnn --config configs/cnn_small.json
#   scripts/train.sh --model-type transformer --epochs 100 --lr 0.0001

set -e

# --- Default Training Arguments ---
DEFAULTS=(
  --model-type transformer
  --data-dir /vol/data/tokenizer
  --epochs 50
  --batch-size 1024
  --lr 0.0003
  --weight-decay 0.01
  --checkpoint-dir /vol/checkpoints
  --val-split 0.1
  --num-workers 4
  --amp
  --compile
  --seed 42
)

echo "=== NeuralChess Training (Modal) ==="
echo "Launching: modal run src/neuralchess/train.py"
echo ""

modal run src/neuralchess/train.py -- "${DEFAULTS[@]}" "$@"
