#!/usr/bin/env bash
set -e

echo "=== NeuralChess Overfit Test ==="

echo "[1/2] Generating tiny dataset for overfitting (1024 rows)..."
uv run neuralchess-data \
    --download \
    --architecture tokenizer \
    --max-rows 128 \
    --output-dir data_overfit

echo "[2/2] Training model to overfit..."
uv run neuralchess-train \
    --model-type transformer \
    --data-dir data_overfit/tokenizer \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.0003 \
    --checkpoint-dir data_overfit/checkpoints

echo "=== Overfit Test Complete ==="