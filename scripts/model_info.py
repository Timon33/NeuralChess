#!/usr/bin/env python
"""
Utility script to print architecture details and parameter counts for NeuralChess models.

Usage:
    uv run python scripts/model_info.py --model-type transformer
    uv run python scripts/model_info.py --model-type cnn --config configs/cnn_small.json
    uv run python scripts/model_info.py --model-type transformer --config configs/transformer_small.json
"""

import argparse
import json
import os
import sys

import torch

from neuralchess.models import MODEL_REGISTRY, create_model


def load_config(config_path: str) -> dict:
    if not os.path.isfile(config_path):
        print(f"Error: config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        return json.load(f)


def print_model_info(name: str, model_type: str, config: dict) -> None:
    model = create_model(model_type, config, torch.device("cpu"))
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    config_cls = MODEL_REGISTRY[model_type]["config_cls"]
    defaults = config_cls()
    merged = {
        k: getattr(defaults, k)
        for k in asdict(defaults).keys()
        if not k.startswith("_")
    }
    merged.update(config)

    print(f"{'=' * 60}")
    print(f"Model: {name}")
    print(f"Type:  {model_type}")
    print(f"{'=' * 60}")
    print(f"Full Config:")
    for k, v in sorted(merged.items()):
        default_val = getattr(defaults, k, None)
        marker = (
            " (default)"
            if config.get(k, default_val) == default_val and k not in config
            else ""
        )
        print(f"  {k}: {v}{marker}")
    print(f"{'-' * 60}")
    print(f"Total Parameters:     {param_count:>12,}")
    print(f"Trainable Parameters: {trainable_count:>12,}")
    print(f"{'-' * 60}")
    print(model)
    print(f"{'=' * 60}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect NeuralChess model architectures"
    )
    parser.add_argument(
        "--model-type", type=str, required=True, choices=sorted(MODEL_REGISTRY.keys())
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to JSON config file"
    )
    parser.add_argument(
        "--name", type=str, default=None, help="Display name for the model"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config: dict = {}
    if args.config:
        config = load_config(args.config)

    name = args.name or f"{args.model_type} (custom)"
    if args.config:
        name = f"{args.model_type} ({os.path.basename(args.config)})"

    print_model_info(name, args.model_type, config)


if __name__ == "__main__":
    from dataclasses import asdict

    main()
