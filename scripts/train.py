"""
Training script for NeuralChess models.

Supports resume, mixed precision, torch.compile, gradient accumulation,
and configurable hyperparameters via CLI arguments.
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuralchess.core.dataset import ChessDataset
from neuralchess.models import CNNConfig, NeuralChessNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    config: CNNConfig, device: torch.device, compile_model: bool
) -> NeuralChessNet:
    model = NeuralChessNet(config).to(device)
    if compile_model:
        model = torch.compile(model)
    return model


def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
    best_val_loss: float,
    config: CNNConfig,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "best_val_loss": best_val_loss,
        "config": {
            "input_channels": config.input_channels,
            "conv_channels": config.conv_channels,
            "kernel_size": config.kernel_size,
            "pool_after_first": config.pool_after_first,
            "fc_hidden": config.fc_hidden,
            "dropout": config.dropout,
        },
    }
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint["epoch"], checkpoint["best_val_loss"]


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_accum: int,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    optimizer.zero_grad()
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).unsqueeze(1)

        with torch.autocast(
            device_type="cuda", enabled=use_amp and device.type == "cuda"
        ):
            outputs = model(inputs)
            loss = criterion(outputs, targets) / grad_accum

        scaler.scale(loss).backward()

        if (batch_idx + 1) % grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        num_batches += 1

    if num_batches % grad_accum != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).unsqueeze(1)

        with torch.autocast(
            device_type="cuda", enabled=use_amp and device.type == "cuda"
        ):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float("inf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NeuralChess model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--data-dir", type=str, default="data/bitboard/")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = ChessDataset(args.data_dir)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    config = CNNConfig()
    model = build_model(config, device, args.compile)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    criterion = nn.MSELoss()

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        print(f"Resuming from {args.resume}")
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        start_epoch += 1

    checkpoint_path = os.path.join(args.checkpoint_dir, "best.pt")

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args.grad_accum_steps,
            args.amp,
        )
        val_loss = validate(model, val_loader, criterion, device, args.amp)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"LR: {current_lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                checkpoint_path,
                epoch,
                model,
                optimizer,
                scheduler,
                best_val_loss,
                config,
            )
            print(f"  Saved best checkpoint (val_loss={val_loss:.6f})")

    print(f"Training complete. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
