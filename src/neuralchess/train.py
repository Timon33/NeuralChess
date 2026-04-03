"""
Training script for NeuralChess models.

Supports resume, mixed precision, torch.compile, gradient accumulation,
and configurable hyperparameters via CLI arguments.
Architecture-agnostic checkpoint format.
"""

import argparse
import os
import random
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from neuralchess.core.dataset import ChessDataset
from neuralchess.models import MODEL_REGISTRY, ChessModel, create_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    model_type: str,
    model_config: dict,
    device: torch.device,
    compile_model: bool,
) -> ChessModel:
    model = create_model(model_type, model_config, device)
    if compile_model:
        model = torch.compile(model)  # type: ignore[assignment]
    return model  # type: ignore[return-value]


def save_checkpoint(
    path: str,
    epoch: int,
    model: ChessModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
    best_val_loss: float,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model_type = model.config.__class__.__name__.replace("Config", "").lower()
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "best_val_loss": best_val_loss,
        "model_type": model_type,
        "model_config": {
            k: v for k, v in asdict(model.config).items() if not k.startswith("_")
        },
    }
    torch.save(state, path)


def load_checkpoint(
    path: str,
    device: torch.device,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Optional[dict[str, Any]],
    int,
    float,
    str,
    dict[str, Any],
]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return (
        checkpoint["model_state"],
        checkpoint["optimizer_state"],
        checkpoint.get("scheduler_state"),
        checkpoint["epoch"],
        checkpoint["best_val_loss"],
        checkpoint["model_type"],
        checkpoint["model_config"],
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_accum: int,
    use_amp: bool,
    writer: Optional[SummaryWriter] = None,
    epoch: int = 0,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    num_batches = 0
    batch_losses: list[float] = []
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")  # type: ignore[attr-defined]

    optimizer.zero_grad()
    pbar = tqdm(
        loader,
        desc="Training",
        leave=True,
        dynamic_ncols=True,
        unit="batch",
    )

    global_step = epoch * len(loader)

    for batch_idx, (inputs, targets) in enumerate(pbar):
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

        loss_val = loss.item() * grad_accum
        total_loss += loss_val
        num_batches += 1
        batch_losses.append(loss_val)

        if writer and num_batches % max(1, len(loader) // 50) == 0:
            writer.add_scalar("Loss/train_step", loss_val, global_step + batch_idx)

        pbar.set_postfix({"loss": f"{total_loss / num_batches:.4f}"})

    if num_batches % grad_accum != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    losses_arr = np.array(batch_losses)
    return {
        "mean": total_loss / num_batches,
        "std": float(losses_arr.std()),
        "min": float(losses_arr.min()),
        "max": float(losses_arr.max()),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    writer: Optional[SummaryWriter] = None,
    epoch: int = 0,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    batch_losses: list[float] = []

    all_targets = []
    all_outputs = []

    pbar = tqdm(
        loader,
        desc="Validating",
        leave=True,
        dynamic_ncols=True,
        unit="batch",
    )
    for inputs, targets in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).unsqueeze(1)

        with torch.autocast(
            device_type="cuda", enabled=use_amp and device.type == "cuda"
        ):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if writer and num_batches == 0:
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

        total_loss += loss.item()
        num_batches += 1
        batch_losses.append(loss.item())
        pbar.set_postfix({"loss": f"{total_loss / num_batches:.4f}"})

    if writer and len(all_targets) > 0:
        targets_np = np.concatenate(all_targets).flatten()
        outputs_np = np.concatenate(all_outputs).flatten()
        writer.add_histogram("Distribution/targets", targets_np, epoch)
        writer.add_histogram("Distribution/outputs", outputs_np, epoch)
        writer.add_scalar("DistributionStats/outputs_mean", outputs_np.mean(), epoch)
        writer.add_scalar("DistributionStats/outputs_std", outputs_np.std(), epoch)

    losses_arr = np.array(batch_losses)
    return {
        "mean": total_loss / num_batches if num_batches > 0 else float("inf"),
        "std": float(losses_arr.std()) if len(losses_arr) > 0 else 0.0,
        "min": float(losses_arr.min()) if len(losses_arr) > 0 else 0.0,
        "max": float(losses_arr.max()) if len(losses_arr) > 0 else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NeuralChess model")
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/")
    parser.add_argument("--checkpoint-name", type=str, default="best.pt")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--no-tensorboard", action="store_true", help="Disable TensorBoard logging"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = ChessDataset(args.data_dir, max_positions=args.max_samples)
    print(f"Data loaded. Dataset size: {len(dataset)}")
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    model_config: dict = {}
    start_epoch = 0
    best_val_loss = float("inf")
    model_type = args.model_type
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)

    if os.path.isfile(checkpoint_path):
        if not args.resume:
            print(f"WARNING: checkpoint {checkpoint_path} already exists")
            exit(1)

    if args.resume:
        resume_path = os.path.join(args.checkpoint_dir, args.resume)
        print(f"Resuming from {resume_path}")
        (
            model_state,
            optimizer_state,
            scheduler_state,
            start_epoch,
            best_val_loss,
            model_type,
            model_config,
        ) = load_checkpoint(str(resume_path), device)
        model = build_model(model_type, model_config, device, args.compile)
        model.load_state_dict(model_state)
        start_epoch += 1
    else:
        model = build_model(model_type, model_config, device, args.compile)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model type: {model_type} | Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2
    )
    criterion = nn.MSELoss()

    writer = None
    if not args.no_tensorboard:
        log_dir = os.path.join(args.checkpoint_dir, "tensorboard")
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logging to: {log_dir}")

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args.grad_accum_steps,
            args.amp,
            writer=writer,
            epoch=epoch,
        )
        val_metrics = validate(
            model, val_loader, criterion, device, args.amp, writer=writer, epoch=epoch
        )
        scheduler.step(val_metrics["mean"])

        current_lr = optimizer.param_groups[0]["lr"]
        if writer:
            writer.add_scalar("Loss/train_epoch", train_metrics["mean"], epoch)
            writer.add_scalar("Loss/val_epoch", val_metrics["mean"], epoch)
            writer.add_scalar("LR", current_lr, epoch)

        train_cv = (
            train_metrics["std"] / train_metrics["mean"]
            if train_metrics["mean"] != 0
            else 0.0
        )
        val_cv = (
            val_metrics["std"] / val_metrics["mean"]
            if val_metrics["mean"] != 0
            else 0.0
        )
        train_range = train_metrics["max"] - train_metrics["min"]
        val_range = val_metrics["max"] - val_metrics["min"]

        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{args.epochs - 1}")
        print(f"{'=' * 60}")
        print(
            f"  Train Loss: {train_metrics['mean']:.4f} ± {train_metrics['std']:.4f}  (range: {train_metrics['min']:.4f} → {train_metrics['max']:.4f})"
        )
        print(
            f"  Val Loss:   {val_metrics['mean']:.4f} ± {val_metrics['std']:.4f}  (range: {val_metrics['min']:.4f} → {val_metrics['max']:.4f})"
        )
        print(f"  LR:         {current_lr:.6f}")
        print(f"{'─' * 60}")
        print(f"  Stability:")
        print(f"    Train CV:  {train_cv:.4f}  |  Range: {train_range:.4f}")
        print(f"    Val CV:    {val_cv:.4f}  |  Range: {val_range:.4f}")
        print(f"{'=' * 60}")

        if val_metrics["mean"] < best_val_loss:
            best_val_loss = val_metrics["mean"]
            save_checkpoint(
                str(checkpoint_path),
                epoch,
                model,
                optimizer,
                scheduler,
                best_val_loss,
            )
            print(f"  ✓ Saved best checkpoint (val_loss={best_val_loss:.4f})")

    if writer:
        writer.close()

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

# --- Modal Integration ---
try:
    import modal
except ImportError:
    modal = None

# Only define Modal objects if the modal package is installed
if modal is not None:
    app = modal.App("neuralchess-trainer")

    # 1. Define the Environment Image
    # Installs dependencies from your local pyproject.toml and adds the local
    # neuralchess package source.
    image = (
        modal.Image.debian_slim()
        .pip_install_from_pyproject("pyproject.toml")
        .add_local_python_source("neuralchess")
    )

    # 2. Mount the Persistent Volume
    volume = modal.Volume.from_name("neuralchess-data", create_if_missing=True)

    # 3. Define the Remote Execution Function
    @app.function(
        image=image,
        volumes={"/vol": volume},
        gpu="T4",  # Automatically acquires an available GPU
        timeout=86400,  # 24 hour timeout for training
    )
    def modal_main(*args):
        import sys
        import threading
        import time

        # Intercept and override sys.argv so your existing argparse logic
        # inside main() works perfectly without modification.
        sys.argv = ["train.py"] + list(args)

        # Continuously sync logs to the cloud every 60 seconds
        stop_event = threading.Event()

        def background_commit():
            while not stop_event.is_set():
                time.sleep(60)
                if stop_event.is_set():
                    break
                try:
                    volume.commit()
                except Exception as e:
                    print(f"Background volume commit failed: {e}")

        committer = threading.Thread(target=background_commit, daemon=True)
        committer.start()

        try:
            # Run your existing logic!
            main()
        finally:
            stop_event.set()
            # Ensure all data (checkpoints, logs) written to /vol is persisted
            volume.commit()

    # 4. Define the Local CLI Entrypoint for Modal
    @app.local_entrypoint()
    def run_modal(*args):
        """
        Invoked via: modal run src/neuralchess/train.py -- --arg1 val1
        """
        print(f"🚀 Launching NeuralChess Training on Modal GPU...")
        print(f"Arguments forwarded: {args}")
        # Forward the arbitrary CLI arguments to the remote function
        modal_main.remote(*args)
