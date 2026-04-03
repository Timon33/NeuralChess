"""
Modal cloud GPU training for NeuralChess.

Usage:
    # Preprocess (CPU, no GPU needed)
    modal run modal_app.py -- --action preprocess --max-rows 100000

    # Train on GPU
    modal run modal_app.py -- --action train --epochs 20 --batch-size 4096

    # Preprocess + train in one run
    modal run modal_app.py -- --action all --epochs 20

    # Save checkpoint with custom name
    modal run modal_app.py -- --action train --checkpoint-name best_20epochs.pt

    # To change GPU, edit the gpu= parameter in the @app.function decorators below.
    # Available GPUs: A100, H100, L40S, L4, T4
"""

import os
import sys
from pathlib import Path

import modal

VOLUME_NAME = "neuralchess-data"
VOLUME_MOUNT = "/root/neuralchess-data"
DATA_DIR = "/root/neuralchess-data/data/bitboard"
CHECKPOINT_DIR = "/root/neuralchess-data/checkpoints"

PROJECT_ROOT = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "torch>=2.5.0",
        "python-chess>=1.999",
        "pandas>=2.2.0",
        "numpy>=2.0.0",
        "tqdm>=4.67.3",
        "kaggle>=1.6.0",
        "tensorboard>=2.14.0",
    )
    .env({"KAGGLE_CONFIG_DIR": "/root/.kaggle"})
    .add_local_dir(PROJECT_ROOT / "src", remote_path="/root/src")
    .add_local_dir(PROJECT_ROOT / "scripts", remote_path="/root/scripts")
)

app = modal.App("neuralchess", image=image)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

kaggle_secret = modal.Secret.from_name("kaggle-creds")


def _setup_paths() -> None:
    sys.path.insert(0, "/root/src")
    sys.path.insert(0, "/root/scripts")


@app.function(
    volumes={VOLUME_MOUNT: volume},
    secrets=[kaggle_secret],
)
def preprocess(
    max_rows: int | None = None,
    parallel: bool = True,
) -> dict:
    _setup_paths()

    from download_data import (
        download_kaggle_dataset,
        preprocess_csv,
        preprocess_csv_parallel,
    )

    raw_dir = "/root/neuralchess-data/data/raw"
    csv_path = download_kaggle_dataset(raw_dir)

    if parallel:
        preprocess_csv_parallel(csv_path, DATA_DIR, "bitboard", max_rows)
    else:
        preprocess_csv(csv_path, DATA_DIR, "bitboard", max_rows)

    volume.commit()

    import numpy as np

    tensors = np.load(os.path.join(DATA_DIR, "tensors.npy"))
    evals = np.load(os.path.join(DATA_DIR, "evals.npy"))

    return {
        "num_positions": len(tensors),
        "tensor_shape": list(tensors.shape),
        "eval_shape": list(evals.shape),
    }


@app.function(
    gpu="T4",
    volumes={VOLUME_MOUNT: volume},
    secrets=[kaggle_secret],
    timeout=24*60*60,
)
def train(
    epochs: int = 20,
    batch_size: int = 4096,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_accum_steps: int = 1,
    amp: bool = True,
    compile_model: bool = False,
    num_workers: int = 4,
    seed: int = 42,
    max_samples: int | None = None,
    val_split: float = 0.2,
    checkpoint_name: str = "best.pt",
) -> dict:
    _setup_paths()

    from train import (
        build_model,
        load_checkpoint,
        save_checkpoint,
        seed_everything,
        train_epoch,
        validate,
    )
    from neuralchess.core.dataset import ChessDataset

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, random_split
    from torch.utils.tensorboard import SummaryWriter

    seed_everything(seed)

    device = torch.device("cuda")
    print(f"Device: {device}")

    dataset = ChessDataset(DATA_DIR, max_positions=max_samples)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )

    model_type = "cnn"
    model_config: dict = {}
    start_epoch = 0
    best_val_loss = float("inf")

    resume_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    if os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        (
            model_state,
            optimizer_state,
            scheduler_state,
            start_epoch,
            best_val_loss,
            model_type,
            model_config,
        ) = load_checkpoint(resume_path, device)
        model = build_model(model_type, model_config, device, compile_model)
        model.load_state_dict(model_state)
        start_epoch += 1
    else:
        model = build_model(model_type, model_config, device, compile_model)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model type: {model_type} | Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    criterion = nn.MSELoss()

    tb_log_dir = os.path.join(CHECKPOINT_DIR, "tensorboard")
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard logging to: {tb_log_dir}")

    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(start_epoch, epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            grad_accum_steps,
            amp,
            writer=writer,
            epoch=epoch,
        )
        val_loss = validate(
            model, val_loader, criterion, device, amp, writer=writer, epoch=epoch
        )
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"lr={current_lr:.6f}"
        )

        if writer:
            writer.add_scalar("Loss/train_epoch", train_loss, epoch)
            writer.add_scalar("Loss/val_epoch", val_loss, epoch)
            writer.add_scalar("LR", current_lr, epoch)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            save_checkpoint(
                os.path.join(CHECKPOINT_DIR, checkpoint_name),
                epoch,
                model,
                optimizer,
                scheduler,
                best_val_loss,
                model_type,
                model_config,
            )
            print(f"  Saved best checkpoint (val_loss={val_loss:.4f})")

    writer.close()
    volume.commit()

    return {
        "best_val_loss": best_val_loss,
        "epochs_completed": epochs - start_epoch,
        "final_train_loss": history["train_loss"][-1]
        if history["train_loss"]
        else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
    }


@app.function(
    gpu="T4",
    volumes={VOLUME_MOUNT: volume},
    secrets=[kaggle_secret],
    timeout=24*60*60,
)
def train_and_preprocess(
    max_rows: int | None = None,
    epochs: int = 20,
    batch_size: int = 4096,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_accum_steps: int = 1,
    amp: bool = True,
    compile_model: bool = False,
    num_workers: int = 4,
    seed: int = 42,
    val_split: float = 0.2,
    checkpoint_name: str = "best.pt",
) -> dict:
    preprocess_result = preprocess.local(max_rows=max_rows)
    print(f"Preprocessed {preprocess_result['num_positions']} positions")
    train_result = train.local(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        grad_accum_steps=grad_accum_steps,
        amp=amp,
        compile_model=compile_model,
        num_workers=num_workers,
        seed=seed,
        val_split=val_split,
        checkpoint_name=checkpoint_name,
    )
    return {"preprocess": preprocess_result, "train": train_result}


@app.local_entrypoint()
def main(
    action: str = "train",
    epochs: int = 20,
    batch_size: int = 4096,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_rows: int | None = None,
    max_samples: int | None = None,
    grad_accum_steps: int = 1,
    num_workers: int = 4,
    val_split: float = 0.2,
    amp: bool = True,
    compile_model: bool = False,
    seed: int = 42,
    checkpoint_name: str = "best.pt",
) -> None:
    if action == "preprocess":
        result = preprocess.remote(max_rows=max_rows)
        print(f"Preprocess complete: {result}")
    elif action == "train":
        result = train.remote(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            grad_accum_steps=grad_accum_steps,
            num_workers=num_workers,
            max_samples=max_samples,
            val_split=val_split,
            amp=amp,
            compile_model=compile_model,
            seed=seed,
            checkpoint_name=checkpoint_name,
        )
        print(f"Training complete: {result}")
    elif action == "all":
        result = train_and_preprocess.remote(
            max_rows=max_rows,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            grad_accum_steps=grad_accum_steps,
            num_workers=num_workers,
            max_samples=max_samples,
            val_split=val_split,
            amp=amp,
            compile_model=compile_model,
            seed=seed,
            checkpoint_name=checkpoint_name,
        )
        print(f"Complete: {result}")
    else:
        print(f"Unknown action: {action}. Use 'preprocess', 'train', or 'all'.")
