"""
Download and preprocess a Kaggle chess evaluation dataset.

Detects FEN and evaluation columns automatically, converts all positions
to tensors using the specified encoder, and saves as precomputed .npy arrays.
"""

import argparse
import math
import os
import re
import subprocess
import sys
import zipfile
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

from neuralchess.encoders import ENCODER_REGISTRY, get_encoder
from neuralchess.encoders.base import PositionEncoder

FEN_ROW_NAME = "FEN"
EVAL_ROW_NAME = "Evaluation"
BATCH_SIZE = 10000
KAGGLE_DATASET = "ronakbadhe/chess-evaluations"
RAW_CSV_NAME = "chessData.csv"

_ENCODER_CACHE: dict[str, "PositionEncoder"] = {}

MATE_VALUE = 2000.0
MATE_PATTERN = re.compile(r"^#([+-]?)(\d+)$")

SAMPLE_SIZE = 5

def parse_eval(raw: str | int | float) -> float:
    """
    convert centipawns stockfish evaluation to win prop
    :param raw: cp
    :return: win prop
    """
    raw = str(raw).strip()
    match = MATE_PATTERN.match(raw)
    if match:
        return -1.0 if match.group(1) == "-" else 1.0

    return 0.5 * (2 / (1 + math.exp(-0.00368208 * float(raw))))


def _worker_init(encoder_name: str) -> None:
    global _ENCODER_CACHE
    _ENCODER_CACHE[encoder_name] = get_encoder(encoder_name)


def _encode_chunk(
    args: tuple[list[str], list[str], str],
) -> tuple[np.ndarray, np.ndarray, int]:
    fens, evals, encoder_name = args
    encoder = _ENCODER_CACHE[encoder_name]
    n = len(fens)
    shape = (n, *encoder.output_shape)
    tensors = np.zeros(shape, dtype=np.float32)
    scaled_evals = np.zeros(n, dtype=np.float32)
    valid_mask = np.ones(n, dtype=bool)

    for i, (fen, cp) in enumerate(zip(fens, evals)):
        try:
            tensors[i] = encoder.encode_position(str(fen)).numpy()
            scaled_evals[i] = parse_eval(str(cp))
        except Exception:
            valid_mask[i] = False

    return tensors[valid_mask], scaled_evals[valid_mask], int(~valid_mask.any())


def download_kaggle_dataset(raw_dir: str) -> str:
    os.makedirs(raw_dir, exist_ok=True)

    csv_path = os.path.join(raw_dir, RAW_CSV_NAME)
    if os.path.exists(csv_path):
        print(f"Raw CSV already exists: {csv_path}")
        return csv_path

    print(f"Downloading dataset: {KAGGLE_DATASET}")
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", raw_dir],
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stderr)
    except FileNotFoundError:
        print("Error: kaggle CLI not found.")
        print("Install it: pip install kaggle")
        print("Set up credentials: https://www.kaggle.com/docs/api")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e.stderr}")
        sys.exit(1)

    zip_path = None
    for f in os.listdir(raw_dir):
        if f.endswith(".zip"):
            zip_path = os.path.join(raw_dir, f)
            break

    if zip_path is None:
        print("Error: No zip file found after download.")
        sys.exit(1)

    print(f"Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)

    os.remove(zip_path)

    if not os.path.exists(csv_path):
        for f in os.listdir(raw_dir):
            if f.endswith(".csv"):
                csv_path = os.path.join(raw_dir, f)
                break

    if not os.path.exists(csv_path):
        print("Error: Could not find chessData.csv in extracted files.")
        sys.exit(1)

    print(f"Found CSV: {csv_path}")
    return csv_path


def _log_sample_data(
    fens: list[str],
    evals: list[str],
    tensors: np.ndarray,
    scaled_evals: np.ndarray,
) -> None:
    n = min(SAMPLE_SIZE, len(fens))

    print(f"\n--- First {n} encoded datapoints ---")
    for i in range(n):
        t = tensors[i]
        print(
            f"  [{i + 1}] tensor_shape={t.shape} | "
            f"tensor_range=[{t.min():.4f}, {t.max():.4f}] | "
            f"scaled_eval={scaled_evals[i]:.4f}"
        )
    print()


def _load_csv_data(
    csv_path: str,
    encoder_name: str,
    max_rows: int | None = None,
) -> tuple[list[str], list[str], "PositionEncoder"]:
    encoder = get_encoder(encoder_name)
    print(f"Using encoder: {encoder_name} → output shape {encoder.output_shape}")

    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df)}")

    if max_rows is not None:
        df = df.head(max_rows)

    fens = df[FEN_ROW_NAME].astype(str).tolist()
    evals = df[EVAL_ROW_NAME].astype(str).tolist()

    n = min(SAMPLE_SIZE, len(fens))
    print(f"\n--- First {n} CSV rows ---")
    for i in range(n):
        fen_preview = fens[i][:80] + ("..." if len(fens[i]) > 80 else "")
        print(f"  [{i + 1}] FEN={fen_preview} | Eval={evals[i]}")
    print()

    return fens, evals, encoder


def _save_results(
    tensors: np.ndarray,
    scaled_evals: np.ndarray,
    output_dir: str,
    skipped: int,
) -> None:
    tensors_path = os.path.join(output_dir, "tensors.npy")
    evals_path = os.path.join(output_dir, "evals.npy")

    print(f"Saving tensors to {tensors_path}")
    np.save(tensors_path, tensors)
    print(f"Saving evals to {evals_path}")
    np.save(evals_path, scaled_evals)

    print(f"Done. Saved {len(tensors)} positions, skipped {skipped}.")


def preprocess_csv(
    csv_path: str,
    output_dir: str,
    encoder_name: str,
    max_rows: int | None = None,
    n_jobs: int | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    fens, evals, encoder = _load_csv_data(csv_path, encoder_name, max_rows)

    if n_jobs is None:
        _preprocess_sequential(fens, evals, encoder, output_dir)
    else:
        _preprocess_parallel(fens, evals, encoder_name, output_dir, n_jobs)


def _preprocess_sequential(
    fens: list[str],
    evals: list[str],
    encoder: "PositionEncoder",
    output_dir: str,
) -> None:
    n = len(fens)
    shape = (n, *encoder.output_shape)
    tensors = np.zeros(shape, dtype=np.float32)
    scaled_evals = np.zeros(n, dtype=np.float32)
    valid_mask = np.ones(n, dtype=bool)
    skipped = 0

    for start in tqdm(range(0, n, BATCH_SIZE), desc="Converting positions"):
        end = min(start + BATCH_SIZE, n)
        batch_fens = fens[start:end]
        batch_evals = evals[start:end]

        for i, (fen, cp) in enumerate(zip(batch_fens, batch_evals)):
            try:
                tensors[start + i] = encoder.encode_position(fen).numpy()
                scaled_evals[start + i] = parse_eval(cp)
            except Exception:
                valid_mask[start + i] = False
                skipped += 1

    tensors = tensors[valid_mask]
    scaled_evals = scaled_evals[valid_mask]
    _log_sample_data(fens, evals, tensors, scaled_evals)
    _save_results(tensors, scaled_evals, output_dir, skipped)


def _preprocess_parallel(
    fens: list[str],
    evals: list[str],
    encoder_name: str,
    output_dir: str,
    n_jobs: int,
) -> None:
    encoder = get_encoder(encoder_name)

    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)

    chunk_size = BATCH_SIZE
    chunks = [
        (
            fens[start : min(start + chunk_size, len(fens))],
            evals[start : min(start + chunk_size, len(fens))],
            encoder_name,
        )
        for start in range(0, len(fens), chunk_size)
    ]
    chunk_sizes = [
        end - start
        for start in range(0, len(fens), chunk_size)
        for end in [min(start + chunk_size, len(fens))]
    ]

    valid_tensors: list[np.ndarray] = []
    valid_evals: list[np.ndarray] = []
    total_skipped = 0

    with Pool(
        processes=n_jobs, initializer=_worker_init, initargs=(encoder_name,)
    ) as pool:
        with tqdm(
            total=len(fens),
            desc="Converting positions",
            unit="pos",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as pbar:
            for idx, (tensors, evals_batch, skipped) in enumerate(
                pool.imap(_encode_chunk, chunks)
            ):
                valid_tensors.append(tensors)
                valid_evals.append(evals_batch)
                total_skipped += skipped
                pbar.update(chunk_sizes[idx])

    if valid_tensors:
        tensors = np.concatenate(valid_tensors, axis=0)
        scaled_evals = np.concatenate(valid_evals, axis=0)
    else:
        tensors = np.zeros((0, *encoder.output_shape), dtype=np.float32)
        scaled_evals = np.zeros(0, dtype=np.float32)

    _log_sample_data(fens[: len(fens)], evals[: len(evals)], tensors, scaled_evals)
    _save_results(tensors, scaled_evals, output_dir, total_skipped)


def main() -> None:
    available_encoders = ", ".join(sorted(ENCODER_REGISTRY.keys()))
    parser = argparse.ArgumentParser(
        description="Download and preprocess chess evaluation data"
    )
    parser.add_argument("--csv", type=str, help="Path to CSV file (skips download)")
    parser.add_argument("--download", action="store_true", help="Download from Kaggle")
    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
        help=f"Encoder architecture (required): {available_encoders}",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None, help="Limit to N rows (default: all)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data", help="Base output directory"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use multiprocessing for encoding (uses all available cores)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Number of parallel workers (default: 8)",
    )
    args = parser.parse_args()

    if args.architecture not in ENCODER_REGISTRY:
        print(
            f"Error: Unknown architecture '{args.architecture}'. Available: {available_encoders}"
        )
        sys.exit(1)

    raw_dir = os.path.join(args.output_dir, "raw")

    if args.csv:
        csv_path = args.csv
    elif args.download:
        csv_path = download_kaggle_dataset(raw_dir)
    else:
        print("No CSV provided. Use one of:")
        print(f"  --download    Auto-download from Kaggle ({KAGGLE_DATASET})")
        print("  --csv <path>  Use an existing CSV file")
        sys.exit(1)

    arch_dir = os.path.join(args.output_dir, args.architecture)
    n_jobs = args.n_jobs if args.parallel else None
    preprocess_csv(csv_path, arch_dir, args.architecture, args.max_rows, n_jobs)


if __name__ == "__main__":
    main()
