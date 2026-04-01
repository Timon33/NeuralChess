"""
Download and preprocess a Kaggle chess evaluation dataset.

Detects FEN and evaluation columns automatically, converts all positions
to tensors using the specified encoder, and saves as precomputed .npy arrays.
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuralchess.core import parse_eval, scale_eval
from neuralchess.encoders import ENCODER_REGISTRY, get_encoder

FEN_CANDIDATES = ["fen", "FEN", "Fen"]
EVAL_CANDIDATES = [
    "eval",
    "evaluation",
    "Evaluation",
    "Eval",
    "centipawns",
    "centipawn",
]

BATCH_SIZE = 10000
KAGGLE_DATASET = "ronakbadhe/chess-evaluations"
RAW_CSV_NAME = "chessData.csv"


def detect_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(
        f"Could not detect column. Available: {list(df.columns)}. "
        f"Expected one of the candidates: {candidates}"
    )


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


def preprocess_csv(
    csv_path: str,
    output_dir: str,
    encoder_name: str,
    max_rows: int | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    encoder = get_encoder(encoder_name)
    print(f"Using encoder: {encoder.name} → output shape {encoder.output_shape}")

    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if max_rows is not None:
        df = df.head(max_rows)

    fen_col = detect_column(df, FEN_CANDIDATES)
    eval_col = detect_column(df, EVAL_CANDIDATES)
    print(f"Detected columns: FEN='{fen_col}', eval='{eval_col}'")
    print(f"Total positions: {len(df)}")

    valid_tensors: list[np.ndarray] = []
    valid_evals: list[float] = []
    skipped = 0

    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Converting positions"):
        end = min(start + BATCH_SIZE, len(df))
        batch_fens = df[fen_col].iloc[start:end]
        batch_evals = df[eval_col].iloc[start:end]

        for fen, cp in zip(batch_fens, batch_evals):
            try:
                tensor = encoder.encode(str(fen)).numpy()
                cp_value = parse_eval(str(cp))
                valid_tensors.append(tensor)
                valid_evals.append(scale_eval(cp_value))
            except Exception:
                skipped += 1

    tensors = np.array(valid_tensors, dtype=np.float32)
    evals = np.array(valid_evals, dtype=np.float32)

    tensors_path = os.path.join(output_dir, "tensors.npy")
    evals_path = os.path.join(output_dir, "evals.npy")

    print(f"Saving tensors to {tensors_path}")
    np.save(tensors_path, tensors)
    print(f"Saving evals to {evals_path}")
    np.save(evals_path, evals)

    print(f"Done. Saved {len(valid_tensors)} positions, skipped {skipped}.")


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
        default="bitboard",
        help=f"Encoder architecture: {available_encoders} (default: bitboard)",
    )
    parser.add_argument(
        "--max-rows", type=int, default=100000, help="Limit to N rows (default: 100000)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data", help="Base output directory"
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
    preprocess_csv(csv_path, arch_dir, args.architecture, args.max_rows)


if __name__ == "__main__":
    main()
