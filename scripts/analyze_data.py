"""
Analyze preprocessed NeuralChess dataset and generate statistical plots.

Usage:
    uv run python scripts/analyze_data.py --data-dir data/bitboard --output analysis
"""

import argparse
import os
import random
from pathlib import Path

import chess
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


def load_data(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    tensors_path = os.path.join(data_dir, "tensors.npy")
    evals_path = os.path.join(data_dir, "evals.npy")

    if not os.path.exists(evals_path):
        raise FileNotFoundError(f"Evals file not found: {evals_path}")
    if not os.path.exists(tensors_path):
        raise FileNotFoundError(f"Tensors file not found: {tensors_path}")

    evals = np.load(evals_path, mmap_mode="r")
    tensors = np.load(tensors_path, mmap_mode="r")
    return evals, tensors


def compute_eval_stats(evals: np.ndarray) -> dict:
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    return {
        "count": len(evals),
        "min": float(evals.min()),
        "max": float(evals.max()),
        "mean": float(evals.mean()),
        "std": float(evals.std()),
        "median": float(np.median(evals)),
        "skewness": float(
            (((evals - evals.mean()) / max(evals.std(), 1e-8)) ** 3).mean()
        ),
        "percentiles": {f"P{p}": float(np.percentile(evals, p)) for p in percentiles},
    }


def compute_tensor_stats(tensors: np.ndarray) -> dict:
    return {
        "shape": list(tensors.shape),
        "dtype": str(tensors.dtype),
        "min": float(tensors.min()),
        "max": float(tensors.max()),
        "mean": float(tensors.mean()),
        "std": float(tensors.std()),
        "sparsity": float(np.mean(tensors == 0)),
        "n_positions": tensors.shape[0],
        "channels": tensors.shape[1] if tensors.ndim > 1 else 1,
    }


def compute_side_to_move_balance(tensors: np.ndarray) -> dict:
    if tensors.ndim < 4 or tensors.shape[1] < 13:
        return {"error": "Insufficient channels for side-to-move analysis"}

    stm_channel = tensors[:, 12, :, :]
    white_count = int(np.sum(stm_channel > 0.5))
    black_count = int(np.sum(stm_channel < 0.5))
    total = white_count + black_count
    return {
        "white_positions": white_count,
        "black_positions": black_count,
        "white_pct": white_count / max(total, 1) * 100,
        "black_pct": black_count / max(total, 1) * 100,
    }


def plot_eval_histogram(evals: np.ndarray, output_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    counts, bins, patches = ax.hist(
        evals,
        bins=100,
        density=True,
        alpha=0.7,
        color="steelblue",
        edgecolor="white",
        linewidth=0.5,
    )

    bin_centers = (bins[:-1] + bins[1:]) / 2
    bandwidth = 1.06 * evals.std() * len(evals) ** (-0.2)
    kde_x = np.linspace(bins[0], bins[-1], 500)
    kde_y = np.zeros_like(kde_x)
    for x_val in evals[::100]:
        kde_y += np.exp(-0.5 * ((kde_x - x_val) / bandwidth) ** 2)
    kde_y /= len(evals) * bandwidth * np.sqrt(2 * np.pi) * 100
    ax.plot(kde_x, kde_y, "r-", linewidth=2, label="KDE")
    ax.axvline(
        evals.mean(),
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {evals.mean():.4f}",
    )
    ax.axvline(
        np.median(evals),
        color="green",
        linestyle="--",
        linewidth=1.5,
        label=f"Median: {np.median(evals):.4f}",
    )
    ax.set_xlabel("Scaled Evaluation")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Scaled Evaluations")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(
        evals,
        bins=100,
        density=True,
        alpha=0.7,
        color="steelblue",
        edgecolor="white",
        linewidth=0.5,
        cumulative=True,
    )
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.axvline(
        np.median(evals),
        color="green",
        linestyle="--",
        linewidth=1.5,
        label=f"Median: {np.median(evals):.4f}",
    )
    ax.set_xlabel("Scaled Evaluation")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Cumulative Distribution Function")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_eval_boxplot(evals: np.ndarray, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    p_values = [np.percentile(evals, p) for p in percentiles]

    colors = ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91bfdb", "#4575b4"]
    bar_colors = [
        colors[0],
        colors[1],
        colors[2],
        colors[3],
        colors[3],
        colors[4],
        colors[4],
        colors[5],
        colors[5],
    ]

    y_pos = np.arange(len(percentiles))
    bars = ax.barh(y_pos, p_values, color=bar_colors, edgecolor="white", linewidth=0.5)

    for i, (p, v) in enumerate(zip(percentiles, p_values)):
        ax.text(
            v + 0.01 * np.sign(v) if abs(v) > 0.01 else 0.01,
            i,
            f"P{p}: {v:.4f}",
            va="center",
            fontsize=9,
        )

    ax.axvline(0, color="black", linewidth=1, linestyle="-")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"P{p}" for p in percentiles])
    ax.set_xlabel("Scaled Evaluation")
    ax.set_title("Evaluation Percentiles")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_stm_eval_correlation(
    tensors: np.ndarray, evals: np.ndarray, output_path: str
) -> None:
    if tensors.ndim < 4 or tensors.shape[1] < 13:
        return

    is_white = tensors[:, 12, 0, 0] > 0.5
    is_black = ~is_white

    white_evals = evals[is_white]
    black_evals = evals[is_black]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        white_evals,
        bins=100,
        density=True,
        alpha=0.6,
        color="#ff7f0e",
        label=f"White to Move (n={len(white_evals):,})",
    )
    ax.hist(
        black_evals,
        bins=100,
        density=True,
        alpha=0.6,
        color="#1f77b4",
        label=f"Black to Move (n={len(black_evals):,})",
    )

    ax.axvline(
        white_evals.mean(),
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.5,
        label=f"White Mean: {white_evals.mean():.4f}",
    )
    ax.axvline(
        black_evals.mean(),
        color="#1f77b4",
        linestyle="--",
        linewidth=1.5,
        label=f"Black Mean: {black_evals.mean():.4f}",
    )

    ax.set_xlabel("Scaled Evaluation (0 to 1)")
    ax.set_ylabel("Density")
    ax.set_title("Evaluation Distribution by Side to Move")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def load_fens_from_csv(csv_path: str, sample_size: int = 5000) -> list[str]:
    df = pd.read_csv(csv_path, usecols=["FEN"])
    fens = df["FEN"].astype(str).tolist()
    if len(fens) > sample_size:
        fens = random.sample(fens, sample_size)
    return fens


def compute_position_types(fens: list[str]) -> dict:
    counts = {
        "quiet": 0,
        "has_capture": 0,
        "in_check": 0,
        "has_check": 0,
        "has_promotion": 0,
        "has_castling": 0,
        "invalid": 0,
    }

    for fen in fens:
        try:
            board = chess.Board(fen)
        except ValueError:
            counts["invalid"] += 1
            continue

        is_check = board.is_check()
        has_capture = any(board.is_capture(m) for m in board.legal_moves)
        has_promotion = any(m.promotion for m in board.legal_moves)
        has_check = any(board.gives_check(m) for m in board.legal_moves)
        has_castling = bool(board.castling_rights)

        if is_check:
            counts["in_check"] += 1
        if has_capture:
            counts["has_capture"] += 1
        if has_check:
            counts["has_check"] += 1
        if has_promotion:
            counts["has_promotion"] += 1
        if has_castling:
            counts["has_castling"] += 1

        if not is_check and not has_capture:
            counts["quiet"] += 1

    total = len(fens) - counts["invalid"]
    result: dict[str, float | int] = dict(counts)
    result["total"] = total
    for key in counts:
        if key not in ("total", "invalid"):
            result[f"{key}_pct"] = counts[key] / max(total, 1) * 100

    return result


def plot_position_types(pos_types: dict, output_path: str) -> None:
    labels = [
        "Quiet",
        "Has Capture",
        "In Check",
        "Can Give Check",
        "Has Promotion",
        "Has Castling",
    ]
    keys = [
        "quiet",
        "has_capture",
        "in_check",
        "has_check",
        "has_promotion",
        "has_castling",
    ]
    values = [pos_types[k] for k in keys]
    total = pos_types["total"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#e377c2", "#17becf"]

    ax = axes[0]
    bars = ax.barh(
        labels[::-1], values[::-1], color=colors[::-1], edgecolor="white", linewidth=0.5
    )
    for i, (label, val) in enumerate(zip(labels[::-1], values[::-1])):
        pct = val / max(total, 1) * 100
        ax.text(
            val + total * 0.005,
            i,
            f"{val:,} ({pct:.1f}%)",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Count")
    ax.set_title("Position Types (Sampled)")
    ax.grid(True, axis="x", alpha=0.3)

    ax = axes[1]
    pcts = [val / max(total, 1) * 100 for val in values]
    wedges, texts, autotexts = ax.pie(
        pcts,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 9},
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
    ax.set_title("Position Type Distribution")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_tensor_heatmap(tensors: np.ndarray, output_path: str) -> None:
    fig, axes = plt.subplots(2, 7, figsize=(18, 6))

    channel_names = [
        "W-Pawn",
        "W-Knight",
        "W-Bishop",
        "W-Rook",
        "W-Queen",
        "W-King",
        "B-Pawn",
        "B-Knight",
        "B-Bishop",
        "B-Rook",
        "B-Queen",
        "B-King",
        "Side-to-move",
        "Castling",
    ]

    sample_idx = len(tensors) // 2

    for i in range(14):
        row, col = divmod(i, 7)
        ax = axes[row, col]
        if i < tensors.shape[1]:
            im = ax.imshow(
                tensors[sample_idx, i],
                cmap="hot",
                interpolation="nearest",
                vmin=0,
                vmax=1,
            )
            ax.set_title(
                channel_names[i] if i < len(channel_names) else f"Ch{i}", fontsize=8
            )
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis("off")

    fig.suptitle(
        f"Sample Position (index {sample_idx}) — Channel Visualization", fontsize=12
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def print_stats_table(
    eval_stats: dict,
    tensor_stats: dict,
    stm_stats: dict,
    pos_types: dict | None = None,
) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("  NeuralChess Dataset Analysis")
    lines.append("=" * 60)

    lines.append("\n--- Evaluation Statistics ---")
    lines.append(f"  Positions:  {eval_stats['count']:>12,}")
    lines.append(f"  Min:        {eval_stats['min']:>12.6f}")
    lines.append(f"  Max:        {eval_stats['max']:>12.6f}")
    lines.append(f"  Mean:       {eval_stats['mean']:>12.6f}")
    lines.append(f"  Std:        {eval_stats['std']:>12.6f}")
    lines.append(f"  Median:     {eval_stats['median']:>12.6f}")
    lines.append(f"  Skewness:   {eval_stats['skewness']:>12.6f}")
    lines.append("")
    for name, val in eval_stats["percentiles"].items():
        lines.append(f"  {name}:       {val:>12.6f}")

    lines.append("\n--- Tensor Statistics ---")
    lines.append(f"  Shape:      {tensor_stats['shape']}")
    lines.append(f"  Dtype:      {tensor_stats['dtype']}")
    lines.append(f"  Min:        {tensor_stats['min']:>12.6f}")
    lines.append(f"  Max:        {tensor_stats['max']:>12.6f}")
    lines.append(f"  Mean:       {tensor_stats['mean']:>12.6f}")
    lines.append(f"  Std:        {tensor_stats['std']:>12.6f}")
    lines.append(f"  Sparsity:   {tensor_stats['sparsity'] * 100:>11.2f}%")

    lines.append("\n--- Side-to-Move Balance ---")
    if "error" not in stm_stats:
        lines.append(
            f"  White:      {stm_stats['white_positions']:>12,} ({stm_stats['white_pct']:.1f}%)"
        )
        lines.append(
            f"  Black:      {stm_stats['black_positions']:>12,} ({stm_stats['black_pct']:.1f}%)"
        )
    else:
        lines.append(f"  {stm_stats['error']}")

    if pos_types is not None:
        lines.append("\n--- Position Types (Sampled) ---")
        total = pos_types["total"]
        lines.append(f"  Sample size:{total:>12,}")
        labels = {
            "quiet": "Quiet",
            "has_capture": "Has Capture",
            "in_check": "In Check",
            "has_check": "Can Give Check",
            "has_promotion": "Has Promotion",
            "has_castling": "Has Castling",
        }
        for key, label in labels.items():
            count = int(pos_types[key])
            pct = pos_types[f"{key}_pct"]
            lines.append(f"  {label}:{count:>12,} ({pct:.1f}%)")
        invalid = int(pos_types["invalid"])
        if invalid:
            lines.append(f"  Invalid FENs:{invalid:>12,}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze NeuralChess dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/bitboard/",
        help="Path to preprocessed data directory",
    )
    parser.add_argument(
        "--output", type=str, default="analysis/", help="Output directory for plots"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to raw CSV with FEN column for position type analysis",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Number of FENs to sample for position type analysis",
    )
    args = parser.parse_args()

    print(f"Loading data from: {args.data_dir}")
    evals, tensors = load_data(args.data_dir)
    print(f"Loaded {len(evals):,} positions, tensor shape: {tensors.shape}")

    print("\nComputing statistics...")
    eval_stats = compute_eval_stats(evals)
    tensor_stats = compute_tensor_stats(tensors)
    stm_stats = compute_side_to_move_balance(tensors)

    os.makedirs(args.output, exist_ok=True)

    print("\nGenerating plots...")
    plot_eval_histogram(evals, os.path.join(args.output, "evals_histogram.png"))
    print(f"  Saved: {args.output}/evals_histogram.png")

    plot_eval_boxplot(evals, os.path.join(args.output, "evals_percentiles.png"))
    print(f"  Saved: {args.output}/evals_percentiles.png")

    plot_tensor_heatmap(tensors, os.path.join(args.output, "tensor_heatmap.png"))
    print(f"  Saved: {args.output}/tensor_heatmap.png")

    plot_stm_eval_correlation(
        tensors, evals, os.path.join(args.output, "stm_eval_correlation.png")
    )
    print(f"  Saved: {args.output}/stm_eval_correlation.png")

    pos_types = None
    if args.csv:
        print(f"\nAnalyzing position types from: {args.csv}")
        fens = load_fens_from_csv(args.csv, args.sample_size)
        print(f"  Sampled {len(fens):,} positions...")
        pos_types = compute_position_types(fens)

        plot_position_types(pos_types, os.path.join(args.output, "position_types.png"))
        print(f"  Saved: {args.output}/position_types.png")
    else:
        print("\nSkipping position type analysis (use --csv to enable)")

    stats_text = print_stats_table(eval_stats, tensor_stats, stm_stats, pos_types)
    print(stats_text)

    stats_path = os.path.join(args.output, "stats.txt")
    with open(stats_path, "w") as f:
        f.write(stats_text)
    print(f"\nStats saved to: {stats_path}")


if __name__ == "__main__":
    main()
