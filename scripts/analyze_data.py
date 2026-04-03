"""
Analyze preprocessed NeuralChess dataset and generate statistical plots.

Usage:
    uv run python scripts/analyze_data.py --data-dir data/bitboard --output analysis
"""

import argparse
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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


def print_stats_table(eval_stats: dict, tensor_stats: dict, stm_stats: dict) -> str:
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

    stats_text = print_stats_table(eval_stats, tensor_stats, stm_stats)
    print(stats_text)

    stats_path = os.path.join(args.output, "stats.txt")
    with open(stats_path, "w") as f:
        f.write(stats_text)
    print(f"\nStats saved to: {stats_path}")


if __name__ == "__main__":
    main()
