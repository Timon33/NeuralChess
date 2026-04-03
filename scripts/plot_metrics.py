"""
Convenience script to launch TensorBoard for monitoring NeuralChess training.

Usage:
    uv run python scripts/plot_metrics.py --logdir checkpoints/tensorboard
"""

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch TensorBoard for NeuralChess")
    parser.add_argument(
        "--logdir",
        type=str,
        default="checkpoints/tensorboard",
        help="Path to TensorBoard log directory",
    )
    parser.add_argument(
        "--port", type=int, default=6006, help="Port to run TensorBoard on"
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="Host to run TensorBoard on"
    )
    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        print(f"Warning: Log directory does not exist: {args.logdir}")
        print("Make sure you have started training or downloaded logs from Modal.")

    print(f"Launching TensorBoard on http://{args.host}:{args.port}")
    print(f"Log directory: {args.logdir}")
    print("Press Ctrl+C to stop.")

    try:
        subprocess.run(
            [
                "tensorboard",
                f"--logdir={args.logdir}",
                f"--port={args.port}",
                f"--host={args.host}",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        print("\nTensorBoard stopped.")
    except FileNotFoundError:
        print(
            "Error: 'tensorboard' command not found. Install it with 'uv add tensorboard'."
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error launching TensorBoard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
