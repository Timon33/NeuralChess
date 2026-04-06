"""
Dataset augmentation script for NeuralChess.

Takes existing FENs from the Kaggle dataset, applies 1-3 random legal moves
to generate out-of-distribution positions (including blunders), and evaluates
them with Stockfish at depth 12 (sufficient to spot 1-move blunders).

Usage:
    python scripts/augment_data.py [--num-samples N] [--output PATH]
"""

import argparse
import csv
import multiprocessing
import os
import random
from typing import List, Tuple

import chess
import chess.engine
from tqdm import tqdm

STOCKFISH_PATH = "/home/xyper/.local/bin/stockfish"
INPUT_CSV = "/home/xyper/code/NeuralChess/data/raw/chessData.csv"
OUTPUT_CSV = "/home/xyper/code/NeuralChess/data/raw/augmentedData.csv"

# Mate value cap for centipawn conversion
MATE_VALUE = 2000


def process_chunk(args: Tuple[List[str], str]) -> List[Tuple[str, str]]:
    """Process a chunk of FENs: apply random moves and evaluate with Stockfish."""
    fens, stockfish_path = args
    results = []

    engine = None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

        for fen in fens:
            try:
                board = chess.Board(fen)

                # Apply 1-3 random legal moves to create OOD positions
                # This naturally includes blunders (hanging pieces, etc.)
                num_random_moves = random.randint(1, 3)
                for _ in range(num_random_moves):
                    legal_moves = list(board.legal_moves)
                    if not legal_moves:
                        break
                    move = random.choice(legal_moves)
                    board.push(move)

                # Skip game-over positions
                if board.is_game_over():
                    continue

                new_fen = board.fen()
                # Depth 12 is enough to reliably spot 1-move blunders
                info = engine.analyse(board, chess.engine.Limit(depth=12))
                pov_score = info.get("score")
                if pov_score is None:
                    continue

                # Convert to absolute (from White's perspective) to match Kaggle format
                score = pov_score.white()

                if score.is_mate():
                    m = score.mate()
                    if m is not None and m > 0:
                        eval_str = f"#+{m}"
                    elif m is not None and m < 0:
                        eval_str = f"{m}"
                        eval_str = eval_str.replace("-", "#-")
                    else:
                        eval_str = "#0"
                else:
                    cp = score.score()
                    if cp is not None and cp > 0:
                        eval_str = f"+{cp}"
                    else:
                        eval_str = str(cp)

                results.append((new_fen, eval_str))
            except Exception:
                # Silently ignore errors (e.g., timeouts, invalid FENs)
                pass

    finally:
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment chess dataset with blunder positions"
    )
    parser.add_argument(
        "--num-samples", type=int, default=100000, help="Number of FENs to process"
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_CSV, help="Output CSV path"
    )
    parser.add_argument("--input", type=str, default=INPUT_CSV, help="Input CSV path")
    parser.add_argument(
        "--stockfish", type=str, default=STOCKFISH_PATH, help="Stockfish binary path"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500, help="FENs per worker chunk"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return

    print(f"Reading FENs from {args.input}...")
    fens = []
    with open(args.input, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            fens.append(row[0])
            if len(fens) >= args.num_samples:
                break

    print(f"Read {len(fens)} FENs. Generating random moves and evaluating...")

    # Group FENs into chunks for parallel processing
    chunks = [
        (fens[i : i + args.chunk_size], args.stockfish)
        for i in range(0, len(fens), args.chunk_size)
    ]

    all_results = []
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_cores} cores...")

    with multiprocessing.Pool(processes=num_cores) as pool:
        for results in tqdm(
            pool.imap_unordered(process_chunk, chunks), total=len(chunks)
        ):
            all_results.extend(results)

    print(f"Saving {len(all_results)} augmented positions to {args.output}...")
    with open(args.output, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["FEN", "Evaluation"])
        for res in all_results:
            writer.writerow(res)

    print("Done! You can now run:")
    print(f"  neuralchess-download --csv {args.output} --architecture tokenizer")


if __name__ == "__main__":
    main()
