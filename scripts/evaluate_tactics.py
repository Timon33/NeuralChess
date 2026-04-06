import argparse
import sys
import time
from pathlib import Path

import chess
import pandas as pd
import torch
from tqdm import tqdm

from neuralchess.engine import NeuralEngine
from neuralchess.models import load_model


def evaluate_tactics(
    checkpoint_path: str,
    dataset_path: str,
    num_puzzles: int,
    max_depth: int,
    device_str: str = "auto"
):
    print(f"Loading dataset from {dataset_path}...")
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"ERROR: Dataset not found at {dataset_path}")
        sys.exit(1)
    
    if num_puzzles > len(df):
        print(f"Warning: Requested {num_puzzles} puzzles, but dataset only has {len(df)}. Using all.")
        num_puzzles = len(df)
        
    sample_df = df.sample(n=num_puzzles, random_state=42).reset_index(drop=True)
    print(f"Sampled {num_puzzles} random puzzles (seed=42).")

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
        
    print(f"Loading model from {checkpoint_path} on {device}...")
    try:
        model = load_model(checkpoint_path, device)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        sys.exit(1)
        
    engine = NeuralEngine(model=model, device=device)
    
    correct = 0
    total_time = 0.0
    
    print(f"Starting evaluation (max_depth={max_depth})...")
    
    with tqdm(total=num_puzzles, desc="Evaluating tactics") as pbar:
        for idx, row in sample_df.iterrows():
            fen = row['FEN']
            expected_move_uci = row['Move']
            
            board = chess.Board(fen)
            
            start_time = time.time()
            score, best_move = engine.evaluate_position(board, evals=max_depth)
            elapsed = time.time() - start_time
            total_time += elapsed
            
            best_move_uci = best_move.uci() if best_move else "None"
            
            if best_move_uci == expected_move_uci:
                correct += 1
                
            pbar.set_postfix({"Accuracy": f"{correct/(idx+1)*100:.1f}%"})
            pbar.update(1)

    accuracy = correct / num_puzzles * 100
    avg_time = total_time / num_puzzles
    
    print("\n" + "="*50)
    print("Tactics Evaluation Results")
    print("="*50)
    print(f"Puzzles evaluated: {num_puzzles}")
    print(f"Correct solutions: {correct}")
    print(f"Accuracy:          {accuracy:.2f}%")
    print(f"Total time:        {total_time:.2f}s")
    print(f"Avg time/puzzle:   {avg_time:.3f}s")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Evaluate NeuralChess engine on tactical puzzles")
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint")
    parser.add_argument("--dataset", default="data/raw/tactic_evals.csv", help="Path to the tactics dataset CSV")
    parser.add_argument("--num-puzzles", type=int, default=100, help="Number of random puzzles to evaluate")
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum search depth for quiescence search")
    parser.add_argument("--device", default="auto", help="Device to use (cpu, cuda, or auto)")
    
    args = parser.parse_args()
    
    evaluate_tactics(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        num_puzzles=args.num_puzzles,
        max_depth=args.max_depth,
        device_str=args.device
    )

if __name__ == "__main__":
    main()
