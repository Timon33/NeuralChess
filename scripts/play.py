"""
Interactive CLI game loop for NeuralChess.
"""

import argparse
import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuralchess.engine import NeuralEngine

PIECE_ASCII = {
    chess.PAWN: "pP",
    chess.KNIGHT: "nN",
    chess.BISHOP: "bB",
    chess.ROOK: "rR",
    chess.QUEEN: "qQ",
    chess.KING: "kK",
}


def print_board(board: chess.Board) -> None:
    print()
    for rank in range(7, -1, -1):
        row = f"{rank + 1} "
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            if piece is None:
                row += ". "
            else:
                is_white = piece.color == chess.WHITE
                symbol = PIECE_ASCII[piece.piece_type][0 if is_white else 1]
                row += f"{symbol} "
        print(row)
    print("  a b c d e f g h")
    print()


def print_status(board: chess.Board) -> None:
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        print(f"Checkmate! {winner} wins.")
    elif board.is_stalemate():
        print("Stalemate! Draw.")
    elif board.is_insufficient_material():
        print("Draw by insufficient material.")
    elif board.is_fifty_moves():
        print("Draw by fifty-move rule.")
    elif board.is_repetition():
        print("Draw by repetition.")
    elif board.is_check():
        print("Check!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play NeuralChess")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--movetime", type=int, default=1000, help="Engine think time in ms"
    )
    parser.add_argument(
        "--color", choices=["white", "black", "random"], default="white"
    )
    args = parser.parse_args()

    engine = NeuralEngine(args.checkpoint)
    board = chess.Board()

    if args.color == "random":
        import random

        player_color = chess.WHITE if random.random() < 0.5 else chess.BLACK
    else:
        player_color = chess.WHITE if args.color == "white" else chess.BLACK

    engine_color = chess.BLACK if player_color == chess.WHITE else chess.WHITE

    print(f"You play as {'White' if player_color == chess.WHITE else 'Black'}")
    print("Enter moves in UCI format (e.g., e2e4) or 'quit' to exit")
    print_board(board)

    while not board.is_game_over():
        if board.turn == player_color:
            try:
                move_str = input("Your move: ").strip()
            except EOFError:
                break

            if move_str.lower() == "quit":
                break

            try:
                move = chess.Move.from_uci(move_str)
            except ValueError:
                print("Invalid move format. Use UCI format (e.g., e2e4)")
                continue

            if move not in board.legal_moves:
                print("Illegal move")
                continue

            board.push(move)
        else:
            print("Engine thinking...")
            start = time.time()
            move, score, pv = engine.search(board, movetime_ms=args.movetime)
            elapsed = time.time() - start

            if move is not None:
                board.push(move)
                pv_str = " ".join(m.uci() for m in pv[:5])
                print(
                    f"Engine plays: {move.uci()} (score: {score * 100:.1f}%, time: {elapsed:.1f}s)"
                )
                if pv_str:
                    print(f"PV: {pv_str}")

        print_board(board)
        print_status(board)

    if board.is_game_over():
        result = board.result()
        print(f"\nGame over. Result: {result}")


if __name__ == "__main__":
    main()
