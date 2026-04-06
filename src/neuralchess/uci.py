"""
UCI protocol handler for NeuralChess engine.
"""

import argparse
import logging

import chess
import torch

from neuralchess.engine import NeuralEngine
from neuralchess.models import load_model

logger = logging.getLogger(__name__)


class UCIHandler:
    def __init__(self, engine: NeuralEngine) -> None:
        self.engine = engine
        self.board = chess.Board()

    def _send(self, message: str) -> None:
        """Send a message to the GUI and log it."""
        print(message, flush=True)
        logger.info(f">> {message}")

    def loop(self) -> None:
        while True:
            try:
                line = input().strip()
            except EOFError:
                break
            if not line:
                continue

            logger.info(f"<< {line}")
            parts = line.split()
            cmd = parts[0]

            if cmd == "uci":
                self._handle_uci()
            elif cmd == "isready":
                self._handle_isready()
            elif cmd == "ucinewgame":
                self._handle_ucinewgame()
            elif cmd == "position":
                self._handle_position(parts[1:])
            elif cmd == "go":
                self._handle_go(parts[1:])
            elif cmd == "stop":
                self._handle_stop()
            elif cmd == "quit":
                self._handle_stop()
                break
            elif cmd == "setoption":
                self._handle_setoption(parts[1:])

    def _handle_uci(self) -> None:
        self._send("id name NeuralChess")
        self._send("id author NeuralChess")
        self._send("uciok")

    def _handle_isready(self) -> None:
        self._send("readyok")

    def _handle_ucinewgame(self) -> None:
        self._handle_stop()
        self.board = chess.Board()

    def _handle_position(self, args: list[str]) -> None:
        self._handle_stop()
        self.board = chess.Board()

        i = 0
        while i < len(args):
            if args[i] == "startpos":
                self.board = chess.Board()
                i += 1
            elif args[i] == "fen":
                fen = " ".join(args[i + 1 : i + 7])
                self.board = chess.Board(fen)
                i += 7
            elif args[i] == "moves":
                i += 1
                while i < len(args):
                    move = chess.Move.from_uci(args[i])
                    if move in self.board.legal_moves:
                        self.board.push(move)
                    i += 1
                break
            else:
                i += 1

    def _handle_go(self, args: list[str]) -> None:
        self._handle_stop()
        score, best_move = self.engine.evaluate_position(self.board)

        if best_move is not None:
            self._send(f"bestmove {best_move.uci()}")
        else:
            logger.error(f"search failed")
            self._send("bestmove 0000")

    def _handle_stop(self) -> None:
        pass

    def _handle_setoption(self, args: list[str]) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuralChess UCI engine")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--debug", action="store_true", help="Enable verbose debug logging"
    )
    parser.add_argument("--log", help="Path to log file")
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    handlers = []
    if args.log:
        handlers.append(logging.FileHandler(args.log, mode="w"))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    engine = NeuralEngine(model=model, device=device)
    handler = UCIHandler(engine)
    handler.loop()


if __name__ == "__main__":
    main()
