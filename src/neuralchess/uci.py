"""
UCI protocol handler for NeuralChess engine.
"""

import argparse
import logging
import sys
import threading
from typing import Optional

import chess
import torch

from neuralchess.engine import NeuralEngine
from neuralchess.models import load_model

logger = logging.getLogger(__name__)


class UCIHandler:
    def __init__(self, engine: NeuralEngine, debug: bool = False) -> None:
        self.engine = engine
        self.board = chess.Board()
        self.debug = debug

        self.stop_event = threading.Event()
        self.search_thread: Optional[threading.Thread] = None
        self.timer: Optional[threading.Timer] = None

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
        self._send("option name movetime type spin default 1000 min 10 max 60000")
        self._send("option name debug type check default false")
        self._send("uciok")

    def _handle_isready(self) -> None:
        self._send("readyok")

    def _handle_ucinewgame(self) -> None:
        self._handle_stop()
        self.board = chess.Board()
        self.engine.tt.clear()

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

        logger.debug(f"position set:\n{self.board.fen()}")

    def _handle_go(self, args: list[str]) -> None:
        self._handle_stop()

        movetime: Optional[int] = None
        wtime: Optional[int] = None
        btime: Optional[int] = None
        depth_limit: int = 64

        i = 0
        while i < len(args):
            if args[i] == "movetime" and i + 1 < len(args):
                movetime = int(args[i + 1])
                i += 2
            elif args[i] == "wtime" and i + 1 < len(args):
                wtime = int(args[i + 1])
                i += 2
            elif args[i] == "btime" and i + 1 < len(args):
                btime = int(args[i + 1])
                i += 2
            elif args[i] == "depth" and i + 1 < len(args):
                depth_limit = int(args[i + 1])
                i += 2
            elif args[i] == "infinite":
                movetime = None  # No timer
                i += 1
            else:
                i += 1

        if movetime is not None:
            time_limit_ms = movetime
        elif wtime is not None and btime is not None:
            remaining = wtime if self.board.turn == chess.WHITE else btime
            time_limit_ms = max(remaining // 10, 100)
        else:
            time_limit_ms = None

        time_limit_ms = 5000

        self.stop_event.clear()
        self.search_thread = threading.Thread(
            target=self._search_worker,
            args=(self.board.copy(), depth_limit),
            daemon=True,
        )

        if time_limit_ms is not None:
            self.timer = threading.Timer(
                time_limit_ms / 1000.0, lambda: self.stop_event.set()
            )
            self.timer.start()

        logger.warning(f"staring search for {time_limit_ms}ms")
        self.search_thread.start()


    def _search_worker(self, board: chess.Board, max_depth: int) -> None:
        best_move, score, pv = self.engine.search(
            board, max_depth=max_depth, stop_event=self.stop_event
        )

        if best_move is not None:
            self._send(f"bestmove {best_move.uci()}")
        else:
            logger.error(f"search failed")
            self._send("bestmove 0000")

    def _handle_stop(self) -> None:
        if self.timer:
            self.timer.cancel()
            self.timer = None

        self.stop_event.set()
        if self.search_thread and self.search_thread.is_alive():
            self.search_thread.join()
        self.search_thread = None

    def _handle_setoption(self, args: list[str]) -> None:
        i = 0
        while i < len(args):
            if args[i] == "name" and i + 1 < len(args):
                name = args[i + 1].lower()
                if name == "movetime" and i + 3 < len(args) and args[i + 2] == "value":
                    self.movetime_ms = int(args[i + 3])
                elif name == "debug" and i + 3 < len(args) and args[i + 2] == "value":
                    self.debug = args[i + 3].lower() == "true"
                i += 4
            else:
                i += 1


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
        handlers.append(logging.FileHandler(args.log))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    engine = NeuralEngine(model=model, device=device, debug=args.debug)
    handler = UCIHandler(engine, debug=args.debug)
    handler.loop()


if __name__ == "__main__":
    main()
