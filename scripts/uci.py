"""
UCI protocol handler for NeuralChess engine.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import chess

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuralchess.engine import NeuralEngine


class UCIHandler:
    def __init__(self, engine: NeuralEngine) -> None:
        self.engine = engine
        self.board = chess.Board()
        self.movetime_ms: int = 1000
        self.searching = False

    def loop(self) -> None:
        while True:
            try:
                line = input().strip()
            except EOFError:
                break
            if not line:
                continue
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
                break
            elif cmd == "setoption":
                self._handle_setoption(parts[1:])

    def _handle_uci(self) -> None:
        print("id name NeuralChess")
        print("id author NeuralChess")
        print("option name movetime type spin default 1000 min 10 max 60000")
        print("uciok")

    def _handle_isready(self) -> None:
        print("readyok")

    def _handle_ucinewgame(self) -> None:
        self.board = chess.Board()
        self.engine.tt.clear()

    def _handle_position(self, args: list[str]) -> None:
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
        movetime: Optional[int] = None
        wtime: Optional[int] = None
        btime: Optional[int] = None

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
            elif args[i] == "infinite":
                movetime = 60000
                i += 1
            else:
                i += 1

        if movetime is None and wtime is None and btime is None:
            movetime = self.movetime_ms

        best_move, score, pv = self.engine.search(
            self.board,
            movetime_ms=movetime,
            wtime_ms=wtime,
            btime_ms=btime,
        )

        depth = len(pv)
        score_cp = int(score * 100)

        pv_str = " ".join(m.uci() for m in pv) if pv else ""
        print(f"info depth {depth} score cp {score_cp} pv {pv_str}".strip())

        if best_move is not None:
            print(f"bestmove {best_move.uci()}")
        else:
            print("bestmove 0000")

    def _handle_stop(self) -> None:
        pass

    def _handle_setoption(self, args: list[str]) -> None:
        i = 0
        while i < len(args):
            if args[i] == "name" and i + 1 < len(args):
                name = args[i + 1].lower()
                if name == "movetime" and i + 3 < len(args) and args[i + 2] == "value":
                    self.movetime_ms = int(args[i + 3])
                i += 4
            else:
                i += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuralChess UCI engine")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    engine = NeuralEngine(args.checkpoint)
    handler = UCIHandler(engine)
    handler.loop()


if __name__ == "__main__":
    main()
