"""
Benchmark NeuralChess against multiple Stockfish skill levels
with robust Elo estimation via logistic curve fitting.
"""

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import chess
import chess.engine
import chess.pgn
from tqdm import tqdm

MATE_CP = 20000
BLUNDER_THRESHOLD_CP = 200

STOCKFISH_ELO: dict[int, int] = {
    0: 1242,
    1: 1315,
    2: 1390,
    3: 1470,
    4: 1555,
    5: 1650,
    6: 1750,
    7: 1860,
    8: 1970,
    9: 2080,
    10: 2190,
    11: 2280,
    12: 2370,
    13: 2450,
    14: 2530,
    15: 2600,
    16: 2670,
    17: 2720,
    18: 2757,
    19: 2930,
    20: 3100,
}


@dataclass
class MoveStats:
    move_number: int
    side: str
    engine_name: str
    move_uci: str
    depth: int
    think_time_ms: float
    score_cp: float
    nodes: int
    is_capture: bool
    is_blunder: bool


@dataclass
class GameResult:
    game_number: int
    skill_level: int
    neural_color: str
    result: str
    neural_result: str
    termination: str
    total_moves: int
    move_stats: list[MoveStats] = field(default_factory=list)
    avg_neural_cpl: float = 0.0
    avg_neural_depth: float = 0.0
    avg_neural_think_ms: float = 0.0
    blunder_count: int = 0


@dataclass
class LevelResult:
    skill_level: int
    known_elo: int
    games: int
    wins: int
    draws: int
    losses: int
    score_pct: float
    game_results: list[GameResult] = field(default_factory=list)


def score_to_cp(score: chess.engine.Score) -> float:
    cp = score.score(mate_score=MATE_CP)
    if cp is None:
        return 0.0
    return float(cp)


def expected_score(neural_elo: float, opponent_elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent_elo - neural_elo) / 400.0))


def log_likelihood(neural_elo: float, level_results: list[LevelResult]) -> float:
    ll = 0.0
    for lr in level_results:
        if lr.games == 0:
            continue
        e = expected_score(neural_elo, lr.known_elo)
        e = max(1e-10, min(1.0 - 1e-10, e))
        ll += lr.wins * math.log(e)
        ll += lr.losses * math.log(1.0 - e)
        if lr.draws > 0:
            draw_prob = 2.0 * e * (1.0 - e)
            draw_prob = max(1e-10, min(1.0 - 1e-10, draw_prob))
            ll += lr.draws * math.log(draw_prob)
    return ll


def fit_elo(
    level_results: list[LevelResult],
) -> tuple[float, float, float]:
    elo_lo = 800.0
    elo_hi = 3600.0
    for _ in range(100):
        mid = (elo_lo + elo_hi) / 2.0
        if log_likelihood(mid, level_results) > log_likelihood(
            mid + 0.001, level_results
        ):
            elo_hi = mid
        else:
            elo_lo = mid
    best_elo = (elo_lo + elo_hi) / 2.0

    delta = 1.0
    ll_center = log_likelihood(best_elo, level_results)
    ll_plus = log_likelihood(best_elo + delta, level_results)
    ll_minus = log_likelihood(best_elo - delta, level_results)
    curvature = -(ll_plus - 2 * ll_center + ll_minus) / (delta * delta)

    if curvature > 0:
        se = 1.0 / math.sqrt(curvature)
    else:
        se = 100.0
    z = 1.96

    return best_elo, best_elo - z * se, best_elo + z * se


def play_game(
    neural_engine: chess.engine.SimpleEngine,
    stockfish_engine: chess.engine.SimpleEngine,
    game_number: int,
    skill_level: int,
    time_per_move: float,
    save_pgn: bool = False,
    move_callback: Optional[
        Callable[[int, str, str, str, float, int, float], None]
    ] = None,
) -> tuple[GameResult, Optional[chess.pgn.Game]]:
    neural_is_white = game_number % 2 == 0
    board = chess.Board()

    neural_engine.configure({})
    stockfish_engine.configure({"Skill Level": skill_level})

    move_stats: list[MoveStats] = []
    prev_neural_score_cp: Optional[float] = None
    move_number = 1

    pgn_root: Optional[chess.pgn.Game] = None
    pgn_node: chess.pgn.GameNode = None  # type: ignore[assignment]

    if save_pgn:
        pgn_root = chess.pgn.Game()
        pgn_root.headers["Event"] = "NeuralChess Benchmark"
        pgn_root.headers["Site"] = "Local"
        pgn_root.headers["White"] = (
            "NeuralChess" if neural_is_white else f"Stockfish (Skill {skill_level})"
        )
        pgn_root.headers["Black"] = (
            f"Stockfish (Skill {skill_level})" if neural_is_white else "NeuralChess"
        )
        pgn_root.headers["TimeControl"] = f"{time_per_move}/move"
        pgn_node = pgn_root

    while not board.is_game_over():
        is_white_turn = board.turn == chess.WHITE
        current_engine = (
            neural_engine if (is_white_turn == neural_is_white) else stockfish_engine
        )
        engine_name = "NeuralChess" if current_engine is neural_engine else "Stockfish"
        side = "white" if is_white_turn else "black"

        limit = chess.engine.Limit(time=time_per_move)
        info_flags = chess.engine.INFO_SCORE | chess.engine.INFO_PV

        start = time.monotonic()
        result = current_engine.play(board, limit, info=info_flags)
        think_ms = (time.monotonic() - start) * 1000.0

        move = result.move
        if move is None:
            break

        info = result.info or {}
        score_obj = info.get("score")
        pov_score = score_obj.pov(board.turn) if score_obj else chess.engine.Cp(0)
        score_cp = score_to_cp(pov_score)

        depth = info.get("depth", 0)
        nodes = info.get("nodes", 0)
        is_capture = board.is_capture(move)

        is_blunder = False
        if engine_name == "NeuralChess" and prev_neural_score_cp is not None:
            score_drop = prev_neural_score_cp - score_cp
            if score_drop > BLUNDER_THRESHOLD_CP:
                is_blunder = True

        ms = MoveStats(
            move_number=move_number,
            side=side,
            engine_name=engine_name,
            move_uci=move.uci(),
            depth=depth,
            think_time_ms=round(think_ms, 1),
            score_cp=round(score_cp, 1),
            nodes=nodes,
            is_capture=is_capture,
            is_blunder=is_blunder,
        )
        move_stats.append(ms)

        if move_callback:
            move_callback(
                len(move_stats),
                engine_name,
                move.uci(),
                side,
                score_cp,
                depth,
                think_ms,
            )

        if save_pgn and pgn_node is not None:
            pgn_node = pgn_node.add_variation(move)

        if engine_name == "NeuralChess":
            prev_neural_score_cp = score_cp

        board.push(move)

        if board.turn == chess.WHITE:
            move_number += 1

    result_str = board.result()
    if result_str == "1-0":
        neural_result = "win" if neural_is_white else "loss"
    elif result_str == "0-1":
        neural_result = "loss" if neural_is_white else "win"
    else:
        neural_result = "draw"

    if board.is_checkmate():
        termination = "checkmate"
    elif board.is_stalemate():
        termination = "stalemate"
    elif board.is_insufficient_material():
        termination = "insufficient material"
    elif board.can_claim_fifty_moves():
        termination = "fifty-move rule"
    elif board.is_repetition():
        termination = "repetition"
    else:
        termination = "other"

    neural_moves_stats = [m for m in move_stats if m.engine_name == "NeuralChess"]

    avg_depth = 0.0
    avg_think = 0.0
    avg_cpl = 0.0
    blunder_count = 0

    if neural_moves_stats:
        avg_depth = statistics.mean(m.depth for m in neural_moves_stats)
        avg_think = statistics.mean(m.think_time_ms for m in neural_moves_stats)
        blunder_count = sum(1 for m in neural_moves_stats if m.is_blunder)

    if save_pgn and pgn_root is not None:
        pgn_root.headers["Result"] = result_str
        pgn_root.headers["Termination"] = termination

    return (
        GameResult(
            game_number=game_number,
            skill_level=skill_level,
            neural_color="white" if neural_is_white else "black",
            result=result_str,
            neural_result=neural_result,
            termination=termination,
            total_moves=len(move_stats),
            move_stats=move_stats,
            avg_neural_cpl=round(avg_cpl, 1),
            avg_neural_depth=round(avg_depth, 1),
            avg_neural_think_ms=round(avg_think, 1),
            blunder_count=blunder_count,
        ),
        pgn_root,
    )


def print_multi_level_summary(
    level_results: list[LevelResult],
    all_game_results: list[GameResult],
    time_per_move: float,
    elo_estimate: float,
    elo_lo: float,
    elo_hi: float,
) -> None:
    total_games = sum(lr.games for lr in level_results)
    total_wins = sum(lr.wins for lr in level_results)
    total_draws = sum(lr.draws for lr in level_results)
    total_losses = sum(lr.losses for lr in level_results)

    print()
    print("=" * 70)
    print("  NeuralChess Multi-Level Benchmark")
    print("=" * 70)
    print(f"Total games: {total_games} | Time: {time_per_move}s/move")
    print()

    print("Per-Level Results:")
    print(
        f"  {'Skill':>6} {'ELO':>6} {'Games':>6} {'W':>4} {'D':>4} {'L':>4} {'Score%':>7}"
    )
    print("  " + "-" * 43)
    for lr in sorted(level_results, key=lambda x: x.skill_level):
        if lr.games > 0:
            print(
                f"  {lr.skill_level:>6} {lr.known_elo:>6} {lr.games:>6} "
                f"{lr.wins:>4} {lr.draws:>4} {lr.losses:>4} {lr.score_pct:>6.1f}%"
            )
    print()

    print("Elo Estimation (Maximum Likelihood):")
    print(f"  Estimated Elo: {elo_estimate:.0f} [{elo_lo:.0f}, {elo_hi:.0f}] (95% CI)")
    print()

    all_neural_depth = [r.avg_neural_depth for r in all_game_results]
    all_neural_think = [r.avg_neural_think_ms for r in all_game_results]
    total_blunders = sum(r.blunder_count for r in all_game_results)
    total_neural_moves = sum(
        len([m for m in r.move_stats if m.engine_name == "NeuralChess"])
        for r in all_game_results
    )

    if all_neural_depth:
        print("Performance Metrics:")
        print(f"  Avg Search Depth:    {statistics.mean(all_neural_depth):>6.1f}")
        print(f"  Avg Think Time:      {statistics.mean(all_neural_think):>6.0f}ms")
    if total_neural_moves > 0:
        print(
            f"  Blunder Rate:        {total_blunders / total_neural_moves * 100:>6.1f}%"
        )
    print()

    print("Score vs Opponent Elo:")
    for lr in sorted(level_results, key=lambda x: x.skill_level):
        if lr.games > 0:
            marker = ">" if lr.score_pct > 50 else ("<" if lr.score_pct < 50 else "=")
            bar_len = int(lr.score_pct / 2)
            bar = "#" * bar_len
            print(
                f"  SF{lr.skill_level:<3} ({lr.known_elo:>4} Elo): {lr.score_pct:>5.1f}% {marker} {bar}"
            )
    print("=" * 70)


def run_benchmark(
    neural_cmd: list[str],
    stockfish_path: str,
    skill_levels: list[int],
    games_per_level: int,
    time_per_move: float,
    output_path: Optional[str],
    pgn_path: Optional[str],
) -> None:
    elo_labels = ", ".join(f"L{lv}({STOCKFISH_ELO[lv]})" for lv in skill_levels)
    tqdm.write("Starting NeuralChess Multi-Level Benchmark...")
    tqdm.write(f"  Skill levels: {elo_labels}")
    tqdm.write(f"  Games per level: {games_per_level}")
    tqdm.write(f"  Time per move: {time_per_move}s")
    tqdm.write("")

    try:
        neural_engine = chess.engine.SimpleEngine.popen_uci(neural_cmd)
    except Exception as e:
        tqdm.write(f"ERROR: Failed to start NeuralChess engine: {e}")
        tqdm.write(f"  Command: {' '.join(neural_cmd)}")
        sys.exit(1)

    try:
        stockfish_engine = chess.engine.SimpleEngine.popen_uci([stockfish_path])
    except FileNotFoundError:
        tqdm.write(f"ERROR: Stockfish not found at '{stockfish_path}'")
        tqdm.write("  Install Stockfish: sudo apt install stockfish  (Linux)")
        tqdm.write("                     brew install stockfish     (macOS)")
        neural_engine.quit()
        sys.exit(1)
    except Exception as e:
        tqdm.write(f"ERROR: Failed to start Stockfish: {e}")
        neural_engine.quit()
        sys.exit(1)

    pgn_games: list[chess.pgn.Game] = []
    all_game_results: list[GameResult] = []
    level_results: list[LevelResult] = []
    interrupted = False

    def save_results() -> None:
        if not all_game_results:
            return

        elo_est, elo_lo, elo_hi = fit_elo(level_results)

        print_multi_level_summary(
            level_results, all_game_results, time_per_move, elo_est, elo_lo, elo_hi
        )

        if output_path:
            output_data = {
                "config": {
                    "skill_levels": skill_levels,
                    "games_per_level": games_per_level,
                    "time_per_move": time_per_move,
                    "stockfish_elo_reference": STOCKFISH_ELO,
                },
                "elo_estimate": {
                    "elo": round(elo_est, 1),
                    "ci_lo": round(elo_lo, 1),
                    "ci_hi": round(elo_hi, 1),
                },
                "per_level": [
                    {
                        "skill_level": lr.skill_level,
                        "known_elo": lr.known_elo,
                        "games": lr.games,
                        "wins": lr.wins,
                        "draws": lr.draws,
                        "losses": lr.losses,
                        "score_pct": round(lr.score_pct, 1),
                    }
                    for lr in level_results
                ],
                "games": [
                    {
                        "game_number": r.game_number,
                        "skill_level": r.skill_level,
                        "neural_color": r.neural_color,
                        "result": r.result,
                        "neural_result": r.neural_result,
                        "termination": r.termination,
                        "total_moves": r.total_moves,
                        "avg_neural_cpl": r.avg_neural_cpl,
                        "avg_neural_depth": r.avg_neural_depth,
                        "avg_neural_think_ms": r.avg_neural_think_ms,
                        "blunder_count": r.blunder_count,
                        "moves": [asdict(m) for m in r.move_stats],
                    }
                    for r in all_game_results
                ],
            }
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            tqdm.write(f"\nDetailed results saved to {output_path}")

        if pgn_path and pgn_games:
            with open(pgn_path, "w") as f:
                for pgn_game in pgn_games:
                    f.write(str(pgn_game) + "\n\n")
            tqdm.write(f"PGN games saved to {pgn_path}")

    def make_move_callback(
        game_num: int, skill: int
    ) -> Callable[[int, str, str, str, float, int, float], None]:
        def callback(
            move_count: int,
            engine_name: str,
            move_uci: str,
            side: str,
            score_cp: float,
            depth: int,
            think_ms: float,
        ) -> None:
            score_sign = "+" if score_cp >= 0 else ""
            side_prefix = "W" if side == "white" else "B"
            tqdm.write(
                f"    Move {move_count:>3} ({side_prefix}) {engine_name:<12} "
                f"{move_uci}  {score_sign}{score_cp:>6.0f}cp  d={depth}  {think_ms:>6.0f}ms"
            )

        return callback

    total_games = games_per_level * len(skill_levels)
    global_game = 0

    try:
        for skill in skill_levels:
            known_elo = STOCKFISH_ELO[skill]
            tqdm.write(f"\n{'=' * 50}")
            tqdm.write(f"  Skill Level {skill} (Elo ~{known_elo})")
            tqdm.write(f"{'=' * 50}")

            level_wins = 0
            level_draws = 0
            level_losses = 0
            level_game_results: list[GameResult] = []

            with tqdm(
                total=games_per_level,
                desc=f"  L{skill}({known_elo})",
                unit="game",
                bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt}",
            ) as pbar:
                for i in range(1, games_per_level + 1):
                    global_game += 1
                    game_result, pgn_game = play_game(
                        neural_engine,
                        stockfish_engine,
                        global_game,
                        skill,
                        time_per_move,
                        pgn_path is not None,
                        make_move_callback(global_game, skill),
                    )

                    all_game_results.append(game_result)
                    level_game_results.append(game_result)
                    if pgn_game:
                        pgn_games.append(pgn_game)

                    if game_result.neural_result == "win":
                        level_wins += 1
                    elif game_result.neural_result == "draw":
                        level_draws += 1
                    else:
                        level_losses += 1

                    result_marker = (
                        "W"
                        if game_result.neural_result == "win"
                        else ("D" if game_result.neural_result == "draw" else "L")
                    )
                    tqdm.write(
                        f"    Game {global_game:>2} [{result_marker}] {game_result.result}  "
                        f"{game_result.termination:<22}  {game_result.total_moves:>3} moves"
                    )
                    pbar.update(1)

            level_games = level_wins + level_draws + level_losses
            level_score = (
                (level_wins + 0.5 * level_draws) / level_games * 100
                if level_games > 0
                else 0
            )

            level_results.append(
                LevelResult(
                    skill_level=skill,
                    known_elo=known_elo,
                    games=level_games,
                    wins=level_wins,
                    draws=level_draws,
                    losses=level_losses,
                    score_pct=level_score,
                    game_results=level_game_results,
                )
            )

            tqdm.write(
                f"  Level {skill} complete: {level_wins}W/{level_draws}D/{level_losses}L ({level_score:.1f}%)"
            )

    except KeyboardInterrupt:
        interrupted = True
        tqdm.write("\nInterrupted by user.")
    finally:
        neural_engine.quit()
        stockfish_engine.quit()

    if interrupted:
        tqdm.write(
            f"Benchmark stopped after {len(all_game_results)}/{total_games} games."
        )
        tqdm.write("Saving partial results...")
        tqdm.write("")

    save_results()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark NeuralChess against multiple Stockfish skill levels"
    )
    parser.add_argument(
        "--neural-checkpoint",
        required=True,
        help="Path to NeuralChess model checkpoint",
    )
    parser.add_argument(
        "--stockfish-path",
        default="stockfish",
        help="Path to Stockfish binary (default: 'stockfish' from PATH)",
    )
    parser.add_argument(
        "--skill-levels",
        default="0,5,10,15,20",
        help="Comma-separated Stockfish skill levels (default: 0,5,10,15,20)",
    )
    parser.add_argument(
        "--games-per-level",
        type=int,
        default=10,
        help="Games to play at each skill level (default: 10)",
    )
    parser.add_argument(
        "--time-per-move",
        type=float,
        default=1.0,
        help="Fixed think time per move in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--output",
        help="Save detailed results as JSON",
    )
    parser.add_argument(
        "--save-pgn",
        help="Save all games as PGN file",
    )
    args = parser.parse_args()

    try:
        skill_levels = [int(x.strip()) for x in args.skill_levels.split(",")]
    except ValueError:
        print("ERROR: --skill-levels must be comma-separated integers")
        sys.exit(1)

    for skill in skill_levels:
        if skill < 0 or skill > 20:
            print(
                f"ERROR: Stockfish skill level must be between 0 and 20 (got {skill})"
            )
            sys.exit(1)

    uci_script = Path(__file__).parent.parent / "src" / "neuralchess" / "uci.py"
    neural_cmd = [
        sys.executable,
        str(uci_script),
        "--checkpoint",
        args.neural_checkpoint,
    ]

    run_benchmark(
        neural_cmd=neural_cmd,
        stockfish_path=args.stockfish_path,
        skill_levels=skill_levels,
        games_per_level=args.games_per_level,
        time_per_move=args.time_per_move,
        output_path=args.output,
        pgn_path=args.save_pgn,
    )


if __name__ == "__main__":
    main()
