"""
Benchmark: Tournament tree reuse at ply=12 with real NN + MLX.

Measures wall-clock time with tree reuse ON vs OFF across multiple
full tournament games at tournament-level settings.

Usage:
    cd /Users/t/projects/slowpoke
    .venv/bin/python -O library/tests/bench_tournament_reuse.py
    .venv/bin/python -O library/tests/bench_tournament_reuse.py --quick
    .venv/bin/python -O library/tests/bench_tournament_reuse.py --deeper
"""

import time
import argparse
import statistics

from slowpoke.core.checkers import CheckerBoard, Black
from slowpoke.search.tmcts import TMCTS
from slowpoke.agents.evaluator.neural import NeuralNetwork


def make_evaluator(use_mlx=True):
    """Create an evaluator object with real NeuralNetwork and MLX."""
    nn = NeuralNetwork([91, 40, 10, 1], use_mlx=use_mlx)
    evaluator = type(
        "Eval",
        (),
        {
            "nn": nn,
            "evaluate_board": lambda self, B, colour: 0.0,
        },
    )()
    return nn, evaluator


def make_tmcts(ply, nn, evaluator, base_round=300, batch_size=512):
    """Create a TMCTS instance wired to real NN."""
    t = TMCTS(ply, evaluator, debug=False, batch_size=batch_size)
    t.base_round = base_round
    t.nn = nn
    t.use_mlx = True
    return t


def play_game_reuse_on(ply, base_round):
    """One game with persistent node cache (tree reuse ON)."""
    nn, evaluator = make_evaluator()
    tmcts = make_tmcts(ply, nn, evaluator, base_round)
    B = CheckerBoard()
    colour = Black
    move_times = []
    cache_rates = []
    while not B.is_over():
        moves = B.get_moves()
        if not moves:
            break

        hits_before = tmcts._cache_hits

        start = time.perf_counter()
        move = tmcts.decide(B, colour)
        elapsed = time.perf_counter() - start

        turn_hits = tmcts._cache_hits - hits_before
        rounds_this_move = tmcts.base_round * tmcts.ply
        hit_rate = (turn_hits / max(rounds_this_move, 1)) * 100

        move_times.append(elapsed)
        cache_rates.append(hit_rate)
        B.make_move(move)

    return {
        "moves": len(move_times),
        "total": sum(move_times),
        "avg": statistics.mean(move_times),
        "median": statistics.median(move_times),
        "times": move_times,
        "cache_rates": cache_rates,
        "avg_cache": statistics.mean(cache_rates) if cache_rates else 0,
        "cache_size": len(tmcts._node_cache),
    }


def play_game_reuse_off(ply, base_round):
    """One game with NO tree reuse (fresh TMCTS each turn)."""
    B = CheckerBoard()
    colour = Black
    move_times = []

    while not B.is_over():
        moves = B.get_moves()
        if not moves:
            break
        nn, evaluator = make_evaluator()
        tmcts = make_tmcts(ply, nn, evaluator, base_round)
        start = time.perf_counter()
        move = tmcts.decide(B, colour)
        elapsed = time.perf_counter() - start
        move_times.append(elapsed)
        B.make_move(move)

    return {
        "moves": len(move_times),
        "total": sum(move_times),
        "avg": statistics.mean(move_times),
        "median": statistics.median(move_times),
        "times": move_times,
    }


def run(ply=12, base_round=300, n_games=5):
    """Run benchmark for both ON and OFF."""
    print(f"{'=' * 80}")
    print(
        f"  TREE REUSE BENCHMARK — ply={ply}, base_round={base_round}, "
        f"{n_games} games, MLX=ON"
    )
    print(f"  rounds/move = {base_round * ply:,}  batch_size = 512")
    print(f"{'=' * 80}")

    # --- OFF ---
    print(f"\n{' CONFIG: REUSE OFF (fresh TMCTS/turn) ':-^80s}")
    off_all = []
    for g in range(n_games):
        t0 = time.perf_counter()
        r = play_game_reuse_off(ply, base_round)
        wall = time.perf_counter() - t0
        off_all.append(r)
        print(
            f"  Game {g + 1:>2d}/{n_games}: {r['moves']:>2d} moves, "
            f"{r['total']:>6.1f}s total, "
            f"{r['avg'] * 1000:>6.1f}ms avg/move "
            f"(wall: {wall:.1f}s)"
        )

    # --- ON ---
    print(f"\n{' CONFIG: REUSE ON (persistent cache) ':-^80s}")
    on_all = []
    for g in range(n_games):
        t0 = time.perf_counter()
        r = play_game_reuse_on(ply, base_round)
        wall = time.perf_counter() - t0
        on_all.append(r)
        print(
            f"  Game {g + 1:>2d}/{n_games}: {r['moves']:>2d} moves, "
            f"{r['total']:>6.1f}s total, "
            f"{r['avg'] * 1000:>6.1f}ms avg/move, "
            f"cache={r['avg_cache']:>4.1f}% hit, "
            f"{r['cache_size']} entries "
            f"(wall: {wall:.1f}s)"
        )

    # --- Summary ---
    print(f"\n{' SUMMARY ':=^80s}")

    def avg_of(key, results):
        return statistics.mean([r[key] for r in results])

    def median_of(key, results):
        return statistics.median([r[key] for r in results])

    off_avg = avg_of("avg", off_all)
    on_avg = avg_of("avg", on_all)
    off_total = sum(r["total"] for r in off_all)
    on_total = sum(r["total"] for r in on_all)

    speedup_avg = off_avg / on_avg if on_avg > 0 else float("inf")
    speedup_total = off_total / on_total if on_total > 0 else float("inf")

    print(f"  {'Metric':<35s} {'REUSE OFF':>14s} {'REUSE ON':>14s} {'Speedup':>12s}")
    print(f"  {'─' * 35} {'─' * 14} {'─' * 14} {'─' * 12}")
    print(
        f"  {'Total wall time':<35s}"
        f" {sum(off_all[g]['total'] for g in range(n_games)):>7.1f}s     "
        f" {sum(on_all[g]['total'] for g in range(n_games)):>7.1f}s     "
        f" {speedup_total:>5.2f}x"
    )
    print(
        f"  {'Avg per-move':<35s}"
        f" {off_avg * 1000:>8.1f}ms   {on_avg * 1000:>8.1f}ms   {speedup_avg:>5.2f}x"
    )
    print(
        f"  {'Median per-move':<35s}"
        f" {median_of('median', off_all) * 1000:>8.1f}ms   "
        f" {median_of('median', on_all) * 1000:>8.1f}ms"
    )
    print(
        f"  {'Avg moves/game':<35s}"
        f" {avg_of('moves', off_all):>14.1f}  {avg_of('moves', on_all):>14.1f}"
    )
    print(
        f"  {'Avg cache hit rate':<35s} {'N/A':>14s}"
        f" {avg_of('avg_cache', on_all):>12.1f}%"
    )
    print(
        f"  {'Final cache size':<35s} {'N/A':>14s}"
        f" {int(avg_of('cache_size', on_all)):>14,d}"
    )

    # Per-move profile from first game
    print("\n  First-game per-move detail:")
    print(f"  {'Move':>5s} {'OFF(ms)':>10s} {'ON(ms)':>10s} {'Spd':>7s} {'Cache%':>7s}")
    print(f"  {'─' * 5} {'─' * 10} {'─' * 10} {'─' * 7} {'─' * 7}")
    for i in range(min(len(off_all[0]["times"]), len(on_all[0]["times"]))):
        ot = off_all[0]["times"][i] * 1000
        nt = on_all[0]["times"][i] * 1000
        sp = ot / nt if nt > 0 else float("inf")
        ch = on_all[0]["cache_rates"][i] if i < len(on_all[0]["cache_rates"]) else 0
        print(f"  {i + 1:>5d} {ot:>8.1f}ms {nt:>8.1f}ms {sp:>6.2f}x {ch:>5.1f}%")

    # Cache warmup breakdown
    print(f"\n  Cache warmup effect (avg over {n_games} games):")
    max_moves = max(len(r["cache_rates"]) for r in on_all)
    for m in range(max_moves):
        rates = [r["cache_rates"][m] for r in on_all if m < len(r["cache_rates"])]
        print(f"    Move {m + 1:>2d}: avg {statistics.mean(rates):.1f}% hit rate")

    if speedup_avg > 1.10:
        print(f"\n  >> Tree reuse provides {speedup_avg:.1f}x avg speedup")
    else:
        print(
            f"\n  >> Minimal impact ({speedup_avg:.2f}x) — "
            "tree structure may not overlap enough across turns"
        )

    return off_all, on_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--deeper", action="store_true")
    p.add_argument("--games", type=int, default=None)
    args = p.parse_args()

    n_games = args.games or (3 if args.quick else 5)
    ply = 16 if args.deeper else 12
    run(ply=ply, base_round=300, n_games=n_games)


if __name__ == "__main__":
    main()
