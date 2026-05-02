#!/usr/bin/env python3
"""
Quick wrapper that runs the benchmark and forces unbuffered stdout.
"""

import sys, os

# Ensure we're in the project root
os.chdir("/Users/t/projects/slowpoke")
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Strip flags so -u propagates to child
os.environ["PYTHONUNBUFFERED"] = "1"

from library.tests.bench_tournament_reuse import run

print("=" * 80)
print("  TREE REUSE BENCHMARK — ply=12, baseRound=300, 3 games, MLX=ON")
print("=" * 80)

# Run just REUSE ON since we already have REUSE OFF data
# (72 moves, 14.6s; 100 moves, 20.6s; 94 moves, 19.8s -> ~200ms/move avg)

import time
from library.core.checkers import CheckerBoard, Black
from library.decision.tmcts import TMCTS
from library.agents.evaluator.neural import NeuralNetwork
import statistics


def make_evaluator(use_mlx=True):
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
    t = TMCTS(ply, evaluator, debug=False, batch_size=batch_size)
    t.baseRound = base_round
    t.nn = nn
    t.use_mlx = True
    return t


n_games = 3
on_all = []

for g in range(n_games):
    nn, evaluator = make_evaluator()
    tmcts = make_tmcts(12, nn, evaluator)
    B = CheckerBoard()
    colour = Black
    move_times = []
    cache_rates = []
    cache_sizes = []

    t0 = time.perf_counter()
    while not B.is_over():
        moves = B.get_moves()
        if not moves:
            break

        hits_before = tmcts._cache_hits
        start = time.perf_counter()
        move = tmcts.decide(B, colour)
        elapsed = time.perf_counter() - start

        turn_hits = tmcts._cache_hits - hits_before
        rounds_this_move = tmcts.baseRound * tmcts.ply
        hit_rate = round((turn_hits / max(rounds_this_move, 1)) * 100, 1)

        move_times.append(elapsed)
        cache_rates.append(hit_rate)
        cache_sizes.append(len(tmcts._node_cache))
        B.make_move(move)
        colour = B.active

    wall = time.perf_counter() - t0
    r = {
        "moves": len(move_times),
        "total": sum(move_times),
        "avg": statistics.mean(move_times),
        "median": statistics.median(move_times),
        "times": move_times,
        "cache_rates": cache_rates,
        "avg_cache": statistics.mean(cache_rates) if cache_rates else 0,
        "cache_size": len(tmcts._node_cache),
    }
    on_all.append(r)

    print(
        f"  Game {g + 1:>2d}/{n_games}: {r['moves']:>2d} moves, "
        f"{r['total']:>6.1f}s total, "
        f"{r['avg'] * 1000:>6.1f}ms avg/move, "
        f"cache={r['avg_cache']:>5.1f}% hit, "
        f"{r['cache_size']} entries "
        f"(wall: {wall:.1f}s)"
    )

# Summary
print(f"\n{' SUMMARY (vs REUSE OFF from prev run) ':=^80s}")

# REUSE OFF data from previous run
off_data = {
    "times": {
        "avg": [0.2024, 0.2063, 0.2101],
        "total": [14.6, 20.6, 19.8],
        "moves": [72, 100, 94],
    },
}

off_avg_avg = statistics.mean(off_data["times"]["avg"])
on_avg_avg = statistics.mean([r["avg"] for r in on_all])
off_total = sum(off_data["times"]["total"])
on_total = sum(r["total"] for r in on_all)

speedup_avg = off_avg_avg / on_avg_avg
speedup_total = off_total / on_total

print(f"  {'Measure':<30s} {'REUSE OFF':>14s} {'REUSE ON':>14s} {'Speedup':>12s}")
print(f"  {'─' * 30} {'─' * 14} {'─' * 14} {'─' * 12}")
print(
    f"  {'Avg total time':<30s}"
    f" {off_total / n_games:>7.1f}s     "
    f" {on_total / n_games:>7.1f}s     "
    f" {speedup_total:>5.2f}x"
)
print(
    f"  {'Avg per-move':<30s}"
    f" {off_avg_avg * 1000:>8.1f}ms   {on_avg_avg * 1000:>8.1f}ms   {speedup_avg:>5.2f}x"
)
print(
    f"  {'Avg cache hit rate':<30s} {'N/A':>14s}"
    f" {statistics.mean([r['avg_cache'] for r in on_all]):>12.1f}%"
)

# Cache warmup profile
print(f"\n  Cache warmup (avg over {n_games} games):")
max_moves = max(len(r["cache_rates"]) for r in on_all)
for m in range(min(max_moves, 10)):  # first 10 moves
    rates = [r["cache_rates"][m] for r in on_all if m < len(r["cache_rates"])]
    print(f"    Move {m + 1:>2d}: avg {statistics.mean(rates):.1f}% hit rate")

print(f"\n  >> Tree reuse provides {speedup_avg:.1f}x avg speedup\n")
