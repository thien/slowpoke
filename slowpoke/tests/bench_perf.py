"""
Benchmark for hot-path functions in the Rust CheckerBoard backend.

Measures per-call timing for get_board_pos_weighted,
push_move/pop_move, is_over, and combined search round simulation.
"""

import time
import random
import numpy as np
from slowpoke.core.checkers import CheckerBoard, Black, White


def _make_boards(count=2000, max_moves=40):
    boards = []
    for _ in range(count):
        B = CheckerBoard()
        for _ in range(random.randint(0, max_moves)):
            m = B.get_moves()
            if m:
                B.push_move(random.choice(m))
        boards.append(B)
    return boards


def bench_get_board_pos_weighted(boards):
    """Benchmark Rust get_board_pos_weighted."""
    weights = {"Black": 1, "White": -1, "empty": 0, "blackKing": 1.5, "whiteKing": -1.5}
    N = len(boards)
    t0 = time.perf_counter()
    for b in boards:
        b.get_board_pos_weighted(Black, weights)
        b.get_board_pos_weighted(White, weights)
    t1 = time.perf_counter()
    us_per = (t1 - t0) / (N * 2) * 1e6
    print(f"get_board_pos_weighted: {us_per:.2f}us/call ({N * 2} calls)")
    return us_per


def bench_search_round(boards):
    """Benchmark a full search round: push + eval Black + eval White + pop."""
    weights = {"Black": 1, "White": -1, "empty": 0, "blackKing": 1.5, "whiteKing": -1.5}
    N = min(2000, len(boards))

    t0 = time.perf_counter()
    for b in boards[:N]:
        m = b.get_moves()
        if m:
            b.push_move(m[0])
            b.get_board_pos_weighted(Black, weights)
            b.get_board_pos_weighted(White, weights)
            b.pop_move()
    t1 = time.perf_counter()
    us_per = (t1 - t0) / N * 1e6
    print(f"Search round (push+eval*2+pop): {us_per:.1f}us")
    return us_per


def bench_push_pop(boards):
    """Benchmark push/pop throughput."""
    N = min(2000, len(boards))

    t0 = time.perf_counter()
    for b in boards[:N]:
        m = b.get_moves()
        if m:
            b.push_move(m[0])
    t1 = time.perf_counter()
    push_t = (t1 - t0) / N * 1e6

    for b in boards[:N]:
        b.pop_move()

    t0 = time.perf_counter()
    for b in boards[:N]:
        b.pop_move()
    t1 = time.perf_counter()
    pop_t = (t1 - t0) / N * 1e6

    print(f"push_move: {push_t:.1f}us")
    print(f"pop_move:  {pop_t:.1f}us")
    return push_t, pop_t


if __name__ == "__main__":
    random.seed(42)
    print("Building boards...", end=" ", flush=True)
    boards = _make_boards(2000, 40)
    print(f"{len(boards)} boards ready")
    print()

    print("=== get_board_pos_weighted ===")
    bench_get_board_pos_weighted(boards)
    print()

    print("=== push_move / pop_move ===")
    bench_push_pop(boards)
    print()

    print("=== Search round ===")
    bench_search_round(boards)
    print()

    print("Benchmark complete.")
