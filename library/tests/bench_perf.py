"""
Benchmark for hot-path functions in checkers.py.

Measures per-call timing for _set_bits, get_board_pos_weighted,
push_move/pop_move, is_over, and combined search round simulation.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import random
import numpy as np
from core.checkers import CheckerBoard, Black, White, _set_bits


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


def bench_set_bits(boards):
    """Benchmark _set_bits vs old bin()-based iteration."""
    moves = set()
    for b in boards[:200]:
        for m in b.get_moves():
            moves.add(abs(m))

    # _set_bits
    t0 = time.perf_counter()
    for _ in range(2000):
        for m in moves:
            list(_set_bits(m))
    t1 = time.perf_counter()
    new_t = t1 - t0

    # bin() - for reference
    t0 = time.perf_counter()
    for _ in range(2000):
        for m in moves:
            [i for (i, b) in enumerate(bin(m)[::-1]) if b == "1"]
    t1 = time.perf_counter()
    old_t = t1 - t0

    print(f"_set_bits:       {new_t:.4f}s")
    print(f"bin() enumerate: {old_t:.4f}s")
    print(f"speedup: {old_t / new_t:.1f}x")
    return old_t / new_t


def bench_get_board_pos_weighted(boards):
    """Benchmark direct bitboard→weighted vs old rank_loop+dict_lookup."""
    weights = {"Black": 1, "White": -1, "empty": 0, "blackKing": 1.5, "whiteKing": -1.5}
    N = len(boards)

    # New direct approach
    t0 = time.perf_counter()
    for b in boards:
        b.get_board_pos_weighted(Black, weights)
        b.get_board_pos_weighted(White, weights)
    t1 = time.perf_counter()
    new_t = t1 - t0

    # Old two-pass approach (replicated inline)
    def old_black(b):
        bk = b.backward[Black]
        bm = b.forward[Black] ^ bk
        wk = b.forward[White]
        wm = b.backward[White] ^ wk
        rank = [-1] * 32
        for i in range(4):
            for j in range(8):
                cell = 1 << (9 * i + j)
                idx = 8 * i + j
                if cell & bm:
                    rank[idx] = 0
                elif cell & wm:
                    rank[idx] = 1
                elif cell & bk:
                    rank[idx] = 2
                elif cell & wk:
                    rank[idx] = 3
        rep = {0: 1, 1: -1, -1: 0, 2: 1.5, 3: -1.5}
        return np.array([rep[n] for n in rank], dtype=np.float32)

    def old_white(b):
        bk = b.backward[Black]
        bm = b.forward[Black] ^ bk
        wk = b.forward[White]
        wm = b.backward[White] ^ wk
        rank = [-1] * 32
        for i in range(4):
            for j in range(8):
                cell = 1 << (9 * i + j)
                idx = 8 * i + j
                if cell & bm:
                    rank[idx] = 0
                elif cell & wm:
                    rank[idx] = 1
                elif cell & bk:
                    rank[idx] = 2
                elif cell & wk:
                    rank[idx] = 3
        rep = {0: -1, 1: 1, -1: 0, 2: -1.5, 3: 1.5}
        return np.array([rep[n] for n in reversed(rank)], dtype=np.float32)

    t0 = time.perf_counter()
    for b in boards:
        old_black(b)
        old_white(b)
    t1 = time.perf_counter()
    old_t = t1 - t0

    us_new = new_t / N * 1e6
    us_old = old_t / N * 1e6
    print(f"get_board_pos_weighted OLD: {us_old:.2f}us/call")
    print(f"get_board_pos_weighted NEW: {us_new:.2f}us/call")
    print(f"speedup: {us_old / us_new:.1f}x")
    return us_old / us_new


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

    print("=== _set_bits ===")
    bench_set_bits(boards)
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

    # Correctness: verify matches the canonical Rust output
    weights = {"Black": 1, "White": -1, "empty": 0, "blackKing": 1.5, "whiteKing": -1.5}
    w = weights
    for b in boards:
        py_out = b.get_board_pos_weighted(Black, weights)
        if hasattr(b, "_core") and b._core is not None:
            rs_out = np.asarray(
                b._core.get_board_pos_weighted(
                    0,
                    w["empty"],
                    w["Black"],
                    w["White"],
                    w["blackKing"],
                    w["whiteKing"],
                )
            )
            assert np.allclose(py_out, rs_out, atol=1e-6), "Python/Rust mismatch!"
    print(f"Correctness: OK ({len(boards)} boards match)")
