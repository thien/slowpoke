"""
Benchmark: Tree reuse across turns via persistent node cache.

Simulates a multi-turn game, measuring how many internal nodes hit
the cache from previous turns. Unlike leaf-only caching, this caches
MCTS values at EVERY visited node (leaf + internal), so positions
that were internal nodes in one turn can be reused as root/internal
nodes in the next turn.

Usage:
    cd /Users/t/projects/slowpoke
    .venv/bin/python library/tests/bench_tree_reuse.py
"""

import sys
import os
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from library.core.checkers import CheckerBoard
from library.decision.tmcts import TMCTS


class MockEvaluator:
    """Mock evaluator with deterministic position->score mapping."""

    class MockNN:
        def __init__(self):
            self._use_mlx = True
            self.layer_size = [91]

        def compute_batch_mlx(self, positions):
            import numpy as np
            fake_scores = []
            for pos in positions:
                h = hash(pos.tobytes()) % 1000
                score = (h / 1000.0) * 2.0 - 1.0
                fake_scores.append(score)
            return np.array(fake_scores, dtype=np.float32)

        def subsquares(self, board_status):
            return board_status

    def __init__(self):
        self.nn = self.MockNN()

    def evaluate_board(self, B, colour):
        return 0.0


def simulate_game(num_moves=12, ply=4, base_round=30):
    B = CheckerBoard()
    colour = 0

    evaluator = MockEvaluator()
    tmcts = TMCTS(ply, evaluator, debug=False, batch_size=256)
    tmcts.baseRound = base_round
    tmcts.progressive_narrowing = False

    rounds_per_turn = base_round * ply

    print(f"\n{'='*75}")
    print(f"Tree Reuse Benchmark: ply={ply}, baseRound={base_round}, "
          f"rounds/turn={rounds_per_turn}")
    print(f"{'='*75}")

    total_time = 0.0
    total_nodes = 0
    total_hits = 0

    for turn in range(num_moves):
        start_time = time.time()
        hits_before = tmcts._cache_hits
        cache_before = len(tmcts._node_cache)

        move = tmcts.Decide(B, colour)

        elapsed = time.time() - start_time
        turn_hits = tmcts._cache_hits - hits_before
        cache_after = len(tmcts._node_cache)
        cache_added = cache_after - cache_before

        # Estimate total nodes visited this turn:
        # Each round visits ~ply internal nodes + 1 leaf = ply+1 nodes
        # But with caching, some nodes are pruned
        # Lower bound: at least 1 node per round (the root was always evaluated)
        # Upper bound: (ply+1) nodes per round
        total_nodes_this_turn = rounds_per_turn
        hit_rate = (turn_hits / total_nodes_this_turn * 100)

        total_hits += turn_hits
        total_nodes += total_nodes_this_turn
        total_time += elapsed

        print(f"  Turn {turn+1:2d}: move={str(move):>10s}  "
              f"cache={cache_before:>5d}->{cache_after:>5d}  "
              f"hits={turn_hits:>5d}/{total_nodes_this_turn:<4d} "
              f"({hit_rate:>5.1f}%)  "
              f"time={elapsed:.4f}s")

        # Apply agent move + random opponent response
        B.push_move(move)
        opponent_moves = B.get_moves()
        if opponent_moves:
            opp_move = random.choice(opponent_moves)
            B.push_move(opp_move)

    overall_hit_rate = (total_hits / total_nodes * 100) if total_nodes > 0 else 0.0
    print(f"\n{'─'*75}")
    print(f"Summary across {num_moves} turns:")
    print(f"  Total node visits: {total_nodes:,}")
    print(f"  Cache hits:        {total_hits:,}")
    print(f"  Overall hit rate:  {overall_hit_rate:.1f}%")
    print(f"  Final cache size:  {len(tmcts._node_cache):,} entries")
    print(f"  Total time:        {total_time:.4f}s")
    print(f"  Avg time/turn:     {total_time/num_moves:.4f}s")
    print(f"  Complexity saved:  {overall_hit_rate:.1f}% fewer ")
    print(f"    moves seen / branches traversed (unbalanced)")


def main():
    # Light simulation (fast)
    print("=== LIGHT: ply=4, baseRound=30 ===")
    simulate_game(num_moves=12, ply=4, base_round=30)

    # Tournament-style depth simulation
    print("\n\n=== DEEP: ply=8, baseRound=60 ===")
    simulate_game(num_moves=8, ply=8, base_round=60)


if __name__ == '__main__':
    main()
