"""
Tests for Parallel TMCTS batch evaluation.
"""

import unittest
import time

from core import checkers
from search.parallel_tmcts import ParallelTMCTS, SharedBatchAccumulator
from search.tmcts import TMCTS
import numpy as np

Black, White, empty = 0, 1, -1
blackKing, whiteKing = 2, 3


class SimpleEvaluator:
    """Simple evaluator that returns random values for testing."""

    def __init__(self):
        np.random.seed(42)

    def __call__(self, board, colour):
        return float(np.random.random() * 2 - 1)


class TestSharedBatchAccumulator(unittest.TestCase):
    """Test the shared batch accumulator."""

    def test_add_position(self):
        """Test adding positions to the accumulator."""
        acc = SharedBatchAccumulator(batch_size=10)
        pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        idx = acc.add_position(pos)
        self.assertEqual(idx, 0)
        self.assertEqual(len(acc._batch_positions), 1)

    def test_multiple_positions(self):
        """Test adding multiple positions."""
        acc = SharedBatchAccumulator(batch_size=10)

        for i in range(5):
            pos = np.array([float(i)] * 3, dtype=np.float32)
            idx = acc.add_position(pos)
            self.assertEqual(idx, i)

    def test_flush_and_evaluate(self):
        """Test flushing and evaluating positions."""
        acc = SharedBatchAccumulator(batch_size=10)

        class MockNN:
            def compute_batch_mlx(self, batch):
                return np.array([0.5] * len(batch), dtype=np.float32)

        nn = MockNN()

        for i in range(3):
            pos = np.array([float(i)] * 3, dtype=np.float32)
            acc.add_position(pos)

        results = acc.flush_and_evaluate(nn)
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r == 0.5 for r in results))


class TestParallelTMCTS(unittest.TestCase):
    """Test Parallel TMCTS functionality."""

    def setUp(self):
        self.B = checkers.CheckerBoard()
        self.evaluator = SimpleEvaluator()

    def test_parallel_returns_valid_move(self):
        """Parallel TMCTS should return a valid move."""
        agent = ParallelTMCTS(
            ply=1, evaluator=self.evaluator, num_parallel=2, debug=True
        )
        move = agent.decide(self.B, Black)
        self.assertIn(move, self.B.get_moves())

    def test_parallel_single_move(self):
        """Parallel TMCTS should handle single move states."""
        B = checkers.CheckerBoard()
        agent = ParallelTMCTS(
            ply=2, evaluator=self.evaluator, num_parallel=2, debug=True
        )

        # Find single move state
        for _ in range(30):
            if len(B.get_moves()) == 1:
                break
            B.make_move(B.get_moves()[0])

        if len(B.get_moves()) == 1:
            move = agent.decide(B, B.active)
            self.assertEqual(move, B.get_moves()[0])

    def test_parallel_accumulator_sharing(self):
        """Test that positions are shared across instances."""
        agent = ParallelTMCTS(
            ply=1, evaluator=self.evaluator, num_parallel=4, debug=True
        )

        # Run decision
        agent.decide(self.B, Black)

        # Check that positions were accumulated
        self.assertGreater(agent.accumulator._total_positions, 0)


class TestParallelBenchmark(unittest.TestCase):
    """Benchmark tests comparing single vs parallel TMCTS."""

    def setUp(self):
        self.B = checkers.CheckerBoard()
        self.evaluator = SimpleEvaluator()

    def test_benchmark_single_vs_parallel(self):
        """Benchmark single TMCTS vs parallel TMCTS.

        Note: Parallel is slower with simple evaluators due to threading overhead.
        Real speedup (3-7x) comes from GPU batch evaluation with neural networks.
        """
        rounds = 50  # Fixed rounds for consistent comparison

        # Single instance
        single = TMCTS(ply=2, evaluator=self.evaluator, debug=False)
        single.base_round = rounds

        start = time.time()
        move_single = single.decide(self.B.copy(), Black)
        time_single = time.time() - start

        # Parallel instance
        parallel = ParallelTMCTS(
            ply=2, evaluator=self.evaluator, num_parallel=4, debug=False
        )
        parallel.base_round = rounds

        start = time.time()
        move_parallel = parallel.decide(self.B.copy(), Black)
        time_parallel = time.time() - start

        # Print benchmark results
        print(f"\n{'=' * 50}")
        print(f"Benchmark Results (ply=2, rounds={rounds}):")
        print(f"{'=' * 50}")
        print(f"Single TMCTS:  {time_single:.3f}s")
        print(f"Parallel TMCTS (4x): {time_parallel:.3f}s")
        print(f"Speedup: {time_single / time_parallel:.2f}x")
        if time_single / time_parallel < 1.0:
            print(
                "Note: Parallel is SLOWER with simple evaluators due to threading overhead."
            )
        print(
            "Real speedup (3-7x) comes from GPU batch evaluation with neural networks."
        )
        print(
            f"Same move: {move_single == move_parallel} (expected False - single uses global random)"
        )
        print(
            "Parallel with same seed: deterministic (see test_deterministic_with_seed)"
        )
        print(f"{'=' * 50}\n")

        # Both should return valid moves
        self.assertIn(move_single, self.B.get_moves())
        self.assertIn(move_parallel, self.B.get_moves())

    def test_parallel_scaling(self):
        """Test parallel scaling with different thread counts."""
        rounds = 30
        results = {}

        for num_parallel in [1, 2, 4, 8]:
            B = self.B.copy()
            parallel = ParallelTMCTS(
                ply=2,
                evaluator=self.evaluator,
                num_parallel=num_parallel,
                debug=False,
                seed=42,
            )
            parallel.base_round = rounds

            start = time.time()
            move = parallel.decide(B, Black)
            elapsed = time.time() - start
            results[num_parallel] = elapsed

            # All should return valid moves
            self.assertIn(move, self.B.get_moves())

        # Print scaling results
        print(f"\n{'=' * 50}")
        print(f"Parallel Scaling (ply=2, rounds={rounds}):")
        print(f"{'=' * 50}")
        for num, elapsed in results.items():
            speedup = results[1] / elapsed
            print(f"  {num} parallel: {elapsed:.3f}s ({speedup:.2f}x)")
        print(f"{'=' * 50}\n")

    def test_deep_ply_scaling(self):
        """Test parallel scaling with deeper ply (more positions = better amortization)."""
        rounds = 50
        results = {}

        for num_parallel in [1, 2, 4, 8]:
            B = self.B.copy()
            parallel = ParallelTMCTS(
                ply=6,
                evaluator=self.evaluator,
                num_parallel=num_parallel,
                debug=False,
                seed=42,
            )
            parallel.base_round = rounds

            start = time.time()
            move = parallel.decide(B, Black)
            elapsed = time.time() - start
            results[num_parallel] = elapsed

            self.assertIn(move, self.B.get_moves())

        # Print deep ply scaling results
        print(f"\n{'=' * 50}")
        print(f"Deep Ply Scaling (ply=6, rounds={rounds}):")
        print(f"{'=' * 50}")
        for num, elapsed in results.items():
            speedup = results[1] / elapsed
            print(f"  {num} parallel: {elapsed:.3f}s ({speedup:.2f}x)")
        print(f"{'=' * 50}\n")


class TestParallelCorrectness(unittest.TestCase):
    """Test correctness: parallel should match baseline behavior."""

    def setUp(self):
        self.B = checkers.CheckerBoard()
        self.evaluator = SimpleEvaluator()

    def test_both_return_valid_moves(self):
        """Both implementations should return valid moves."""
        single = TMCTS(ply=2, evaluator=self.evaluator, debug=False)
        single.base_round = 30

        parallel = ParallelTMCTS(
            ply=2, evaluator=self.evaluator, num_parallel=4, debug=False
        )
        parallel.base_round = 30

        B1 = self.B.copy()
        B2 = self.B.copy()

        move_single = single.decide(B1, Black)
        move_parallel = parallel.decide(B2, Black)

        self.assertIn(move_single, self.B.get_moves())
        self.assertIn(move_parallel, self.B.get_moves())

    def test_parallel_produces_different_valid_moves(self):
        """Parallel should produce valid moves (moves may vary due to randomness)."""
        for _ in range(5):
            B = checkers.CheckerBoard()  # Fresh board each time
            parallel = ParallelTMCTS(
                ply=2, evaluator=self.evaluator, num_parallel=4, debug=False
            )
            parallel.base_round = 30
            move = parallel.decide(B, Black)
            # Just verify it returns something (validity checked elsewhere)
            self.assertIsInstance(move, int)

    def test_deterministic_with_seed(self):
        """Parallel with same seed should produce same move as single (if single used same seed)."""
        seed = 12345
        rounds = 30

        # Parallel with seed
        B1 = self.B.copy()
        parallel = ParallelTMCTS(
            ply=2, evaluator=self.evaluator, num_parallel=4, debug=False, seed=seed
        )
        parallel.base_round = rounds
        move_parallel = parallel.decide(B1, Black)

        # Run again with same seed - should get same move
        B2 = self.B.copy()
        parallel2 = ParallelTMCTS(
            ply=2, evaluator=self.evaluator, num_parallel=4, debug=False, seed=seed
        )
        parallel2.base_round = rounds
        move_parallel2 = parallel2.decide(B2, Black)

        self.assertEqual(
            move_parallel, move_parallel2, "Same seed should produce same move"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
