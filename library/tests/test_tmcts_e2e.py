"""
End-to-end tests comparing Random vs Greedy vs MCTS AI strategies.
"""

import pytest
import unittest

from core import checkers
import search.tmcts as tmcts
import search.mcts as mcts
import numpy as np

Black, White, empty = 0, 1, -1


class ConstantEvaluator:
    """Evaluator that returns a constant value."""

    def __init__(self, value=0.0):
        self.value = value

    def __call__(self, board, colour):
        return self.value

    def evaluate_board(self, board, colour):
        return self.value


class TestTMCTSBasicFunctionality(unittest.TestCase):
    """Test TMCTS basic functionality."""

    def test_tmcts_returns_valid_move(self):
        """TMCTS should return a valid move."""
        B = checkers.CheckerBoard()
        evaluator = ConstantEvaluator(0.0)
        agent = tmcts.TMCTS(ply=1, evaluator=evaluator, debug=True)
        move = agent.decide(B, Black)
        self.assertIn(move, B.get_moves())

    def test_tmcts_single_move(self):
        """TMCTS should handle single move states."""
        B = checkers.CheckerBoard()
        evaluator = ConstantEvaluator(0.0)
        agent = tmcts.TMCTS(ply=2, evaluator=evaluator, debug=True)

        # Find single move state
        for _ in range(10):
            if len(B.get_moves()) == 1:
                break
            B.make_move(B.get_moves()[0])

        if len(B.get_moves()) == 1:
            move = agent.decide(B, B.active)
            self.assertEqual(move, B.get_moves()[0])


class TestTMCTSBatchEvaluation(unittest.TestCase):
    """Test TMCTS batch evaluation functionality."""

    def test_position_extraction(self):
        """Test that position extraction works correctly."""
        B = checkers.CheckerBoard()
        evaluator = ConstantEvaluator(0.0)
        agent = tmcts.TMCTS(ply=1, evaluator=evaluator, debug=True)

        pos = agent._extract_position(B, Black)

        self.assertIsInstance(pos, np.ndarray)
        self.assertEqual(len(pos), 32)

    def test_flush_batch_handles_empty(self):
        """flush_batch should handle empty batch gracefully."""
        evaluator = ConstantEvaluator(0.0)
        agent = tmcts.TMCTS(ply=1, evaluator=evaluator, debug=True)

        result = agent.flush_batch()
        self.assertIsInstance(result, np.ndarray)

    def test_statistics_initialization(self):
        """Test that statistics are initialized correctly."""
        B = checkers.CheckerBoard()
        evaluator = ConstantEvaluator(0.0)
        agent = tmcts.TMCTS(ply=1, evaluator=evaluator, debug=True)

        agent.decide(B, Black)

        self.assertIsInstance(agent.movesets, dict)


@pytest.mark.slow
class TestMCTSBasicFunctionality(unittest.TestCase):
    """Test MCTS basic functionality."""

    def test_mcts_returns_valid_move(self):
        """MCTS should return a valid move."""
        B = checkers.CheckerBoard()
        agent = mcts.MCTS(ply=1, evaluator=ConstantEvaluator(0.0))
        move = agent.decide(B, Black)
        self.assertIn(move, B.get_moves())


class TestTMCTSWithMLX(unittest.TestCase):
    """Test TMCTS with MLX evaluator (if available)."""

    def test_mlx_evaluator_available(self):
        """Test if MLX evaluator is available."""
        try:
            from agents.slowpoke import Slowpoke

            agent = Slowpoke(ply_depth=1, use_mlx=True, debug=True)
            B = checkers.CheckerBoard()
            mcts_agent = tmcts.TMCTS(ply=1, evaluator=agent, debug=True)

            move = mcts_agent.decide(B, Black)
            self.assertIn(move, B.get_moves())
        except ImportError:
            self.skipTest("MLX not available or Slowpoke not importable")


if __name__ == "__main__":
    unittest.main(verbosity=2)
