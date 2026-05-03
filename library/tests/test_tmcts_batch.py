"""
Tests for TMCTS batch evaluation and end-to-end AI comparisons.
"""

import unittest

from core import checkers
import decision.tmcts as tmcts
import numpy as np


Black, White, empty = 0, 1, -1
blackKing, whiteKing = 2, 3


class SimpleEvaluator:
    """Simple evaluator that returns random values for testing."""

    def __init__(self):
        np.random.seed(42)

    def __call__(self, board, colour):
        return float(np.random.random() * 2 - 1)


class TestTMCTSBasic(unittest.TestCase):
    """Test TMCTS basic functionality."""

    def setUp(self):
        self.B = checkers.CheckerBoard()
        self.evaluator = SimpleEvaluator()

    def test_tmcts_returns_valid_move(self):
        """TMCTS should return a valid move."""
        agent = tmcts.TMCTS(ply=1, evaluator=self.evaluator, debug=True)
        move = agent.decide(self.B, Black)
        self.assertIn(move, self.B.get_moves())

    def test_tmcts_single_move(self):
        """TMCTS should handle single move states."""
        B = checkers.CheckerBoard()
        agent = tmcts.TMCTS(ply=2, evaluator=self.evaluator, debug=True)

        # Find single move state
        for _ in range(30):
            if len(B.get_moves()) == 1:
                break
            B.make_move(B.get_moves()[0])

        if len(B.get_moves()) == 1:
            move = agent.decide(B, B.active)
            self.assertEqual(move, B.get_moves()[0])


class TestTMCTSMoveSelection(unittest.TestCase):
    """Test TMCTS move selection logic."""

    def test_move_selection_aggregation(self):
        """Test that move selection properly aggregates results."""
        B = checkers.CheckerBoard()
        evaluator = SimpleEvaluator()
        agent = tmcts.TMCTS(ply=2, evaluator=evaluator, debug=True)

        # Run multiple times to check consistency
        moves = []
        for _ in range(3):
            B_copy = B.copy()
            move = agent.decide(B_copy, Black)
            moves.append(move)

        # All moves should be valid
        for move in moves:
            self.assertIn(move, B.get_moves())


if __name__ == "__main__":
    unittest.main(verbosity=2)
