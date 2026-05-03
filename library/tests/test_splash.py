"""Tests for the Splash (random move) decision module."""

import unittest

from decision.splash import Splash
from core import checkers


Black, White = 0, 1


class TestSplashInit(unittest.TestCase):
    """Test Splash initialisation."""

    def test_init_does_not_crash(self):
        """Init should work without arguments."""
        s = Splash()
        self.assertIsNotNone(s)


class TestSplashdecide(unittest.TestCase):
    """Test Splash decision making.

    Note: Splash defines decide(self, B) which overrides decide(self, B, colour).
    The only working signature is decide(board) without colour.
    """

    def setUp(self):
        self.splash = Splash()
        self.B = checkers.CheckerBoard()

    def test_decide_returns_valid_move(self):
        """decide(board) should return a valid move."""
        move = self.splash.decide(self.B)
        self.assertIn(move, self.B.get_moves())

    def test_decide_returns_integer(self):
        """Returned move should be an integer."""
        move = self.splash.decide(self.B)
        self.assertIsInstance(move, int)

    def test_next_move_different(self):
        """Consecutive calls should sometimes return different moves."""
        moves = set()
        for _ in range(20):
            move = self.splash.decide(self.B)
            moves.add(move)
        # With 7 legal moves, 20 random choices should produce at least 3 unique
        self.assertGreaterEqual(
            len(moves), 3, "Splash should produce diverse random moves"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
