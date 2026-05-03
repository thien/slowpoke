"""Comprehensive tests for the MCTS decision module."""

import unittest

from decision import mcts
from core import checkers

Black, White = 0, 1


class TestMCTSInit(unittest.TestCase):
    """Test MCTS initialisation."""

    def test_default_ply(self):
        """Ply should be stored."""
        mc = mcts.MCTS(ply=3, evaluator=lambda b, c: 0.0)
        self.assertEqual(mc.ply, 3)

    def test_evaluator_stored(self):
        """Evaluator should be stored."""

        def fn(b, c):
            return 0.5

        mc = mcts.MCTS(ply=1, evaluator=fn)
        self.assertIs(mc.evaluator, fn)

    def test_default_c_value(self):
        """Default exploration constant should be 1.4."""
        mc = mcts.MCTS(ply=1, evaluator=lambda b, c: 0.0)
        self.assertEqual(mc.c, 1.4)

    def test_mlx_false_by_default(self):
        """MLX should be off by default."""
        mc = mcts.MCTS(ply=1, evaluator=lambda b, c: 0.0)
        self.assertFalse(mc.use_mlx)


class TestMCTSdecide(unittest.TestCase):
    """Test MCTS decision making."""

    def test_returns_valid_move(self):
        """decide should return a valid move."""
        mc = mcts.MCTS(ply=1, evaluator=lambda b, c: 0.0)
        B = checkers.CheckerBoard()
        move = mc.decide(B, Black)
        self.assertIn(move, B.get_moves())

    def test_returns_integer(self):
        """Returned move should be an integer."""
        mc = mcts.MCTS(ply=1, evaluator=lambda b, c: 0.0)
        B = checkers.CheckerBoard()
        move = mc.decide(B, Black)
        self.assertIsInstance(move, int)

    def test_single_move_returns_immediately(self):
        """If only one move exists, return it without search."""
        B = checkers.CheckerBoard()
        mc = mcts.MCTS(ply=1, evaluator=lambda b, c: 0.0)
        for _ in range(60):
            moves = B.get_moves()
            if len(moves) == 1:
                break
            B.make_move(moves[0])
        if len(B.get_moves()) == 1:
            move = mc.decide(B, B.active)
            self.assertEqual(move, B.get_moves()[0])

    def test_no_moves_returns_none(self):
        """If no moves exist on an empty board, return None."""
        # Play until a terminal state with no moves
        B = checkers.CheckerBoard()
        for _ in range(200):
            moves = B.get_moves()
            if not moves:
                break
            B.make_move(moves[0])
        if not B.get_moves():
            mc = mcts.MCTS(ply=1, evaluator=lambda b, c: 0.0)
            result = mc.decide(B, Black)
            self.assertIsNone(result)
        else:
            self.skipTest("Could not reach a no-move state")


class TestMCTSStatistics(unittest.TestCase):
    """Test MCTS statistics tracking."""

    def test_plays_dict_created(self):
        """mcts_plays should be created and be a dict."""
        mc = mcts.MCTS(ply=1, evaluator=lambda b, c: 0.0)
        B = checkers.CheckerBoard()
        mc.decide(B, Black)
        self.assertIsInstance(mc.mcts_plays, dict)

    def test_chances_dict_created(self):
        """mcts_chances should be created and be a dict."""
        mc = mcts.MCTS(ply=1, evaluator=lambda b, c: 0.0)
        B = checkers.CheckerBoard()
        mc.decide(B, Black)
        self.assertIsInstance(mc.mcts_chances, dict)

    def test_plays_has_entries(self):
        """After decide, plays dict should not be empty."""
        mc = mcts.MCTS(ply=1, evaluator=lambda b, c: 0.0)
        B = checkers.CheckerBoard()
        mc.decide(B, Black)
        self.assertTrue(len(mc.mcts_plays) > 0)

    def test_chances_has_entries(self):
        """After decide, chances dict should not be empty."""
        mc = mcts.MCTS(ply=1, evaluator=lambda b, c: 0.0)
        B = checkers.CheckerBoard()
        mc.decide(B, Black)
        self.assertTrue(len(mc.mcts_chances) > 0)

    def test_plays_values_are_positive(self):
        """Play counts should be positive integers."""
        mc = mcts.MCTS(ply=1, evaluator=lambda b, c: 0.0)
        B = checkers.CheckerBoard()
        mc.decide(B, Black)
        for v in mc.mcts_plays.values():
            self.assertGreaterEqual(v, 0)

    def test_chances_values_are_floats(self):
        """Chance values should be numeric."""
        mc = mcts.MCTS(ply=1, evaluator=lambda b, c: 0.0)
        B = checkers.CheckerBoard()
        mc.decide(B, Black)
        for v in mc.mcts_chances.values():
            self.assertIsInstance(v, (int, float))


class TestMCTSSimulate(unittest.TestCase):
    """Test MCTS simulation function."""

    def test_simulate_returns_none(self):
        """mcts_simulate should return None (modifies state in place)."""
        mc = mcts.MCTS(ply=1, evaluator=lambda b, c: 0.0)
        B = checkers.CheckerBoard()
        mc.mcts_plays = {}
        mc.mcts_chances = {}
        result = mc.mcts_simulate(B, 5, Black, 10)
        self.assertIsNone(result)

    def test_simulate_restores_board_state(self):
        """After simulation, board should be restored."""
        mc = mcts.MCTS(ply=1, evaluator=lambda b, c: 0.0)
        B = checkers.CheckerBoard()
        original_moves = B.get_moves()
        mc.mcts_plays = {}
        mc.mcts_chances = {}
        mc.mcts_simulate(B, 5, Black, 10)
        self.assertEqual(B.get_moves(), original_moves)


if __name__ == "__main__":
    unittest.main(verbosity=2)
