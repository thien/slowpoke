"""Tests for the RandomTS decision module."""

import unittest

from decision import random_ts
from core import checkers

Black, White = 0, 1


class MockEvaluator:
    """Simple mock evaluator for testing."""

    def evaluate_board(self, board, colour):
        return 0.0


class TestRandomTSInit(unittest.TestCase):
    """Test RandomTS initialisation."""

    def test_ply_stored(self):
        """Ply depth should be stored."""
        rts = random_ts.RandomTS(ply=3, evaluator=MockEvaluator())
        self.assertEqual(rts.ply, 3)

    def test_evaluator_stored(self):
        """Evaluator should be stored."""
        ev = MockEvaluator()
        rts = random_ts.RandomTS(ply=1, evaluator=ev)
        self.assertIs(rts.evaluator, ev)


class TestRandomTSdecide(unittest.TestCase):
    """Test RandomTS decision making."""

    def test_returns_valid_move(self):
        """decide should return a legal move."""
        rts = random_ts.RandomTS(ply=1, evaluator=MockEvaluator())
        B = checkers.CheckerBoard()
        move = rts.decide(B, Black)
        self.assertIn(move, B.get_moves())

    def test_returns_integer(self):
        """Returned move should be an integer."""
        rts = random_ts.RandomTS(ply=1, evaluator=MockEvaluator())
        B = checkers.CheckerBoard()
        move = rts.decide(B, Black)
        self.assertIsInstance(move, int)

    def test_single_move_returns_immediately(self):
        """If only one move exists, return it."""
        B = checkers.CheckerBoard()
        rts = random_ts.RandomTS(ply=1, evaluator=MockEvaluator())
        for _ in range(60):
            moves = B.get_moves()
            if len(moves) == 1:
                break
            B.make_move(moves[0])
        if len(B.get_moves()) == 1:
            move = rts.decide(B, B.active)
            self.assertEqual(move, B.get_moves()[0])


class TestRandomTSTreeSearch(unittest.TestCase):
    """Test RandomTS internal tree search."""

    def test_tree_search_returns_score(self):
        """_tree_search should return a numeric score."""
        rts = random_ts.RandomTS(ply=1, evaluator=MockEvaluator())
        B = checkers.CheckerBoard()
        score = rts._tree_search(B, 0, Black)
        self.assertIsInstance(score, (int, float))

    def test_tree_search_score_range(self):
        """Score should be in [-1, 1]."""
        rts = random_ts.RandomTS(ply=1, evaluator=MockEvaluator())
        B = checkers.CheckerBoard()
        score = rts._tree_search(B, 0, Black)
        self.assertGreaterEqual(score, -1)
        self.assertLessEqual(score, 1)

    def test_tree_search_restores_board(self):
        """Board state should be restored after search."""
        rts = random_ts.RandomTS(ply=1, evaluator=MockEvaluator())
        B = checkers.CheckerBoard()
        original_moves = B.get_moves()
        rts._tree_search(B, 0, Black)
        self.assertEqual(B.get_moves(), original_moves)

    def test_tree_search_win_at_terminal(self):
        """Terminal winning position should return minimax_win."""
        from unittest import mock
        from search.minimax import minimax_win

        with mock.patch.object(checkers.CheckerBoard, "is_over", return_value=True):
            rts = random_ts.RandomTS(ply=0, evaluator=MockEvaluator())
            B = checkers.CheckerBoard()
            B.winner = Black
            score = rts._tree_search(B, 0, Black)
            self.assertEqual(score, minimax_win)

    def test_tree_search_loss_at_terminal(self):
        """Terminal losing position should return minimax_lose."""
        from unittest import mock
        from search.minimax import minimax_lose

        with mock.patch.object(checkers.CheckerBoard, "is_over", return_value=True):
            rts = random_ts.RandomTS(ply=0, evaluator=MockEvaluator())
            B = checkers.CheckerBoard()
            B.winner = White
            score = rts._tree_search(B, 0, Black)
            self.assertEqual(score, minimax_lose)

    def test_tree_search_draw_at_terminal(self):
        """Terminal draw should return minimax_draw."""
        from unittest import mock
        from search.minimax import minimax_draw

        with mock.patch.object(checkers.CheckerBoard, "is_over", return_value=True):
            rts = random_ts.RandomTS(ply=0, evaluator=MockEvaluator())
            B = checkers.CheckerBoard()
            B.winner = -1
            score = rts._tree_search(B, 0, Black)
            self.assertEqual(score, minimax_draw)


if __name__ == "__main__":
    unittest.main(verbosity=2)
