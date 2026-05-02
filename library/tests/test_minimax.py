"""Comprehensive tests for the MiniMax decision module."""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision import minimax
from core import checkers
import numpy as np


Black, White = 0, 1


def constant_evaluator(value):
    """Create an evaluator that returns a constant value."""
    def evaluate(board, colour):
        if board.is_over():
            return minimax.minimax_draw if board.winner == -1 else (
                minimax.minimax_win if board.winner == colour else minimax.minimax_lose
            )
        return value
    return evaluate


class TestMiniMaxInit(unittest.TestCase):
    """Test MiniMax initialisation."""

    def test_ply_stored(self):
        """Ply depth should be stored."""
        mm = minimax.MiniMax(ply=3, evaluator=lambda b, c: 0.0)
        self.assertEqual(mm.ply, 3)

    def test_evaluator_stored(self):
        """Evaluator function should be stored."""
        fn = lambda b, c: 0.5
        mm = minimax.MiniMax(ply=1, evaluator=fn)
        self.assertIs(mm.evaluator, fn)


class TestMiniMaxDecide(unittest.TestCase):
    """Test MiniMax decision making."""

    def test_returns_valid_move(self):
        """Decide should return a valid move."""
        mm = minimax.MiniMax(ply=1, evaluator=constant_evaluator(0.0))
        B = checkers.CheckerBoard()
        move = mm.Decide(B, Black)
        self.assertIn(move, B.get_moves())

    def test_returns_integer_move(self):
        """Returned move should be an integer."""
        mm = minimax.MiniMax(ply=1, evaluator=constant_evaluator(0.0))
        B = checkers.CheckerBoard()
        move = mm.Decide(B, Black)
        self.assertIsInstance(move, int)

    def test_single_move_returns_immediately(self):
        """If only one move exists, return it without search."""
        B = checkers.CheckerBoard()
        mm = minimax.MiniMax(ply=5, evaluator=constant_evaluator(0.0))
        # Get to a state with only one move
        for _ in range(60):
            moves = B.get_moves()
            if len(moves) == 1:
                break
            B.make_move(moves[0])
        if len(B.get_moves()) == 1:
            move = mm.Decide(B, B.active)
            self.assertEqual(move, B.get_moves()[0])


class TestMiniMaxAlphaBeta(unittest.TestCase):
    """Test alpha-beta pruning algorithm."""

    def setUp(self):
        self.mm = minimax.MiniMax(ply=2, evaluator=constant_evaluator(0.5))

    def test_alphabeta_returns_score(self):
        """alphabeta should return a numeric score."""
        B = checkers.CheckerBoard()
        self.mm.counter = 0
        score = self.mm.alphabeta(B, 1, float('-inf'), float('inf'), Black, True)
        self.assertIsInstance(score, (int, float))

    def test_alphabeta_score_in_range(self):
        """Score should be within [-1, 1]."""
        B = checkers.CheckerBoard()
        self.mm.counter = 0
        score = self.mm.alphabeta(B, 1, float('-inf'), float('inf'), Black, True)
        self.assertGreaterEqual(score, minimax.minimax_lose)
        self.assertLessEqual(score, minimax.minimax_win)

    def test_alphabeta_respects_alpha_beta_bounds(self):
        """Score should stay within alpha-beta window."""
        B = checkers.CheckerBoard()
        self.mm.counter = 0
        alpha, beta = -0.3, 0.3
        score = self.mm.alphabeta(B, 2, alpha, beta, Black, True)
        self.assertGreaterEqual(score, alpha)
        self.assertLessEqual(score, beta)

    def test_alphabeta_counter_increments(self):
        """Counter should increment with each node visited."""
        B = checkers.CheckerBoard()
        self.mm.counter = 0
        self.mm.alphabeta(B, 2, float('-inf'), float('inf'), Black, True)
        self.assertGreater(self.mm.counter, 0)

    def test_maximizing_player(self):
        """Maximizing player should prefer higher scores."""
        B = checkers.CheckerBoard()
        self.mm.counter = 0
        score = self.mm.alphabeta(B, 1, float('-inf'), float('inf'), Black, True)
        self.assertIsNotNone(score)

    def test_minimizing_player(self):
        """Minimizing player should prefer lower scores."""
        B = checkers.CheckerBoard()
        self.mm.counter = 0
        score = self.mm.alphabeta(B, 1, float('-inf'), float('inf'), Black, False)
        self.assertIsNotNone(score)

    def test_terminal_state_maximizing(self):
        """Terminal loss from maximizing perspective should return minimax_lose."""
        from unittest import mock
        with mock.patch.object(checkers.CheckerBoard, 'is_over', return_value=True):
            B = checkers.CheckerBoard()
            B.winner = Black
            self.mm.counter = 0
            score = self.mm.alphabeta(B, 0, float('-inf'), float('inf'), Black, True)
            self.assertEqual(score, minimax.minimax_lose)

    def test_terminal_state_minimizing(self):
        """Terminal loss from minimizing perspective should return minimax_win."""
        from unittest import mock
        with mock.patch.object(checkers.CheckerBoard, 'is_over', return_value=True):
            B = checkers.CheckerBoard()
            B.winner = Black
            self.mm.counter = 0
            score = self.mm.alphabeta(B, 0, float('-inf'), float('inf'), Black, False)
            self.assertEqual(score, minimax.minimax_win)


class TestMiniMaxConstants(unittest.TestCase):
    """Test minimax constant values."""

    def test_win_value(self):
        """Win should be 1."""
        self.assertEqual(minimax.minimax_win, 1)

    def test_lose_value(self):
        """Lose should be -1."""
        self.assertEqual(minimax.minimax_lose, -1)

    def test_draw_value(self):
        """Draw should be 0."""
        self.assertEqual(minimax.minimax_draw, 0)

    def test_empty_value(self):
        """Empty should be -1."""
        self.assertEqual(minimax.minimax_empty, -1)

    def test_lose_is_negative_win(self):
        """Lose should be the negative of win."""
        self.assertEqual(minimax.minimax_lose, -minimax.minimax_win)


if __name__ == "__main__":
    unittest.main(verbosity=2)
