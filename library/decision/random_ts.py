"""RandomTS — random tree search for checkers."""

from __future__ import annotations

import random
from typing import Any

from core.constants import MINIMAX_WIN as minimax_win, MINIMAX_LOSE as minimax_lose, MINIMAX_DRAW as minimax_draw, MINIMAX_EMPTY as minimax_empty


class RandomTS:
    """Random tree search using random move selection with push/pop."""

    def __init__(self, ply: int, evaluator: Any) -> None:
        """Initialise RandomTS.

        Args:
            ply: Search depth.
            evaluator: Object with evaluate_board(board, colour) method.
        """
        self.ply = ply
        self.evaluator = evaluator

    def decide(self, B: Any, colour: int) -> int:
        """Return the best move found by random tree search."""
        return self._random_ts(B, self.ply, colour)

    def _random_ts(self, B: Any, ply: int, colour: int) -> int:
        """Run random tree search for a number of rounds.

        Args:
            B: Board state.
            ply: Search depth remaining.
            colour: Current player colour.

        Returns:
            Best move found.
        """
        moves = B.get_moves()
        best_move = moves[0]
        best_score = float("-inf")

        if len(moves) == 1:
            return moves[0]

        random_rounds = 1
        if self.ply > 0:
            random_rounds = 300 * self.ply

        for _ in range(random_rounds):
            random_move = random.choice(moves)
            B.push_move(random_move)
            score = self._tree_search(B, ply - 1, colour)
            B.pop_move()
            if score > best_score:
                best_score = score
                if best_move != random_move:
                    best_move = random_move
        return best_move

    def _tree_search(self, B: Any, ply: int, colour: int) -> float:
        """Single random playout from the current position.

        Args:
            B: Board state.
            ply: Remaining depth.
            colour: Current player colour.

        Returns:
            Score from colour's perspective.
        """
        if B.is_over():
            if B.winner != minimax_empty:
                return minimax_win if B.winner == colour else minimax_lose
            return minimax_draw

        moves = B.get_moves()
        move = random.choice(moves)
        B.push_move(move)
        if ply < 1:
            score = self.evaluator.evaluate_board(B, colour)
        else:
            score = self._tree_search(B, ply - 1, colour)
        B.pop_move()
        return score
