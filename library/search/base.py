"""MCTSBase — shared helpers for MCTS and TMCTS."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import numpy as np

from core.constants import (
    MLX_AVAILABLE,
    mx,
    MINIMAX_WIN,
    MINIMAX_LOSE,
    MINIMAX_DRAW,
    MINIMAX_EMPTY,
)


class MCTSBase:
    """Base class with shared helpers for MCTS variants.

    Subclasses must implement ``decide()``.
    Provides ``is_over()``, ``_extract_position()``, ``_detect_mlx()``,
    and ``_ucb1_score()``.
    """

    def __init__(
        self,
        ply: int,
        evaluator: Optional[Any] = None,
        debug: bool = False,
        batch_size: int = 512,
    ) -> None:
        self.ply = ply
        self.evaluator = evaluator
        self.debug = debug
        self.batch_size = batch_size
        self._detect_mlx()

    def _detect_mlx(self) -> None:
        """Detect MLX availability from the evaluator's NN."""
        self.nn = getattr(self.evaluator, "nn", None) if self.evaluator else None
        self.use_mlx = (
            (self.nn is not None and getattr(self.nn, "_use_mlx", False))
            if self.evaluator
            else False
        )

    def is_over(self, B: Any, colour: int) -> Tuple[bool, int]:
        """Check if the board is in a terminal state from colour's perspective.

        Returns:
            (True, +1) if colour wins, (True, -1) if colour loses,
            (True, 0) if draw, (False, -1) if game continues.
        """
        if B.is_over(check_repetition=False):
            if B.winner != MINIMAX_EMPTY:
                if B.winner == colour:
                    return (True, MINIMAX_WIN)
                else:
                    return (True, MINIMAX_LOSE)
            else:
                return (True, MINIMAX_DRAW)
        return (False, -1)

    def _extract_position(self, B: Any, colour: int) -> np.ndarray:
        """Extract board position as a float32 array for NN evaluation."""
        board_status = B.get_board_pos_weighted(
            colour,
            {
                "Black": 1,
                "White": -1,
                "empty": 0,
                "blackKing": 1.5,
                "whiteKing": -1.5,
            },
        )
        if (
            self.nn is not None
            and hasattr(self.nn, "layer_size")
            and self.nn.layer_size[0] == 91
        ):
            board_status = self.nn.subsquares(board_status)
        return np.asarray(board_status, dtype=np.float32)

    @staticmethod
    def _ucb1_score(
        wins: float, visits: int, total_visits: int, c: float = 1.4
    ) -> float:
        """Compute UCB1 score for a move.

        Args:
            wins: Number of wins from this move.
            visits: Number of times this move was tried.
            total_visits: Total visits across all moves at this level.
            c: Exploration constant (default 1.4).

        Returns:
            UCB1 score. Infinite for unvisited moves.
        """
        if visits == 0:
            return float("inf")
        return wins / visits + c * math.sqrt(math.log(total_visits + 1) / visits)
