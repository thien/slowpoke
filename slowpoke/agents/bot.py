from __future__ import annotations

from typing import Any


class Bot:
    """Base class for game AI bots.

    Every bot must implement move_function(board, colour), which returns
    a legal move from board.get_moves(). Subclasses may also define
    optional attributes like nn, cache, evaluate_board, or enable_cache
    depending on their capabilities.
    """

    def move_function(self, board: Any, colour: int) -> int:
        """Return a legal move for the given board position and colour.

        Args:
            board: A CheckerBoard instance.
            colour: 0 (Black) or 1 (White).

        Returns:
            A move object from board.get_moves().
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement move_function(board, colour)"
        )
