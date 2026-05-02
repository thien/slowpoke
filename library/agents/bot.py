"""
Bot - Base class for all game AI bots.

All bots must implement move_function(board, colour).
Additional attributes (nn, cache, evaluate_board, etc.) are optional
and specific to the bot type.
"""

class Bot:
    """Base class for game AI bots.

    Every bot must implement move_function(board, colour), which returns
    a legal move from board.get_moves(). Subclasses may also define
    optional attributes like nn, cache, evaluate_board, or enableCache
    depending on their capabilities.
    """

    def move_function(self, board, colour):
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
