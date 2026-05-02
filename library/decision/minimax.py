"""Minimax decision module with alpha-beta pruning."""

from __future__ import annotations

from typing import Any, Callable, List

minimax_win = 1
minimax_lose = -minimax_win
minimax_draw = 0
minimax_empty = -1


class MiniMax:
    """Minimax search with alpha-beta pruning."""

    def __init__(self, ply: int, evaluator: Callable[[Any, int], float]) -> None:
        """Initialise MiniMax.

        Args:
            ply: Search depth.
            evaluator: Board evaluation function (board, colour) -> score.
        """
        self.ply = ply
        self.evaluator = evaluator
        self.counter: int = 0
        self.movesConsidered: List[int] = []

    def decide(self, B: Any, colour: int) -> int:
        """Return the best move found by minimax search."""
        return self.minimax(B, colour)

    def alpha_beta(
        self,
        B: Any,
        ply: int,
        alpha: float,
        beta: float,
        colour: int,
        maximizing: bool = True,
    ) -> float:
        """Alpha-beta search.

        Args:
            B: Board state.
            ply: Remaining search depth.
            alpha: Lower bound for maximizing player.
            beta: Upper bound for minimizing player.
            colour: Current player colour.
            maximizing: Whether this is a maximizing node.

        Returns:
            Evaluated score for this position.
        """
        self.counter += 1
        if B.is_over():
            if B.winner != minimax_empty:
                return minimax_lose if maximizing else minimax_win
            else:
                return minimax_draw

        moves = B.get_moves()
        for move in moves:
            B.push_move(move)
            if ply == 0:
                score = self.evaluator(B, colour)
            else:
                score = self.alpha_beta(
                    B, ply - 1, alpha, beta, B.current_player(), not maximizing
                )
            B.pop_move()

            if maximizing:
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    return alpha
            else:
                if score < beta:
                    beta = score
                if beta <= alpha:
                    return beta
        return alpha if maximizing else beta

    def minimax(self, B: Any, colour: int) -> int:
        """Run minimax search with alpha-beta pruning to find the best move.

        Args:
            B: Board state.
            colour: Current player colour.

        Returns:
            Best move as an integer.
        """
        self.counter = 0
        self.movesConsidered = []

        moves = B.get_moves()
        best_move = moves[0]
        best_score = float("-inf")

        alpha = float("-inf")
        beta = float("inf")

        if len(moves) == 1:
            return moves[0]

        for move in moves:
            B.push_move(move)
            if self.ply == 0:
                score = self.evaluator(B, colour)
            else:
                score = self.alpha_beta(
                    B, self.ply - 1, alpha, beta, B.current_player(), False
                )
            B.pop_move()
            if score > best_score:
                best_move = move
                best_score = score
        self.movesConsidered.append(self.counter)
        return best_move
