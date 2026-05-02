"""Geodude — checkers AI that uses plain MCTS for decision making."""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "..")
import decision.mcts as mcts
from agents.bot import Bot


class Geodude(Bot):
    """MCTS-based checkers bot with no neural network."""

    def __init__(self, ply_depth: int = 4) -> None:
        """Initialise Geodude agent.

        Args:
            ply_depth: Number of plies for MCTS search.
        """
        self.ply = ply_depth
        self.decision_function = mcts.MCTS(self.ply)

    def move_function(self, board: Any, colour: int) -> int:
        """Return a move using plain MCTS."""
        return self.decision_function.decide(board, colour)
