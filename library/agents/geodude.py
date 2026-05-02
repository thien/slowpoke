"""Geodude — checkers AI that uses plain MCTS for decision making."""

from __future__ import annotations

import sys
from typing import Any

sys.path.insert(0, "..")
import decision.mcts as mcts
from agents.bot import Bot


class Geodude(Bot):
    """MCTS-based checkers bot with no neural network."""

    def __init__(self, plyDepth: int = 4) -> None:
        """Initialise Geodude agent.

        Args:
            plyDepth: Number of plies for MCTS search.
        """
        self.ply = plyDepth
        self.decisionFunction = mcts.MCTS(self.ply)

    def move_function(self, board: Any, colour: int) -> int:
        """Return a move using plain MCTS."""
        return self.decisionFunction.Decide(board, colour)
