"""Splash — random move selection agent."""

from __future__ import annotations

import random
from typing import Any


class Splash:
    """Agent that makes a random legal move."""

    def __init__(self) -> None:
        pass

    def decide(self, B: Any) -> int:
        """Return a random legal move.

        Note: the second overload (decide(self, B, colour)) is intentionally
        removed — the single-argument variant is the canonical interface.
        """
        return random.choice(B.get_moves())
