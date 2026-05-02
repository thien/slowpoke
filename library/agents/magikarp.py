"""Magikarp is a draughts AI that plays completely randomly."""

from __future__ import annotations

import random
from typing import Any, Optional


class Magikarp:

  def __init__(self) -> None:
    self.null: Optional[Any] = None

  def move_function(self, B: Any, colour: Optional[int] = None) -> int:
    """Return a random legal move."""
    return random.choice(B.get_moves())
