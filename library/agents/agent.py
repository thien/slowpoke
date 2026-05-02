"""
Agent

This represents a computer player;
It'll contain information about its ranking,
and its move function.

It also has a default ELO.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional

from core import checkers


class Agent:
  def __init__(self, bot: Any, agent_id: Optional[str] = None, initial_elo: Optional[float] = None) -> None:
    self.bot = bot
    self.elo = initial_elo if initial_elo is not None else 1200  # Will be updated by Population.allocatePoints()
    self.points = 0
    self.champRange = 0
    self.champScore = 0
    self.move_function = bot.move_function
    self.colour = None
    self.games_played = 0  # Track games played for K-factor calibration
    
    # check for ID
    if agent_id is not None:
      self.setID(agent_id)
    else:
      self.genID()

    # genomic properties.
    self.origin = []
    # store parent's ID
    self.parents = []
    self.generateOrigin()

  def generateOrigin(self) -> None:
    """Generate genesis origin block for evolution tracking."""
    self.origin.append([0, 0, 0])

  def genID(self) -> None:
    """Generate a unique ID from NN coefficients or timestamp."""
    try:
      self.id = hashlib.md5(self.bot.nn.getAllCoefficents()).hexdigest()
    except Exception:
      k = str(time.time()).encode('utf-8')
      self.id = hashlib.md5(k).hexdigest()

  def setID(self, value: str) -> None:
    """Set the agent ID explicitly."""
    self.id = value

  def getDict(self) -> Dict[str, Any]:
    """Serialize agent state to a dictionary."""
    try:
        weights = self.bot.nn.weights.tolist()
    except AttributeError:
        weights = None
    return {
      "_id": self.id,
      'weights': weights,
      'elo': self.elo,
      'points': self.points,
    }

  def assignColour(self, colID: int) -> None:
    """Assign colour to agent (0=black, 1=white)."""
    self.colour = colID
    try:
      self.bot.currentColour = colID
    except AttributeError:
      pass

  def make_move(self, board: checkers.CheckerBoard, colour: int) -> int:
    """Make a move by delegating to the bot's move_function."""
    return self.bot.move_function(board, colour)
