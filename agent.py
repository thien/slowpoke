"""
Agent

This represents a computer player;
It'll contain information about its ranking,
and its move function.

It also has a default ELO.
"""

import hashlib

class Agent:
  def __init__(self, bot):
    self.bot = bot
    self.elo = 1600
    self.points = 0
    self.move_function = bot.move_function
    self.colour = None

  def generateID(self):
    self._id = hashlib.md5(bot.nn.weights).hexdigest()

  def assignColour(self,colID):
    # black is 0, white is 1
    self.colour = colID
    self.bot.currentColour = colID

  def make_move(self, board):
    return self.bot.move_function(board)
