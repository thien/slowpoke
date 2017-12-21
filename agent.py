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
    self.elo = 1200
    self.points = 0
    self.champRange = 0
    self.champScore = 0
    self.move_function = bot.move_function
    self.colour = None
    self.genID()

  def genID(self):
    self.id = hashlib.md5(self.bot.nn.getAllCoefficents()).hexdigest()

  def setID(self, value):
    self.id = value

  def getDict(self):
    return {
      "_id" : self.id,
      'weights': self.bot.nn.weights.tolist(),
      'elo': self.elo,
      'points' : self.points
    }

  def assignColour(self,colID):
    # black is 0, white is 1
    self.colour = colID
    self.bot.currentColour = colID

  def make_move(self, board):
    return self.bot.move_function(board)
