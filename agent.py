"""
Agent

This represents a computer player;
It'll contain information about its ranking,
and its move function.

It also has a default ELO.
"""

import hashlib
import time

class Agent:
  def __init__(self, bot, agent_id=None):
    self.bot = bot
    self.elo = 1200
    self.points = 0
    self.champRange = 0
    self.champScore = 0
    self.move_function = bot.move_function
    self.colour = None
    if agent_id is not None:
      self.setID(agent_id)
    else:
      self.genID()

  def genID(self):
    try:
      self.id = hashlib.md5(self.bot.nn.getAllCoefficents()).hexdigest()
    except:
      # default into using the time as the checksum
      k = str(time.time()).encode('utf-8')
      self.id = hashlib.md5(k).hexdigest()

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

  def make_move(self, board, colour):
    return self.bot.move_function(board, colour)
