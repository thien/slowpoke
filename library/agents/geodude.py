"""
  Geodude
                                                  _,.---.
                                              _,-'       `.
                                          _,'  ,          \
                                        ,'  _,'   .        `.
                                        /  ,'     ,'          `.
              __                       .,'    _,'              `.
          _,..'  `-....___              :    ,'     '             \
        ,'   /            :             /`.,'      /               `
      /    /  ._         |         __..|  `.    .'       ,         `.
      |   |   ,'"--._    |      ,-'    `-._`.,-'       ,:            .
      .'\   \     _,'.    `'___.'           `"`.     _,' /            |
      |  \   \---'       ,"'  .-""'"----.       `.  '  ,'             |
      `. `-.'          /    /                    `-..^._             '
        |._|    _.    /    /                            `._           .
        `...:--'--+..'   ,'                              /            |
            '._  `|   ,-'       _..._                   j     \       |
              |` |   /       ,-'     `-.__              |      L      |
              |  |  /      ,'                           |      |      |
              |_,'        /         _,-                  .     |      |
              ,'  ,   |  ,'        ,|            ,..._     \    |      '
            ,     \ j  '       _." |           /     `-.__'    '    ,'
              +._   '|       ,'|    |          /        ,'    .'    /
              |  `._  `-' .:|  |    '.       -'        '           j
              '    |`    ' |'  |     |                             |
              `.  |       |--'     _|        .                    |
                \ |       '----'"'"'           \      __,....-+----'
                | '                            `---""      .' 
                `. `.                                     ,
                  `" \_...-"''"'--..         _+          ,'
                        '            -.'  `'  `.  ."-..'
                        `-..'._            _____,.'
                              `-'-'.....,-"' mh
"""

import numpy as np
import random
import math
import datetime
from math import log, sqrt

# import decision files
import sys
sys.path.insert(0, '..')
import decision.mcts as mcts

"""
Piece Weights
"""
pieceWeights = {
  "Black" : 0,
  "White" : 1,
  "empty" : -1,
  "blackKing" : -2,
  "whiteKing" : -3
}

# We arbitrarily defined the value of a winning board as +1.0 and a losing board as −1.0. All other boards would receive values between −1.0 and +1.0, with a neural network favoring boards with higher values.

minimax_win = 1
minimax_lose = -minimax_win
minimax_draw = 0
minimax_empty = -1

# -----------------------------------------------------------

class Geodude:
  def __init__(self, plyDepth=4):
    """
    Initialise Agent

    Note that we keep the weights since it is
    essential for the bot to evaluate the board.
    """
    self.ply = plyDepth
    self.decisionFunction = mcts.MCTS(self.ply)


  def move_function(self, board, colour):
    # return self.mcts_code(board,self.ply, colour)
    return self.decisionFunction.Decide(board, colour)