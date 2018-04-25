import datetime
from .evaluator.neural import NeuralNetwork
import json

# import decision files
import sys
sys.path.insert(0, '..')
import decision.mcts as mcts
import decision.tmcts as tmcts
import decision.minimax as minimax

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

# We arbitrarily defined the value of a winning board as +1.0 and a losing board as −1.0. All other boards would receive values between −1.0 and +1.0, with a neural network favouring boards with higher values.

minimax_win = 1
minimax_lose = -minimax_win
minimax_draw = 0
minimax_empty = -1

# -----------------------------------------------------------

class Slowpoke:
  
  def __init__(self, plyDepth=4, kingWeight=1.5, weights=[], layers=[91,40,10,1], isminimax=False, debug=False):
    """
    Initialise Agent

    Note that we keep the weights since it is
    essential for the bot to evaluate the board.
    """
    self.debug = debug
    self.chooseMinimax = isminimax
    self.nn = False
    self.ply = plyDepth
    self.layers = layers
    self.pieceWeights = {
      "Black" : 1,
      "White" : -1,
      "empty" : 0,
      "blackKing" : kingWeight,
      "whiteKing" : -kingWeight
    }

    # Once we have everything we are ready to initiate
    # the board.
    self.initiateNeuralNetwork(layers, weights)
    self.movesConsidered = []

    self.decisionFunction = None
    if isminimax:
      self.decisionFunction = minimax.MiniMax(self.ply, self.evaluate_board)
    else:
      self.decisionFunction = tmcts.TMCTS(self.ply,self.evaluate_board, debug=self.debug)

    # optional cache
    self.cache = {}
    self.enableCache = True

  def initiateNeuralNetwork(self, layers, weights=[]):
    """
    This function initiates the neural network and adds it
    to the AI class.
    """
    # Now we can initialise the neural network.
    self.nn = NeuralNetwork(layers)
    if weights:
      self.loadWeights(weights)
  
  def loadWeights(self, weights):
    # loads weights to the neural network
    self.nn.loadCoefficents(weights)

  def move_function(self, board, colour):
    return self.decisionFunction.Decide(board, colour)

  def evaluate_board(self,board,colour):
    """
    We throw in the board into the neural network here, and
    then the neural network evaluates the position of the
    board.

    Make the bot think it's always playing from blacks perspective.
    """

    if board.is_over():
      if board.winner != minimax_empty:
        if board.winner == colour:
          # print(board)
          # print(colour, ", you is winner")
          # input()
          return minimax_win
        else:
          return minimax_lose
      else:
        return minimax_draw
    else:
      # print("you are", colour)
      # Get the current status of the board.
      boardStatus = board.getBoardPosWeighted(colour, self.pieceWeights)

      
      if self.layers[0] == 91:
        boardStatus = self.nn.subsquares(boardStatus)
      # Evaluate the board array using our CNN.
      hashd = None
      if self.enableCache:
        hashd = tuple(boardStatus)
        if hashd in self.cache:
          return self.cache[hashd]

      val = self.nn.compute(boardStatus)

      if self.enableCache:
        self.cache[hashd] = val
      return val

  # """
  # Checks the current stage of the board.
  # """
  # @staticmethod
  # def checkGameStage(board):
  #   """
  #   There are three stages of Checkers:
  #     Beginning: Both players have at least three pieces on the board. No kings.
  #     Kings: Both players have at least three pieces on the board. At least one king.
  #     Ending: one player has less than three pieces on the board.

  #   Returns:
  #     A value that is either 0,1, or 2.
  #     0: beginning
  #     1: kings
  #     2: ending
  #   """
  #   b = 0
  #   w = 0
  #   kb = 0
  #   kw = 0
  #   for i in range(len(board)):
  #     if pieceWeights['empty'] != board[i]:
  #       if board[i] == pieceWeights['Black']:
  #         b += 1
  #       elif board[i] == pieceWeights['White']:
  #         w += 1
  #       elif board[i] == pieceWeights['blackKing']:
  #         kb += 1
  #       elif board[i] == pieceWeights['whiteKing']:
  #         kw += 1
  #   if b >= 3 and w >= 3:
  #     # we're either at Kings or Beginning.
  #     if kb > 1 or kw > 1:
  #       # we're at intermediate.
  #       return 1
  #     else:
  #       # we're at beginning.
  #       return 0
  #   else:
  #     # we've reached the ending.
  #     return 2

  # @staticmethod
  # def chooseGameStage(nn_results, stage):
  #   if stage == 0:
  #     return nn_results[0]
  #   elif stage == 1:
  #     return nn_results[1]
  #   else:
  #     return nn_results[2]

