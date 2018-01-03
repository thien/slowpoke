"""
  Slowpoke
                                      _.---"'----"'`--.._
                                _,.-'                   `-._
                            _,."                            -.
                        .-""   ___...---------.._             `.
                        `---'""                  `-.            `.
                                                    `.            \
                                                      `.           \
                                                        \           \
                                                          .           \
                                                          |            .
                                                          |            |
                                    _________             |            |
                              _,.-'"         `"'-.._      :            |
                          _,-'                      `-._.'             |
                      _.'                              `.             '
            _.-.    _,+......__                           `.          .
          .'    `-"'           `"-.,-""--._                 \        /
        /    ,'                  |    __  \                 \      /
        `   ..                       +"  )  \                 \    /
        `.'  \          ,-"`-..    |       |                  \  /
          / " |        .'       \   '.    _.'                   .'
        |,.."--"-"--..|    "    |    `""`.                     |
      ,"               `-._     |        |                     |
    .'                     `-._+         |                     |
    /                           `.                        /     |
    |    `     '                  |                      /      |
    `-.....--.__                  |              |      /       |
      `./ "| / `-.........--.-   '              |    ,'        '
        /| ||        `.'  ,'   .'               |_,-+         /
        / ' '.`.        _,'   ,'     `.          |   '   _,.. /
      /   `.  `"'"'""'"   _,^--------"`.        |    `.'_  _/
      /... _.`:.________,.'              `._,.-..|        "'
    `.__.'                                 `._  /
                                              "' 

  Slowpoke is a draughts AI based on a convolutional neural network.
  It's been trained using genetic algorithms :)
"""

import numpy as np
import random
import math
# Import Neural Network Class :)
from neural import NeuralNetwork

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

# -----------------------------------------------------------

class Slowpoke:
  
  """
  Initialisation Functions
  """

  def __init__(self, plyDepth=4, kingWeight=1.5, weights=[]):
    """
    Initialise Agent

    Note that we keep the weights since it is
    essential for the bot to evaluate the board.
    """
    self.nn = False
    self.ply = plyDepth

    self.pieceWeights = {
      "Black" : -1,
      "White" : 1,
      "empty" : 0,
      "blackKing" : -kingWeight,
      "whiteKing" : kingWeight
    }

    # Once we have everything we are ready to initiate
    # the board.
    self.initiateNeuralNetwork(weights)

  def initiateNeuralNetwork(self,weights=[]):
    """
    This function initiates the neural network and adds it
    to the AI class.
    """
    layers = [32,40,10,1]
    # Now we can initialise the neural network.
    self.nn = NeuralNetwork(layers)
    if weights:
      self.loadWeights(weights)
  
  def loadWeights(self, weights):
    # loads weights to the neural network
    self.nn.loadCoefficents(weights)

  """
  Move Functions (to be called by game.py)
  """

  def move_function(self, board, colour):
    # proceed to generate the possible moves the bot can make. For 
    # each one, we'll shove it in our neural network and generate
    # an outcome based on that.

    # Now we can shove boardStatus in the neural network!
    # call minimax algorithm
    move = self.miniMaxAB(board, self.ply, colour)

    # we'll need to get the current pieces of the board, manipulated 
    # in a way so that it can be shoved into the NN.
    # stage = self.checkGameStage(boardStatus)

    # Now we choose appropiately which outcome we want. From here, we add
    # that current board choice and the probability of it doing damage.
    # chance = chooseGameStage(chances, stage)
    return move

  def miniMaxAB(self, B, ply, colour):
    # We arbitrarily defined the value of a winning board as +1.0 and a losing board as −1.0. All other boards would receive values between −1.0 and +1.0, with a neural network favoring boards with higher values.
    minimax_win = 1
    minimax_lose = -minimax_win

    def minPlayAB(B, ply, alpha, beta, colour):
      if B.is_over():
        return minimax_win
      # get the moves
      moves = B.get_moves()
      # iterate through moves
      for move in moves:
        HB = B.copy()
        HB.make_move(move)

        if ply == 0:
          score = self.evaluateBoard(HB, colour)
        else:
          score = maxPlayAB(HB, ply-1, alpha, beta, colour)
        
        if score < beta:
          beta = score
        if beta <= alpha:
          return beta
      return beta

    def maxPlayAB(B, ply, alpha, beta, colour):
      if B.is_over():
        return minimax_lose
      # get the moves
      moves = B.get_moves()
      # iterate through moves
      for move in moves:
        HB = B.copy()
        HB.make_move(move)

        if ply == 0:
          score = self.evaluateBoard(HB, colour)
        else:
          score = minPlayAB(HB, ply-1, alpha, beta, colour)

        if score > alpha:
          alpha = score
        if alpha >= beta:
          return alpha  
      return alpha

    # ---------------------------------------------
    moves = B.get_moves()
    best_move = moves[0]
    best_score = float('-inf')

    alpha = float('-inf')
    beta = float('inf')

    # iterate through the current possible moves.
    for move in moves:
      HB = B.copy()
      HB.make_move(move)
      if ply == 0:
        score = self.evaluateBoard(HB)
      else:
        score = minPlayAB(HB, ply-1, alpha, beta, colour)
      if score > best_score:
        best_move = move
        best_score = score
    return best_move

  """
  Evaluation Function calls the neural network and evaluates the current position of the board.
  """

  def evaluateBoard(self,board,colour):
    """
    We throw in the board into the neural network here, and
    then the neural network evaluates the position of the
    board.

    Make the bot think it's always playing from blacks perspective.
    """

    # Get the current status of the board.
    boardStatus = board.getBoardPosWeighted(colour, self.pieceWeights)
    # Get an array of the board.
    boardArray = np.array(boardStatus,dtype=np.float32)
    # Evaluate the board array using our CNN.
    result = self.nn.compute(boardStatus)

    # Return the results.
    return result

  """
  Prints the colour string.
  """
  @staticmethod
  def getColourString():
    if config['colour'] == pieceWeights['Black']:
      return "Black"
    elif config['colour'] == pieceWeights['White']:
      return "White"
    else:
      return "NAN"


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

