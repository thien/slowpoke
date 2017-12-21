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
  Machine Learning Agent Class
  """
  def __init__(self, plyDepth=4):
    """
    Initialise the Machine Learning Agent
    Note that we keep the weights since it is
    essential for the bot to evaluate the board.
    """
    self.weights = []
    self.kingWeight = 1.5
    self.nn = False
    self.ply = plyDepth
    self.currentColour = None

    """
    Custom Weights:
    These weights are considered on the board.
    """
    self.customWeights = {
      "Black" : -1,
      "White" : 1,
      "empty" : 0,
      "blackKing" : -1.5,
      "whiteKing" : 1.5
    }

    # Once we have everything we are ready to initiate
    # the board.
    self.initiateNeuralNetwork(self.weights)

  def initiateNeuralNetwork(self,weights):
    """
    This function initiates the neural network and adds it
    to the AI class.
    """
    layers = [32,40,10,1]
    # Now we can initialise the neural network.
    self.nn = NeuralNetwork(layers)
  

  def evaluateBoard(self,board):
    """
    We throw in the board into the neural network here, and
    then the neural network evaluates the position of the
    board.
    """

    # Get the current status of the board.
    colour = self.currentColour
    boardStatus = board.getBoardPosWeighted(self.currentColour, self.customWeights)
    # Get an array of the board.
    boardArray = np.array(boardStatus,dtype=np.float32)
    # Evaluate the board array using our CNN.
    result = self.nn.computeOutputs(boardStatus)

    # Return the results.
    return result

  def miniMax(self, B, ply):
    def minPlay(B, ply):
      if B.is_over():
        return 1
      moves = B.get_moves()
      best_score = float('inf')
      for move in moves:
        HB = B.copy()
        HB.make_move(move)
        if ply == 0:
          score = self.evaluateBoard(HB)
        else:
          score = maxPlay(HB, ply-1)
        if score < best_score:
          best_move = move
          best_score = score
        return best_score
    def maxPlay(B, ply):
      if B.is_over():
        return -1
      moves = B.get_moves()
      best_score = float('-inf')
      for move in moves:
        HB = B.copy()
        HB.make_move(move)
        if ply == 0:
          score = self.evaluateBoard(HB)
        else:
          score = minPlay(HB, ply-1)
        if score > best_score:
          best_move = move
          best_score = score
        return best_score

    moves = B.get_moves()
    best_move = moves[0]
    best_score = float('-inf')

    # iterate through the current possible moves.
    for move in moves:
      HB = B.copy()
      HB.make_move(move)
      if ply == 0:
        score = self.evaluateBoard(HB)
      else:
        score = minPlay(HB, ply-1)
      if rating > best_score:
        best_move = move
        best_score = score
      return best_move

  def miniMaxAB(self, B, ply):
    def minPlayAB(B, ply, alpha, beta):
      if B.is_over():
        return 1

      # get the moves
      moves = B.get_moves()

      # iterate through moves
      for move in moves:
        HB = B.copy()
        HB.make_move(move)

        if ply == 0:
          score = self.evaluateBoard(HB)
        else:
          score = maxPlayAB(HB, ply-1, alpha, beta)

        if score < beta:
          beta = score
        if beta <= alpha:
          return beta
      return beta

    def maxPlayAB(B, ply, alpha, beta):
      if B.is_over():
        return -1

      # get the moves
      moves = B.get_moves()

      # iterate through moves
      for move in moves:
        HB = B.copy()
        HB.make_move(move)

        if ply == 0:
          score = self.evaluateBoard(HB)
        else:
          score = minPlayAB(HB, ply-1, alpha, beta)

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
        score = minPlayAB(HB, ply-1, alpha, beta)
      if score > best_score:
        best_move = move
        best_score = score
    return best_move

  def chooseMove(self,board):
    # call minimax algorithm
    move = self.miniMaxAB(board, self.ply)
    # return that move.
    return move

  def move_function(self, board):
    # # first we need to establish whether we're white or black.
    # self.checkColour(col)

    # # This function deals with getting the colour that the bot represents.
    # self.getColourString()

    # now that we have our colour, we can now proceed to generate the 
    # possible moves the bot can make. For each one, we'll shove it in our
    # neural network and generate an outcome based on that.

    # we'll need to get the current pieces of the board, manipulated 
    # in a way so that it can be shoved into the NN.
    # boardStatus = board.getBoardPosWeighted(self.currentColour, self.customWeights)
    # stage = self.checkGameStage(boardStatus)

    # Now we can shove boardStatus in the neural network!
    move = self.chooseMove(board)
    # Now we choose appropiately which outcome we want. From here, we add
    # that current board choice and the probability of it doing damage.
    # chance = chooseGameStage(chances, stage)


    return move

  """
  Checks the current stage of the board.
  """
  @staticmethod
  def checkGameStage(board):
    """
    There are three stages of Checkers:
      Beginning: Both players have at least three pieces on the board. No kings.
      Kings: Both players have at least three pieces on the board. At least one king.
      Ending: one player has less than three pieces on the board.

    Returns:
      A value that is either 0,1, or 2.
      0: beginning
      1: kings
      2: ending
    """
    b = 0
    w = 0
    kb = 0
    kw = 0
    for i in range(len(board)):
      if pieceWeights['empty'] != board[i]:
        if board[i] == pieceWeights['Black']:
          b += 1
        elif board[i] == pieceWeights['White']:
          w += 1
        elif board[i] == pieceWeights['blackKing']:
          kb += 1
        elif board[i] == pieceWeights['whiteKing']:
          kw += 1
    if b >= 3 and w >= 3:
      # we're either at Kings or Beginning.
      if kb > 1 or kw > 1:
        # we're at intermediate.
        return 1
      else:
        # we're at beginning.
        return 0
    else:
      # we've reached the ending.
      return 2

  @staticmethod
  def chooseGameStage(nn_results, stage):
    if stage == 0:
      return nn_results[0]
    elif stage == 1:
      return nn_results[1]
    else:
      return nn_results[2]

  # """
  # Checks the board's colour choice; this is needed so we can
  # determine whether we need to rotate the board and manipulate
  # the inputs for the NN.
  # """
  # @staticmethod
  # def checkColour(colour):
  #   if config['colourCheck'] == False:
  #     if colour == pieceWeights['Black']:
  #       config['colour'] = pieceWeights['Black']
  #     else:
  #       config['colour'] = pieceWeights['White']
  #     return config['colour']

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

