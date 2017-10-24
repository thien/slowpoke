# -*- coding: utf-8 -*-

"""
Alakazam is a draughts AI based on a feed forward
neural network chucked alongside a genetic algorithm.
"""
import random
import os

# we'll be using PyTorch
import torch
import random
from torch.autograd import Variable



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
customWeights = {
  "Black" : -1,
  "White" : 1,
  "empty" : 0,
  "blackKing" : -1.5,
  "whiteKing" : 1.5
}

"""
Configuration 
(don't change this unless you know what you're doing!)
"""
config = {
  "colour": None,
  "colourCheck" : False
}

"""
----------------------------------------------------------------------
Functions that deal with draughts logic.
----------------------------------------------------------------------
"""

"""
Checks the current stage of the board.
"""
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


def chooseGameStage(nn_results, stage):
  if stage == 0:
    return nn_results[0]
  elif stage == 1:
    return nn_results[1]
  else:
    return nn_results[2]

"""
Checks the board's colour choice; this is needed so we can
determine whether we need to rotate the board and manipulate
the inputs for the NN.
"""
def checkColour(colour):
  if config['colourCheck'] == False:
    if colour == pieceWeights['Black']:
      config['colour'] = pieceWeights['Black']
    else:
      config['colour'] = pieceWeights['White']
    return config['colour']

"""
Prints the colour string.
"""
def getColourString():
  if config['colour'] == pieceWeights['Black']:
    return "Black"
  elif config['colour'] == pieceWeights['White']:
    return "White"
  else:
    return "NAN"



"""
This function gets called by the game client!
"""
def move_function(board, col, depth):
  # first we need to establish whether we're white or black.
  checkColour(col)

  # This function deals with getting the colour that the bot represents.
  getColourString()


  # now that we have our colour, we can now proceed to generate the 
  # possible moves the bot can make.


  # we'll need to get the current pieces of the board, manipulated 
  # in a way so that it can be shoved into the NN.
  boardStatus = board.getBoardPosWeighted(config['colour'], customWeights)
  
  # Now we can shove boardStatus in the neural network!
  runNN(boardStatus)

  # 
  return random.choice(board.get_moves())


"""
This function gets called by the game client! (Pimped)
"""
def move_function2(board, col, depth):
  # first we need to establish whether we're white or black.
  checkColour(col)

  # This function deals with getting the colour that the bot represents.
  getColourString()


  # now that we have our colour, we can now proceed to generate the 
  # possible moves the bot can make. For each one, we'll shove it in our
  # neural network and generate an outcome based on that.

  # we'll need to get the current pieces of the board, manipulated 
  # in a way so that it can be shoved into the NN.
  boardStatus = board.getBoardPosWeighted(config['colour'], customWeights)
  stage = checkGameStage(boardStatus)

  # Now we can shove boardStatus in the neural network!
  chances = runNN(boardStatus)
  # Now we choose appropiately which outcome we want. From here, we add
  # that current board choice and the probability of it doing damage.
  chance = chooseGameStage(chances, stage)


  return random.choice(board.get_moves())


"""
----------------------------------------------------------------------
Neural Network Logic
We're using PyTorch for this.
----------------------------------------------------------------------
"""

class dynamicNN(torch.nn.Module):
  def __init__(self, inputDimension, hiddenDimension, outputDimension):
    """
    In the constructor we construct three nn.Linear instances that we will use
    in the forward pass.
    """
    super(dynamicNN, self).__init__()
    self.input_linear = torch.nn.Linear(inputDimension, hiddenDimension)
    self.middle_linear = torch.nn.Linear(hiddenDimension, hiddenDimension)
    self.output_linear = torch.nn.Linear(hiddenDimension, outputDimension)

  def forward(self, x):
    """
    For the forward pass of the model, we iterate 3 times
    and reuse the middle_linear Module that many times to compute hidden layer
    representations.

    Since each forward pass builds a dynamic computation graph, we can use normal
    Python control-flow operators like loops or conditional statements when
    defining the forward pass of the model.
    """
    hidden_relu = self.input_linear(x).clamp(min=0)
    print("--------------------------------------------------------------------")
    print("start",hidden_relu)
    for sk in range(0, 3):
      hidden_relu = self.middle_linear(hidden_relu).clamp(min=0)
      print(sk,hidden_relu)
    y_pred = self.output_linear(hidden_relu)
    return y_pred


# -------------------------------------------------------------------------------

def runNN(board):
  # batchSize is batch size; inputDimension is input dimension;
  # hiddenDimension is hidden dimension; outputDimension is output dimension.
  batchSize, inputDimension, hiddenDimension, outputDimension = 1, 10, 20, 3

  # This represents the number of trials
  trials = 43

  based = torch.Tensor([1.0, 1.1, 1.1, 0, 1])
  print(based)

  # Create random Tensors to hold inputs and outputs, and wrap them in Variables
  baseInput = torch.randn(batchSize, inputDimension)

  print(baseInput)
  x = Variable(baseInput)
  
  # Construct our model by instantiating the class defined above
  model = dynamicNN(inputDimension, hiddenDimension, outputDimension)

  print("STARTING ML")
  y_pred = model(x)
  print(y_pred)

  return y_pred

runNN(0)