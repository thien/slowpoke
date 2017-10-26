# -*- coding: utf-8 -*-

"""
Alakazam is a draughts AI based on a feed forward
neural network chucked alongside a genetic algorithm.
"""
import os
import numpy as np
import random
import math

# we'll be using PyTorch
import torch
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

# helper functions

def loadFile(df):
  # load a comma-delimited text file into an np matrix
  resultList = []
  f = open(df, 'r')
  for line in f:
    line = line.rstrip('\n')  # "1.0,2.0,3.0"
    sVals = line.split(',')   # ["1.0", "2.0, "3.0"]
    fVals = list(map(np.float3232, sVals))  # [1.0, 2.0, 3.0]
    resultList.append(fVals)  # [[1.0, 2.0, 3.0] , [4.0, 5.0, 6.0]]
  f.close()
  return np.asarray(resultList, dtype=np.float32)  # not necessary
# end loadFile
  
def showVector(v, dec):
  fmt = "%." + str(dec) + "f" # like %.4f
  for i in range(len(v)):
    x = v[i]
    if x >= 0.0: print(' ', end='')
    print(fmt % x + '  ', end='')
  print('')
  
def showMatrix(m, dec):
  fmt = "%." + str(dec) + "f" # like %.4f  
  for i in range(len(m)):
    for j in range(len(m[i])):
      x = m[i,j]
      if x >= 0.0: print(' ', end='')
      print(fmt % x + '  ', end='')
    print('')
  
# -----
  
class NeuralNetwork:

  def __init__(self, numInput, numHidden, numOutput):
    self.ni = numInput
    self.nh = numHidden
    self.no = numOutput
  
    self.iNodes = np.zeros(shape=[self.ni], dtype=np.float32)
    self.hNodes = np.zeros(shape=[self.nh], dtype=np.float32)
    self.oNodes = np.zeros(shape=[self.no], dtype=np.float32)
  
    self.ihWeights = np.zeros(shape=[self.ni,self.nh], dtype=np.float32)
    self.hoWeights = np.zeros(shape=[self.nh,self.no], dtype=np.float32)
  
    self.hBiases = np.zeros(shape=[self.nh], dtype=np.float32)
    self.oBiases = np.zeros(shape=[self.no], dtype=np.float32)
  
    self.rnd = random.Random(0) # allows multiple instances
    self.initializeWeights()
  
  def setWeights(self, weights):
    if len(weights) != self.totalWeights(self.ni, self.nh, self.no):
      print("Warning: len(weights) error in setWeights()")  

    idx = 0
    for i in range(self.ni):
      for j in range(self.nh):
        self.ihWeights[i][j] = weights[idx]
        idx += 1
    
    for j in range(self.nh):
      self.hBiases[j] = weights[idx]
      idx +=1

    for i in range(self.nh):
      for j in range(self.no):
        self.hoWeights[i][j] = weights[idx]
        idx += 1
    
    for k in range(self.no):
      self.oBiases[k] = weights[idx]
      idx += 1
    
  def getWeights(self):
    tw = self.totalWeights(self.ni, self.nh, self.no)
    result = np.zeros(shape=[tw], dtype=np.float32)
    idx = 0  # points into result
    
    for i in range(self.ni):
      for j in range(self.nh):
        result[idx] = self.ihWeights[i][j]
        idx += 1
    
    for j in range(self.nh):
      result[idx] = self.hBiases[j]
      idx +=1

    for i in range(self.nh):
      for j in range(self.no):
        result[idx] = self.hoWeights[i][j]
        idx += 1
    
    for k in range(self.no):
      result[idx] = self.oBiases[k]
      idx += 1
    
    return result
  
  def initializeWeights(self):
    numWts = self.totalWeights(self.ni, self.nh, self.no)
    wts = np.zeros(shape=[numWts], dtype=np.float32)
    lo = -0.01; hi = 0.01
    for idx in range(len(wts)):
      wts[idx] = (hi - lo) * self.rnd.random() + lo
    self.setWeights(wts)

  def computeOutputs(self, xValues):
    print("\n ihWeights: ")
    showMatrix(self.ihWeights, 2)
  
    print("\n hBiases: ")
    showVector(self.hBiases, 2)
  
    print("\n hoWeights: ")
    showMatrix(self.hoWeights, 2)
  
    print("\n oBiases: ")
    showVector(self.oBiases, 2)  
  
    hSums = np.zeros(shape=[self.nh], dtype=np.float32)
    oSums = np.zeros(shape=[self.no], dtype=np.float32)

    for i in range(self.ni):
      self.iNodes[i] = xValues[i]

    for j in range(self.nh):
      for i in range(self.ni):
        hSums[j] += self.iNodes[i] * self.ihWeights[i][j]

    for j in range(self.nh):
      hSums[j] += self.hBiases[j]
    
    print("\n pre-tanh activation hidden node values: ")
    showVector(hSums, 4)

    for j in range(self.nh):
      self.hNodes[j] = self.hypertan(hSums[j])
    
    print("\n after activation hidden node values: ")
    showVector(self.hNodes, 4)

    for k in range(self.no):
      for j in range(self.nh):
        oSums[k] += self.hNodes[j] * self.hoWeights[j][k]

    for k in range(self.no):
      oSums[k] += self.oBiases[k]
    
    print("\n pre-softmax output values: ")
    showVector(oSums, 4)

    softOut = self.softmax(oSums)
    for k in range(self.no):
      self.oNodes[k] = softOut[k]
    
    result = np.zeros(shape=self.no, dtype=np.float32)
    for k in range(self.no):
      result[k] = self.oNodes[k]
    
    return result
  
  @staticmethod
  def hypertan(x):
    if x < -20.0:
      return -1.0
    elif x > 20.0:
      return 1.0
    else:
      return math.tanh(x)

  @staticmethod   
  def softmax(oSums):
    result = np.zeros(shape=[len(oSums)], dtype=np.float32)
    m = max(oSums)
    divisor = 0.0
    for k in range(len(oSums)):
       divisor += math.exp(oSums[k] - m)
    for k in range(len(result)):
      result[k] =  math.exp(oSums[k] - m) / divisor
    return result
  
  @staticmethod
  def totalWeights(nInput, nHidden, nOutput):
   tw = (nInput * nHidden) + (nHidden * nOutput) + nHidden + nOutput
   return tw

# end class NeuralNetwork

def main():
  print("\nBegin NN demo \n")
  
  # np.random.seed(0)  # does not affect the NN
  numInput = 3
  numHidden = 4
  numOutput = 2
  print("Creating a %d-%d-%d neural network " % (numInput, numHidden, numOutput) )
  nn = NeuralNetwork(numInput, numHidden, numOutput)
  
  print("\nSetting weights and biases ")
  numWts = NeuralNetwork.totalWeights(numInput, numHidden, numOutput)
  wts = np.zeros(shape=[numWts], dtype=np.float32)  # 26 cells
  for i in range(len(wts)):
    wts[i] = ((i+1) * 0.01)  # [0.01, 0.02, . . 0.26 ]
  nn.setWeights(wts)
  
  # wts = nn.getWeights()  # verify weights and bias values
  # showVector(wts, 2)

  xValues = np.array([1.0, 2.0, 3.0], dtype=np.float32)
  print("\nInput values are: ")
  showVector(xValues, 1)
  
  yValues = nn.computeOutputs(xValues)
  print("\nOutput values are: ")
  showVector(yValues, 4)

  print("\nEnd demo \n")
   
if __name__ == "__main__":
  main()

# end script
