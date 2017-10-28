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
  """
  Code based on
  https://visualstudiomagazine.com/Articles/2017/05/01/Python-and-NumPy.aspx
  """

  def __init__(self, numInput, numHidden, numHidden2, numOutput):
    self.numInputs = numInput
    self.numHidden1 = numHidden
    self.numOutput = numOutput
	
    self.iNodes = np.zeros(shape=[self.numInputs], dtype=np.float32)
    self.hNodes = np.zeros(shape=[self.numHidden1], dtype=np.float32)
    self.oNodes = np.zeros(shape=[self.numOutput], dtype=np.float32)
	
    self.ihWeights = np.zeros(shape=[self.numInputs,self.numHidden1], dtype=np.float32)
    self.hoWeights = np.zeros(shape=[self.numHidden1,self.numOutput], dtype=np.float32)
	
    self.hBiases = np.zeros(shape=[self.numHidden1], dtype=np.float32)
    self.oBiases = np.zeros(shape=[self.numOutput], dtype=np.float32)
	
    self.rnd = random.Random(0) # allows multiple instances
 	
  def setWeights(self, weights):
    if len(weights) != self.totalWeights(self.numInputs, self.numHidden1, self.numOutput):
      print("Warning: len(weights) error in setWeights()")	

    idx = 0
    for i in range(self.numInputs):
      for j in range(self.numHidden1):
        self.ihWeights[i][j] = weights[idx]
        idx += 1
		
    # adds biases
    for j in range(self.numHidden1):
      self.hBiases[j] = weights[idx]
      idx +=1

    for i in range(self.numHidden1):
      for j in range(self.numOutput):
        self.hoWeights[i][j] = weights[idx]
        idx += 1
	  
    # adds biases
    for k in range(self.numOutput):
      self.oBiases[k] = weights[idx]
      idx += 1
	  
  def getWeights(self):
    tw = self.totalWeights(self.numInputs, self.numHidden1, self.numOutput)
    result = np.zeros(shape=[tw], dtype=np.float32)
    idx = 0  # points into result
    
    for i in range(self.numInputs):
      for j in range(self.numHidden1):
        result[idx] = self.ihWeights[i][j]
        idx += 1
		
    for j in range(self.numHidden1):
      result[idx] = self.hBiases[j]
      idx +=1

    for i in range(self.numHidden1):
      for j in range(self.numOutput):
        result[idx] = self.hoWeights[i][j]
        idx += 1
	  
    for k in range(self.numOutput):
      result[idx] = self.oBiases[k]
      idx += 1
	  
    return result
 	
  def initialiseRandomWeights(self):
    numWts = self.totalWeights(self.numInputs, self.numHidden1, self.numOutput)
    weights = np.zeros(shape=[numWts], dtype=np.float32)
    lo = -0.2; hi = 0.2
    for idx in range(len(weights)):
      weights[idx] = (hi - lo) * self.rnd.random() + lo
    self.setWeights(weights)

  # deals with computing outputs.
  def computeOutputs(self, xValues):
    print("\n ihWeights: ")
    showMatrix(self.ihWeights, 2)
	
    print("\n hBiases: ")
    showVector(self.hBiases, 2)
	
    print("\n hoWeights: ")
    showMatrix(self.hoWeights, 2)
  
    print("\n oBiases: ")
    showVector(self.oBiases, 2)  
  
    hSums = np.zeros(shape=[self.numHidden1], dtype=np.float32)
    oSums = np.zeros(shape=[self.numOutput], dtype=np.float32)

    for i in range(self.numInputs):
      self.iNodes[i] = xValues[i]

    for j in range(self.numHidden1):
      for i in range(self.numInputs):
        hSums[j] += self.iNodes[i] * self.ihWeights[i][j]

    for j in range(self.numHidden1):
      hSums[j] += self.hBiases[j]
	  
    print("\n pre-tanh activation hidden node values: ")
    showVector(hSums, 4)

    for j in range(self.numHidden1):
      self.hNodes[j] = self.sigmoid(hSums[j])
	  
    print("\n after activation hidden node values: ")
    showVector(self.hNodes, 4)

    for k in range(self.numOutput):
      for j in range(self.numHidden1):
        oSums[k] += self.hNodes[j] * self.hoWeights[j][k]

    for k in range(self.numOutput):
      oSums[k] += self.oBiases[k]
	  
    print("\n pre-softmax output values: ")
    showVector(oSums, 4)

    if np.prod(oSums.shape) > 1:
      softOut = self.softmax(oSums)
      for k in range(self.numOutput):
        self.oNodes[k] = softOut[k]
  	  
      result = np.zeros(shape=self.numOutput, dtype=np.float32)
      for k in range(self.numOutput):
        result[k] = self.oNodes[k]
  	  
      return result
    else:
      return oSums

  # Evaluates Board
  def evaluateBoard(self, BoardState):
    xValues = np.array(BoardState, dtype=np.float32)
  
    # print("\n ihWeights: ")
    # showMatrix(self.ihWeights, 2)
  
    # print("\n hBiases: ")
    # showVector(self.hBiases, 2)
  
    # print("\n hoWeights: ")
    # showMatrix(self.hoWeights, 2)
  
    # print("\n oBiases: ")
    # showVector(self.oBiases, 2)  
  
    hSums = np.zeros(shape=[self.numHidden1], dtype=np.float32)
    oSums = np.zeros(shape=[self.numOutput], dtype=np.float32)

    for i in range(self.numInputs):
      self.iNodes[i] = xValues[i]

    for j in range(self.numHidden1):
      for i in range(self.numInputs):
        hSums[j] += self.iNodes[i] * self.ihWeights[i][j]

    for j in range(self.numHidden1):
      hSums[j] += self.hBiases[j]
    
    # print("\n pre-tanh activation hidden node values: ")
    # showVector(hSums, 4)

    for j in range(self.numHidden1):
      self.hNodes[j] = self.sigmoid(hSums[j])
    
    # print("\n after activation hidden node values: ")
    # showVector(self.hNodes, 4)

    for k in range(self.numOutput):
      for j in range(self.numHidden1):
        oSums[k] += self.hNodes[j] * self.hoWeights[j][k]

    for k in range(self.numOutput):
      oSums[k] += self.oBiases[k]
    
    # print("\n pre-softmax output values: ")
    # showVector(oSums, 4)

    if np.prod(oSums.shape) > 1:
      softOut = self.softmax(oSums)
      for k in range(self.numOutput):
        self.oNodes[k] = softOut[k]
      
      result = np.zeros(shape=self.numOutput, dtype=np.float32)
      for k in range(self.numOutput):
        result[k] = self.oNodes[k]
      
      return result
    else:
      # One item weight, just return that one number.
      return oSums.item(0)

  @staticmethod
  def sigmoid(val):
    # tanh function with the range of [-1,1]
    raw = math.tanh(val)
    if raw >= 0:
      return min(raw, 1)
    else:
      return max(raw, -1)

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

# -----------------------------------------------------------

class Slowpoke:
  """
  Machine Learning Agent Class
  """
  def __init__(self):
    """
    Initialise the Machine Learning Agent
    Note that we keep the weights since it is
    essential for the bot to evaluate the board.
    """
    self.weights = []
    self.kingWeight = 1.5
    self.nn = False
    self.ply = 4
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
    self.nnNodes = {
      'input' : 32,
      'hidden1' : 40,
      'hidden2' : 10,
      'output' : 1
    }
    # Now we can initialise the neural network.
    self.nn = NeuralNetwork(self.nnNodes['input'], self.nnNodes['hidden1'], 
                           self.nnNodes['hidden2'], self.nnNodes['output'])
    # make it initialise random weights.
    self.nn.initialiseRandomWeights()

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
    result = self.nn.evaluateBoard(boardArray)

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



if __name__ == "__main__":
  # np.random.seed(0)  # does not affect the NN
  numInput = 32
  numHidden1 = 40
  numHidden2 = 10
  numOutput = 1
  # print("Creating a %d-%d-%d-%d neural network " % (numInput, numHidden1, numHidden2, numOutput) )
  nn = NeuralNetwork(numInput, numHidden1, numHidden2, numOutput)
  # make it initialise random weights.
  nn.initialiseRandomWeights()
  
  # Insert checkerboard.
  xValues = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1.1, 1.1, 0, 0], dtype=np.float32)
  # Run Neural Network
  yValues = nn.computeOutputs(xValues)
  print("\nOutput values are: ")
  showVector(yValues, 4)

  print("\nEnd demo \n")

# end script
