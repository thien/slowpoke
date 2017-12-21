import numpy as np
import random
import math

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
  Code based on a tutorial at the following source:
  https://visualstudiomagazine.com/Articles/2017/05/01/Python-and-NumPy.aspx

  The NN since then has massively diverged :)
  """

  def __init__(self, numInput, numHidden, numHidden2, numOutput):
    self.numInputs = numInput
    self.numHidden1 = numHidden
    self.numHidden2 = numHidden2
    self.numOutput = numOutput
  
    self.iNodes = np.zeros(shape=[self.numInputs], dtype=np.float32)
    self.h1Nodes = np.zeros(shape=[self.numHidden1], dtype=np.float32)
    self.h2Nodes = np.zeros(shape=[self.numHidden2], dtype=np.float32)
    self.oNodes = np.zeros(shape=[self.numOutput], dtype=np.float32)
  
    self.ihWeights = np.zeros(shape=[self.numInputs,self.numHidden1], dtype=np.float32)
    self.hhWeights = np.zeros(shape=[self.numHidden1,self.numHidden2], dtype=np.float32)
    self.hoWeights = np.zeros(shape=[self.numHidden2,self.numOutput], dtype=np.float32)
  
    self.h1Biases = np.zeros(shape=[self.numHidden1], dtype=np.float32)
    self.h2Biases = np.zeros(shape=[self.numHidden2], dtype=np.float32)
    self.oBiases = np.zeros(shape=[self.numOutput], dtype=np.float32)

    self.weights = None
    self.numberOfWeights = 1741
    # self.rnd = random.Random(0) # this is a seed!
    self.rnd = random.Random(1) # allows multiple instances
  
  def lenWeights(self):
    return self.numberOfWeights

  def setWeights(self, weights):
    if len(weights) != self.totalWeights(self.numInputs, self.numHidden1,  self.numHidden2, self.numOutput):
      print("Warning: len(weights) error in setWeights()")  

    count = 0

    # adds weights for input to hidden 1
    for i in range(self.numInputs):
      for j in range(self.numHidden1):
        # print("memes")
        # print(weights[count])
        # print(count, len(weights))
        self.ihWeights[i][j] = weights[count]
        count += 1
    
    # adds biases for hidden 1
    for j in range(self.numHidden1):
      self.h1Biases[j] = weights[count]
      count +=1

    # adds weights for hidden 1 to 2
    for i in range(self.numHidden1):
      for j in range(self.numHidden2):
        self.hhWeights[i][j] = weights[count]
        count += 1

    # adds biases for hidden 2
    for j in range(self.numHidden2):
      self.h2Biases[j] = weights[count]
      count +=1

    # adds weights for hidden 1 to output
    for i in range(self.numHidden2):
      for j in range(self.numOutput):
        self.hoWeights[i][j] = weights[count]
        count += 1
    
    # adds biases for output
    for k in range(self.numOutput):
      self.oBiases[k] = weights[count]
      count += 1

  def getWeights(self):
    tw = self.totalWeights(self.numInputs, self.numHidden1, self.numHidden2, self.numOutput)
    result = np.zeros(shape=[tw], dtype=np.float32)
    count = 0  # points into result
    
    # get weights for input->hidden1
    for i in range(self.numInputs):
      for j in range(self.numHidden1):
        result[count] = self.ihWeights[i][j]
        count += 1
    # get biases for hidden1
    for j in range(self.numHidden1):
      result[count] = self.h1Biases[j]
      count +=1

    # get weights for hidden1->hidden2
    for i in range(self.numHidden1):
      for j in range(self.numHidden2):
        result[count] = self.hhWeights[i][j]
        count += 1

    # get biases for hidden1
    for j in range(self.numHidden2):
      result[count] = self.h2Biases[j]
      count +=1

    # get weights for hidden2->output
    for i in range(self.numHidden2):
      for j in range(self.numOutput):
        result[count] = self.hoWeights[i][j]
        count += 1
    
    # get biases for output
    for k in range(self.numOutput):
      result[count] = self.oBiases[k]
      count += 1
    
    return result
  
  def initialiseRandomWeights(self):
    numWts = self.totalWeights(self.numInputs, self.numHidden1, self.numHidden2, self.numOutput)
    weights = np.zeros(shape=[numWts], dtype=np.float32)
    lo = -0.2; hi = 0.2
    for count in range(len(weights)):
      weights[count] = (hi - lo) * self.rnd.random() + lo
    # print(weights)
    self.weights = weights
    self.setWeights(weights)

  # deals with computing outputs.
  def computeOutputs(self, xValues):
    # print("\n ihWeights: ")
    # showMatrix(self.ihWeights, 2)
  
    # print("\n h1Biases: ")
    # showVector(self.h1Biases, 2)

    # print("\n h2Biases: ")
    # showVector(self.h2Biases, 2)
  
    # print("\n hoWeights: ")
    # showMatrix(self.hoWeights, 2)
  
    # print("\n oBiases: ")
    # showVector(self.oBiases, 2)  
  
    h1Sums = np.zeros(shape=[self.numHidden1], dtype=np.float32)
    h2Sums = np.zeros(shape=[self.numHidden2], dtype=np.float32)
    oSums = np.zeros(shape=[self.numOutput], dtype=np.float32)

    # deal with input nodes
    for i in range(self.numInputs):
      self.iNodes[i] = xValues[i]

# --------------
    # deal with hidden layer 1
    for j in range(self.numHidden1):
      for i in range(self.numInputs):
        h1Sums[j] += self.iNodes[i] * self.ihWeights[i][j]

    for j in range(self.numHidden1):
      h1Sums[j] += self.h1Biases[j]

    # print("\n pre-tanh activation hidden node values: ")
    # showVector(h1Sums, 4)

    for j in range(self.numHidden1):
      self.h1Nodes[j] = self.sigmoid(h1Sums[j])

    # print("\n after activation hidden node values: ")
    # showVector(self.h1Nodes, 4)

# --------------
    # deal with hidden layer 2
    for j in range(self.numHidden2):
      for i in range(self.numHidden1):
        h2Sums[j] += h1Sums[i] * self.hhWeights[i][j]

    for j in range(self.numHidden2):
      h2Sums[j] += self.h2Biases[j]

    for j in range(self.numHidden2):
      self.h2Nodes[j] = self.sigmoid(h2Sums[j])

# --------------

    for k in range(self.numOutput):
      for j in range(self.numHidden2):
        oSums[k] += h2Sums[j] * self.hoWeights[j][k]

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
      return oSums

  # board evaluation function
  def evaluateBoard(self, BoardState):
    """
    This function is used in slowpoke.py! The other code is for testing.
    """

    # convert board state into a numpy array
    xValues = np.array(BoardState, dtype=np.float32)

    # initiate board state.
    h1Sums = np.zeros(shape=[self.numHidden1], dtype=np.float32)
    oSums = np.zeros(shape=[self.numOutput], dtype=np.float32)

    # update the input nodes with values from the board state
    for i in range(self.numInputs):
      self.iNodes[i] = xValues[i]

    # multiply the input nodes with the weights to feed forward to the next layer

    for j in range(self.numHidden1):
      for i in range(self.numInputs):
        h1Sums[j] += self.iNodes[i] * self.ihWeights[i][j]

    for j in range(self.numHidden1):
      h1Sums[j] += self.h1Biases[j]
    
    # print("\n pre-tanh activation hidden node values: ")
    # showVector(h1Sums, 4)

    for j in range(self.numHidden1):
      self.h1Nodes[j] = self.sigmoid(h1Sums[j])
    
    # print("\n after activation hidden node values: ")
    # showVector(self.h1Nodes, 4)

    for k in range(self.numOutput):
      for j in range(self.numHidden1):
        oSums[k] += self.h1Nodes[j] * self.hoWeights[j][k]

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
    """
    Function to softmax output values.
    """
    result = np.zeros(shape=[len(oSums)], dtype=np.float32)
    m = max(oSums)
    divisor = 0.0
    for k in range(len(oSums)):
       divisor += math.exp(oSums[k] - m)
    for k in range(len(result)):
      result[k] =  math.exp(oSums[k] - m) / divisor
    return result
  
  @staticmethod
  def totalWeights(nInput, nHidden1, nHidden2, nOutput):
    # add weights from the hidden and output layers
    tw = (nInput * nHidden1) + (nHidden1 * nHidden2) + (nHidden2 * nOutput)
    # add bias values (because they're weights too)
    tw += (nHidden1 + nHidden2 + nOutput)
    # print("Total Weights:", tw)
    return tw

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
  import datetime
  startTime = datetime.datetime.now()
  for i in range(0,10000):
    yValues = nn.computeOutputs(xValues)
  print(datetime.datetime.now() - startTime)
  # print("\nOutput values are: ")
  # showVector(yValues, 4)

  # print("\nEnd demo \n")

# end script
