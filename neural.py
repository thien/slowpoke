import numpy as np
import random
import math

def showVector(v, dec):
  fmt = "%." + str(dec) + "f" # like %.4f
  for i in range(len(v)):
    x = v[i]
    if x >= 0.0: print(' ', end='')
    print(fmt % x + '  ', end='')

class NeuralNetwork:
  def __init__(self, layer_list=[32,40,10,1]):
    self.layer_size = layer_list
    self.NumberOfLayers = len(self.layer_size)
    self.NumberOfHiddenLayers = self.NumberOfLayers - 2
    self.layers = []
    self.weights = []
    self.biases = []
    self.lenCoefficents = 0
    self.rebuildCoefficents = None
    self.rnd = np.random.seed()
    # initiate layers
    self.initiateLayers()
    self.initiateWeights()
    self.initiateBiases()

  def initiateLayers(self):
    for i in self.layer_size:
      nodes = np.zeros(shape=[i], dtype=np.float32)
      self.layers.append(nodes)
  
  def initiateWeights(self):
    for i in range(self.NumberOfLayers-1):
      inputNodes = self.layer_size[i]
      outputNodes = self.layer_size[i+1]
      # increment the number of coefficents
      self.lenCoefficents += inputNodes * outputNodes
      weights = np.random.random_sample([inputNodes,outputNodes])
      weights = self.normaliseVectors(weights)
      self.weights.append(weights)
  
  def initiateBiases(self):
    for i in range(self.NumberOfLayers-1):
      biasNodes = self.layer_size[i+1]
      self.lenCoefficents += biasNodes
      biases = np.random.random_sample(biasNodes)
      biases = self.normaliseVectors(biases)
      self.biases.append(biases)

  def getAllCoefficents(self):
    self.ravel = np.array([])
    # ravel weights
    for i in self.weights:
      ting = np.ravel(i)
      self.ravel = np.hstack((self.ravel,ting))
    # ravel biases
    for i in self.biases:
      ting = np.ravel(i)
      self.ravel = np.hstack((self.ravel,ting))
    return self.ravel

  def loadCoefficents(self, ravelled):
    if len(ravelled) != self.lenCoefficents:
      raise ValueError('The number of coefficents do not match.')
    # calculate number of weights to split array from
    totalNumWeights = 0
    for i in self.weights:
      totalNumWeights += i.shape[0] * i.shape[1]

    # rebuild weights
    weights = ravelled[:totalNumWeights]
    
    weight_inc = 0
    for i in range(len(self.weights)):
      # get the dimensions of i
      resolution = self.weights[i].shape[0] * self.weights[i].shape[1]
      sub_weight = weights[weight_inc:weight_inc+resolution]
      splitter = np.split(sub_weight, self.weights[i].shape[0])
      splitter = np.matrix(splitter)
      self.weights[i] = splitter
      weight_inc += resolution

    # rebuild biases
    biases = ravelled[totalNumWeights:]

    biases_inc = 0
    for i in range(len(self.biases)):
      resolution = self.biases[i].shape[0]
      sub_biases = biases[biases_inc:biases_inc+resolution]
      biases_inc += resolution
      self.biases[i] = sub_biases

    return True

  def compute(self, x):
    sums = []
    # initate placeholders to compute results.
    for i in range(self.NumberOfLayers-1):
      holder = np.zeros(shape=[self.layer_size[i+1]], dtype=np.float32)
      sums.append(holder)
    
    # assign input values to input layer
    self.layers[0] = x

    # compute neural network propagation for hidden layers
    for n in range(len(sums)):
      # compute weight addition
      for j in range(self.layer_size[n+1]):
        if n == 0:
          sums[n] = np.array([self.layers[n]]).dot(self.weights[n])
        else:
          sums[n] = sums[n-1].dot(self.weights[n])

      # check if output layer so we can feed the sum of the input layer directly
      if n == len(sums)-1:
        # on output layer
        sums[n] = sums[n] + np.sum(self.layers[0])
    
      # add biases
      sums[n] += self.biases[n][j]

      # perform nonlinear_function if we're not computing the final layer
      self.layers[n+1] = self.nonlinear_function(sums[n])

    flatten = self.layers[self.NumberOfLayers-1].flatten()
    if flatten.size == 1:
      return flatten[0]
    else:
      return self.layers[self.NumberOfLayers-1]

  @staticmethod
  def subsquares(x):
    # This is the fastest way (albeit also very ugly) i can think of to approach this problem
    return np.array([(x[0] + x[4] + x[5] + x[8]),(x[0] + x[1] + x[5] + x[8] + x[9]),(x[1] + x[5] + x[6] + x[9]),(x[1] + x[2] + x[6] + x[9] + x[10]),(x[2] + x[6] + x[7] + x[10]),(x[2] + x[3] + x[7] + x[10] + x[11]),(x[4] + x[5] + x[8] + x[12] + x[13]),(x[5] + x[8] + x[9] + x[13]),(x[5] + x[6] + x[9] + x[13] + x[14]),(x[6] + x[9] + x[10] + x[14]),(x[6] + x[7] + x[10] + x[14] + x[15]),(x[7] + x[10] + x[11] + x[15]),(x[8] + x[12] + x[13] + x[16]),(x[8] + x[9] + x[13] + x[16] + x[17]),(x[9] + x[13] + x[14] + x[17]),(x[9] + x[10] + x[14] + x[17] + x[18]),(x[10] + x[14] + x[15] + x[18]),(x[10] + x[11] + x[15] + x[18] + x[19]),(x[12] + x[13] + x[16] + x[20] + x[21]),(x[13] + x[16] + x[17] + x[21]),(x[13] + x[14] + x[17] + x[21] + x[22]),(x[14] + x[17] + x[18] + x[22]),(x[14] + x[15] + x[18] + x[22] + x[23]),(x[15] + x[18] + x[19] + x[23]),(x[16] + x[20] + x[21] + x[24]),(x[16] + x[17] + x[21] + x[24] + x[25]),(x[17] + x[21] + x[22] + x[25]),(x[17] + x[18] + x[22] + x[25] + x[26]),(x[18] + x[22] + x[23] + x[26]),(x[18] + x[19] + x[23] + x[26] + x[27]),(x[20] + x[21] + x[24] + x[28] + x[29]),(x[21] + x[24] + x[25] + x[29]),(x[21] + x[22] + x[25] + x[29] + x[30]),(x[22] + x[25] + x[26] + x[30]),(x[22] + x[23] + x[26] + x[30] + x[31]),(x[23] + x[26] + x[27] + x[31]),(x[0] + x[1] + x[4] + x[5] + x[8] + x[9] + x[12] + x[13]),(x[0] + x[1] + x[5] + x[6] + x[8] + x[9] + x[13] + x[14]),(x[1] + x[2] + x[5] + x[6] + x[9] + x[10] + x[13] + x[14]),(x[1] + x[2] + x[6] + x[7] + x[9] + x[10] + x[14] + x[15]),(x[2] + x[3] + x[6] + x[7] + x[10] + x[11] + x[14] + x[15]),(x[4] + x[5] + x[8] + x[9] + x[12] + x[13] + x[16] + x[17]),(x[5] + x[6] + x[8] + x[9] + x[13] + x[14] + x[16] + x[17]),(x[5] + x[6] + x[9] + x[10] + x[13] + x[14] + x[17] + x[18]),(x[6] + x[7] + x[9] + x[10] + x[14] + x[15] + x[17] + x[18]),(x[6] + x[7] + x[10] + x[11] + x[14] + x[15] + x[18] + x[19]),(x[8] + x[9] + x[12] + x[13] + x[16] + x[17] + x[20] + x[21]),(x[8] + x[9] + x[13] + x[14] + x[16] + x[17] + x[21] + x[22]),(x[9] + x[10] + x[13] + x[14] + x[17] + x[18] + x[21] + x[22]),(x[9] + x[10] + x[14] + x[15] + x[17] + x[18] + x[22] + x[23]),(x[10] + x[11] + x[14] + x[15] + x[18] + x[19] + x[22] + x[23]),(x[12] + x[13] + x[16] + x[17] + x[20] + x[21] + x[24] + x[25]),(x[13] + x[14] + x[16] + x[17] + x[21] + x[22] + x[24] + x[25]),(x[13] + x[14] + x[17] + x[18] + x[21] + x[22] + x[25] + x[26]),(x[14] + x[15] + x[17] + x[18] + x[22] + x[23] + x[25] + x[26]),(x[14] + x[15] + x[18] + x[19] + x[22] + x[23] + x[26] + x[27]),(x[16] + x[17] + x[20] + x[21] + x[24] + x[25] + x[28] + x[29]),(x[16] + x[17] + x[21] + x[22] + x[24] + x[25] + x[29] + x[30]),(x[17] + x[18] + x[21] + x[22] + x[25] + x[26] + x[29] + x[30]),(x[17] + x[18] + x[22] + x[23] + x[25] + x[26] + x[30] + x[31]),(x[18] + x[19] + x[22] + x[23] + x[26] + x[27] + x[30] + x[31]),(x[0] + x[1] + x[4] + x[5] + x[6] + x[8] + x[9] + x[12] + x[13] + x[14] + x[16] + x[17]),(x[0] + x[1] + x[2] + x[5] + x[6] + x[8] + x[9] + x[10] + x[13] + x[14] + x[16] + x[17] + x[18]),(x[1] + x[2] + x[5] + x[6] + x[7] + x[9] + x[10] + x[13] + x[14] + x[15] + x[17] + x[18]),(x[1] + x[2] + x[3] + x[6] + x[7] + x[9] + x[10] + x[11] + x[14] + x[15] + x[17] + x[18] + x[19]),(x[4] + x[5] + x[6] + x[8] + x[9] + x[12] + x[13] + x[14] + x[16] + x[17] + x[20] + x[21] + x[22]),(x[5] + x[6] + x[8] + x[9] + x[10] + x[13] + x[14] + x[16] + x[17] + x[18] + x[21] + x[22]),(x[5] + x[6] + x[7] + x[9] + x[10] + x[13] + x[14] + x[15] + x[17] + x[18] + x[21] + x[22] + x[23]),(x[6] + x[7] + x[9] + x[10] + x[11] + x[14] + x[15] + x[17] + x[18] + x[19] + x[22] + x[23]),(x[8] + x[9] + x[12] + x[13] + x[14] + x[16] + x[17] + x[20] + x[21] + x[22] + x[24] + x[25]),(x[8] + x[9] + x[10] + x[13] + x[14] + x[16] + x[17] + x[18] + x[21] + x[22] + x[24] + x[25] + x[26]),(x[9] + x[10] + x[13] + x[14] + x[15] + x[17] + x[18] + x[21] + x[22] + x[23] + x[25] + x[26]),(x[9] + x[10] + x[11] + x[14] + x[15] + x[17] + x[18] + x[19] + x[22] + x[23] + x[25] + x[26] + x[27]),(x[12] + x[13] + x[14] + x[16] + x[17] + x[20] + x[21] + x[22] + x[24] + x[25] + x[28] + x[29] + x[30]),(x[13] + x[14] + x[16] + x[17] + x[18] + x[21] + x[22] + x[24] + x[25] + x[26] + x[29] + x[30]),(x[13] + x[14] + x[15] + x[17] + x[18] + x[21] + x[22] + x[23] + x[25] + x[26] + x[29] + x[30] + x[31]),(x[14] + x[15] + x[17] + x[18] + x[19] + x[22] + x[23] + x[25] + x[26] + x[27] + x[30] + x[31]),(x[0] + x[1] + x[2] + x[4] + x[5] + x[6] + x[8] + x[9] + x[10] + x[12] + x[13] + x[14] + x[16] + x[17] + x[18] + x[20] + x[21] + x[22]),(x[0] + x[1] + x[2] + x[5] + x[6] + x[7] + x[8] + x[9] + x[10] + x[13] + x[14] + x[15] + x[16] + x[17] + x[18] + x[21] + x[22] + x[23]),(x[1] + x[2] + x[3] + x[5] + x[6] + x[7] + x[9] + x[10] + x[11] + x[13] + x[14] + x[15] + x[17] + x[18] + x[19] + x[21] + x[22] + x[23]),(x[4] + x[5] + x[6] + x[8] + x[9] + x[10] + x[12] + x[13] + x[14] + x[16] + x[17] + x[18] + x[20] + x[21] + x[22] + x[24] + x[25] + x[26]),(x[5] + x[6] + x[7] + x[8] + x[9] + x[10] + x[13] + x[14] + x[15] + x[16] + x[17] + x[18] + x[21] + x[22] + x[23] + x[24] + x[25] + x[26]),(x[5] + x[6] + x[7] + x[9] + x[10] + x[11] + x[13] + x[14] + x[15] + x[17] + x[18] + x[19] + x[21] + x[22] + x[23] + x[25] + x[26] + x[27]),(x[8] + x[9] + x[10] + x[12] + x[13] + x[14] + x[16] + x[17] + x[18] + x[20] + x[21] + x[22] + x[24] + x[25] + x[26] + x[28] + x[29] + x[30]),(x[8] + x[9] + x[10] + x[13] + x[14] + x[15] + x[16] + x[17] + x[18] + x[21] + x[22] + x[23] + x[24] + x[25] + x[26] + x[29] + x[30] + x[31]),(x[9] + x[10] + x[11] + x[13] + x[14] + x[15] + x[17] + x[18] + x[19] + x[21] + x[22] + x[23] + x[25] + x[26] + x[27] + x[29] + x[30] + x[31]),(x[0] + x[1] + x[2] + x[4] + x[5] + x[6] + x[7] + x[8] + x[9] + x[10] + x[12] + x[13] + x[14] + x[15] + x[16] + x[17] + x[18] + x[20] + x[21] + x[22] + x[23] + x[24] + x[25] + x[26]),(x[0] + x[1] + x[2] + x[3] + x[5] + x[6] + x[7] + x[8] + x[9] + x[10] + x[11] + x[13] + x[14] + x[15] + x[16] + x[17] + x[18] + x[19] + x[21] + x[22] + x[23] + x[24] + x[25] + x[26] + x[27]),(x[4] + x[5] + x[6] + x[7] + x[8] + x[9] + x[10] + x[12] + x[13] + x[14] + x[15] + x[16] + x[17] + x[18] + x[20] + x[21] + x[22] + x[23] + x[24] + x[25] + x[26] + x[28] + x[29] + x[30] + x[31]),(x[5] + x[6] + x[7] + x[8] + x[9] + x[10] + x[11] + x[13] + x[14] + x[15] + x[16] + x[17] + x[18] + x[19] + x[21] + x[22] + x[23] + x[24] + x[25] + x[26] + x[27] + x[29] + x[30] + x[31]),(x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + x[8] + x[9] + x[10] + x[11] + x[12] + x[13] + x[14] + x[15] + x[16] + x[17] + x[18] + x[19] + x[20] + x[21] + x[22] + x[23] + x[24] + x[25] + x[26] + x[27] + x[28] + x[29] + x[30] + x[31])])

  @staticmethod
  def normaliseVectors(vector):
    # normalise to a range from -0.2 to 0.2
    return (vector-0.5) * 0.4

  @staticmethod
  def nonlinear_function(val):
    return np.tanh(val)

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

if __name__ == "__main__":
  inputs = [91,40,10,1]
  nn = NeuralNetwork(inputs)
  # Insert checkerboard.
  x = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1], dtype=np.float32)
  
  import datetime
  start = datetime.datetime.now().timestamp()
  x = nn.subsquares(x)
  print(np.sum(x))
  end = datetime.datetime.now().timestamp() - start
  mu = datetime.datetime.now().timestamp()
  print("TIME GENUGL:",end)

  # x = np.random.random_sample(32)
  # Run Neural Network
  # import datetime
  # startTime = datetime.datetime.now()
  # for i in range(0,10000):
  yValues = nn.compute(x)
  print("Probability:",yValues)
  print("TIME TO RUN:",datetime.datetime.now().timestamp() - start)
  # print("\nOutput values are: ")
  # showVector(yValues, 4)