import numpy as np
import random
import math

def showVector(v, dec):
  fmt = "%." + str(dec) + "f" # like %.4f
  for i in range(len(v)):
    x = v[i]
    if x >= 0.0: print(' ', end='')
    print(fmt % x + '  ', end='')
  print('')

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
      # print
      # grab section
      sub_weight = weights[weight_inc:weight_inc+resolution]
  
      # assign to weights
      splitter = np.split(sub_weight, self.weights[i].shape[0])

      splitter = np.matrix(splitter)
      self.weights[i] = splitter

      # increment
      weight_inc += resolution

    # rebuild biases
    biases = ravelled[totalNumWeights:]

    biases_inc = 0
    for i in range(len(self.biases)):
      # get the dimensions of i
      resolution = self.biases[i].shape[0]
      sub_biases = biases[biases_inc:biases_inc+resolution]
      biases_inc += resolution
      # fold first half of micro_split.
      self.biases[i] = sub_biases

    # print(totalNumWeights+totalNumBiases)
    return True

  def compute(self, xValues):
    sums = []
    # initate placeholders to compute results.
    for i in range(self.NumberOfLayers-1):
      holder = np.zeros(shape=[self.layer_size[i+1]], dtype=np.float32)
      sums.append(holder)
    
    # assign input values to input layer
    self.layers[0] = xValues

    # compute neural network propagation for hidden layers
    for n in range(len(sums)):
      # compute weight addition
      for j in range(self.layer_size[n+1]):
        if n == 0:
          sums[n] = np.array([self.layers[n]]).dot(self.weights[n])
        else:
          sums[n] = sums[n-1].dot(self.weights[n])
      # add biases
      sums[n] += self.biases[n][j]

      # perform nonlinear_function if we're not computing the final layer
      self.layers[n+1] = self.nonlinear_function(sums[n])

    return self.layers[self.NumberOfLayers-1]

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

  inputs = [32,40,10,1]

  nn = NeuralNetwork(inputs)

  # Insert checkerboard.
  xValues = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1.1, 1.1, 0, 0], dtype=np.float32)
  # xValues = np.random.random_sample(32)
  # Run Neural Network
  # import datetime
  # startTime = datetime.datetime.now()
  # for i in range(0,10000):
  yValues = nn.compute(xValues)
  print("\nOutput values are: ")
  showVector(yValues, 4)
  print("-------------------")
  # print(datetime.datetime.now() - startTime)
  cof = nn.getAllCoefficents()
  print(nn.loadCoefficents(cof))
  yValues = nn.compute(xValues)
  print("\nOutput values are: ")
  showVector(yValues, 4)
  print("-------------------")
  # print("\nOutput values are: ")
  # showVector(yValues, 4)

  # print("\nEnd demo \n")

# end script