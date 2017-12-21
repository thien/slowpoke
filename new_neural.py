import numpy as np
import random
import math

class NeuralNetwork:
  def __init__(self, layer_list):
    self.layer_size = layer_list
    self.NumberOfLayers = len(self.layer_size)
    self.NumberOfHiddenLayers = self.NumberOfLayers - 2
    self.layers = []
    self.weights = []
    self.biases = []
    self.lenCoefficents = 0
    self.rnd = random.Random(0)
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
    return False

  def compute(self, xValues):
    sums = []
    # initate placeholders to compute results.
    for i in range(self.NumberOfLayers-1):
      holder = np.zeros(shape=[self.layer_size[i+1]], dtype=np.float32)
      sums.append(holder)
    
    # assign input values to input layer
    self.layers[0] = xValues

# ----------------------------

    # compute neural network propagation for hidden layers
    # print(len(sums))
    for i in range(len(sums)):
      print("working on layer", i+1)

      # !!!!!!!!!!!
      # compute weight addition
      print(self.layers[i].size)
      print(self.weights[i].size)
      for j in range(self.layer_size[i+1]):
        for k in range(self.layer_size[i]):
          sums[i] = self.layers[i] * self.weights[i][k][j]

      # add biases
      sums[i] += self.biases[i][j]

      # perform nonlinear_function if we're not computing the final layer
      if len(sums) > i:
        self.layers[i+1] = self.nonlinear_function(sums[i])
      print("finished working on layer", i+1)
# ----------------------------

    # check if output layer size is greater than 1
    outputLayerNum = self.NumberOfLayers-1
    outputLayer = layers[outputLayerNum]
    if np.prod(outputLayer.shape) > 1:
      softOut = self.softmax(outputLayer)
      for k in range(self.layer_size[outputLayerNum]):
        self.layers[outputLayerNum] = softOut[k]
        result = np.zeros(shape=outputLayerNum, dtype=np.float32)
        for k in range(self.numOutput):
          result[k] = outputLayer[k]
      return result
    else:
      return layers[self.NumberOfLayers-1]

  @staticmethod
  def normaliseVectors(vector):
    # normalise to a range from -0.2 to 0.2
    return (vector-0.5) * 0.4

  @staticmethod
  def nonlinear_function(val):
    # tanh function with the range of [-1,1]
    raw = np.tanh(val)

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
  # print("Creating a %d-%d-%d-%d neural network " % (numInput, numHidden1, numHidden2, numOutput) )
  nn = NeuralNetwork(inputs)

  # Insert checkerboard.
  xValues = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1.1, 1.1, 0, 0], dtype=np.float32)
  # Run Neural Network
  yValues = nn.compute(xValues)
  print("\nOutput values are: ")
  showVector(yValues, 4)

  print("\nEnd demo \n")

# end script
