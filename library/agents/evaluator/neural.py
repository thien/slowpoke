import numpy as np
import random
import math

try:
  import agents.evaluator.subsquares as subsquares
except ImportError:
  from library.agents.evaluator import subsquares

def showVector(v, dec):
  fmt = "%." + str(dec) + "f" # like %.4f
  for i in range(len(v)):
    x = v[i]
    if x >= 0.0: print(' ', end='')
    print(fmt % x + '  ', end='')

class NeuralNetwork:
  __slots__ = ['layer_size', 'NumberOfLayers', 'NumberOfHiddenLayers', 'layers', 
               'weights', 'biases', 'lenCoefficents', 'rebuildCoefficents', 'rnd', 'ravel']
  
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
    """Optimized: collect all weights and biases in one pass."""
    arrays = []
    for w in self.weights:
      arrays.append(np.ravel(w))
    for b in self.biases:
      arrays.append(np.ravel(b))
    return np.concatenate(arrays)
  
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
      # Reshape to (input_nodes, output_nodes) - store as ndarray, not matrix
      self.weights[i] = sub_weight.reshape(self.weights[i].shape).astype(np.float32)
      weight_inc += resolution
    
    # rebuild biases
    biases = ravelled[totalNumWeights:]
    
    biases_inc = 0
    for i in range(len(self.biases)):
      resolution = self.biases[i].shape[0]
      sub_biases = biases[biases_inc:biases_inc+resolution]
      biases_inc += resolution
      self.biases[i] = sub_biases.astype(np.float32)
    
    return True

  def compute(self, x):
    """
    Optimized forward pass through the neural network.
    Fully vectorized - no loops over neurons.
    """
    current = x
    
    # Forward pass through all hidden layers
    for n in range(self.NumberOfLayers - 2):
      # Vectorized: matrix multiply + bias in one step
      current = self.weights[n].T.dot(current) + self.biases[n]
      current = self.nonlinear_function(current)
    
    # Final layer
    current = self.weights[-1].T.dot(current) + self.biases[-1]
    
    # Add input contribution for terminal layer (special heuristic)
    if x.size == 91:
      current = current + x[-1] * 32
    else:
      current = current + np.sum(x)
    
    return float(current[0]) if current.size == 1 else current

  @staticmethod
  def subsquares(x):
    """
    Calculates 3x3 to 8x8 set of subsquares on the checkerboard.
    """
    return subsquares.subsquares(x)

  @staticmethod
  def normaliseVectors(vector):
    # normalise to a range from -0.2 to 0.2
    return (vector-0.5) * 0.4
    # normalise to a range from -1 to 1
    # return (vector-0.5) * 2

  def nonlinear_function(self,val):
    # tanh/sigmoid
    return self.tanh(val)
    # return self.crelu(val)
    # return self.relu(val)

  @staticmethod
  def tanh(val):
    return np.tanh(val)

  @staticmethod
  def relu(x):
    # rectifier method; it turns out that this is not very effective at all.
    x[x<0] =0
    return x

  @staticmethod
  def crelu(x):
    # linear cap from -1
    x[x<-1] =-1
    return x


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

  # Insert checkerboard.
  x = np.array([1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1], dtype=np.float32)
  

  # standard neural network
  inputs = [32,40,10,1]
  nn = NeuralNetwork(inputs)

  # subsquare neural network
  subsq = [91,40,10,1]
  nn2 = NeuralNetwork(subsq)


  import datetime


  # print("Regular Neural Network")
  start = datetime.datetime.now().timestamp()

  yValues = nn.compute(x)
  print("RNN:",yValues)
  end = datetime.datetime.now().timestamp() - start
  # print("RNN Time:",end)

  x = nn.subsquares(x)

  # print("Subsquare Processed Neural Network")
  mu = datetime.datetime.now().timestamp()

  # print(x.size)
  yValues = nn2.compute(x)
  print("SNN:",yValues)
  end2 = datetime.datetime.now().timestamp() - start
  # print("SNN Time:",end2)

  # print("\nOutput values are: ")
  # showVector(yValues, 4)

  print("Time Multiplier:",end2/end)