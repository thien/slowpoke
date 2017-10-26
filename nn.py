# nn_io.py
# Python 3.x

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

def main():
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
   
if __name__ == "__main__":
  main()

# end script
