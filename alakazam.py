"""
Alakazam is a draughts AI based on a feed forward
neural network chucked alongside a genetic algorithm.
"""
import random
import os
import numpy as np

# import Tensorflow!
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


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

# --------------------------------------------------------------------

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

  # now that we have our colour, we'll need to get the 
  # current pieces of the board, manipulated in a way so that
  # it can be shoved into the NN.
  print(getColourString())
  boardStatus = board.getBoardPosWeighted(config['colour'], customWeights)
  print(boardStatus)


  return random.choice(board.get_moves())


# --------------------------------------------------------------------
"""
Neural Network Stuff!
This is one of the exciting parts of the project.
"""

# def leaky_relu(z, alpha=0.01):
#     return np.maximum(alpha*z, z)

# n_inputs = 28 * 28  # MNIST
# n_hidden1 = 300
# n_hidden2 = 100
# n_outputs = 10

# X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
# y = tf.placeholder(tf.int64, shape=(None), name="y")

# with tf.name_scope("dnn"):
#     hidden1 = tf.layers.dense(X, n_hidden1, activation=leaky_relu, name="hidden1")
#     hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=leaky_relu, name="hidden2")
#     logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

# with tf.name_scope("loss"):
#     xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
#     loss = tf.reduce_mean(xentropy, name="loss")