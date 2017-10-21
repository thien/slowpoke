"""
Alakazam is a draughts AI based on a feed forward
neural network chucked alongside a genetic algorithm.
"""
import random
import os
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

def move_function(board):
  # first we need to establish whether we're white or black.
  print('dank')
  return random.choice(board.get_moves())

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