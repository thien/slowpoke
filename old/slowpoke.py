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
                                               "' mh

Slowpoke is a draughts AI based on a feed forward convolutional
neural network chucked alongside a genetic algorithm.

Some of the code is based on Sean K's Neural Network code, of
which the source code can be found here:
http://seank.50webs.com.

"""


# Iimports
import math
import pickle
import threading
# everyone loves a good debugger.
import pdb
import random


def move_function(board):
    return random.choice(board.get_moves())

def sigmoid(val):
  # tanh function with the range of [-1,1]
  raw = math.tanh(val)
  if raw >= 0:
    return min(raw, 1)
  else:
    return max(raw, -1)

class Neuron:
  # Class to define a neuron.
  def __init__(self, weights, bias):
    """
    Creates a new Neuron.

    Args:
      weights:
        an interable of floats that represent the weights
        of the neuron.
      bias:
        a float that represents the bias of the neuron.
    """

    if not weights:
      pdb.set_trace()
    self.weights = weights
    self.bias = bias

  def evaluate(self, inputs):
    """
    Evaluates an output based on a list of inputs.

    Arguments:
      inputs:
      a list of inputs where the size if equal to the
      size of the weights

    Return:
      a float corresponding to the output of the neuron.
    """
    raw = sum([i*j for i,j in zip(self.weights, inputs)])
    biased_evaluation = raw - self.bias
    return sigmoid(biased_evaluation)

  def combine(self, neuron, f):
    """
    Takes a neuron and a function f, and creates another neuron
    where all the weights have been combined according to the
    rules of f.

    neuron:
      A neuron to combine with self.
    f:
      A binary function that takes two weights (floats) and
      returns a new weight.
    """
    weights = [f(a,b) for a,b in zip(self.weights, neuron.weights)]
    bias = fn(self.bias, neuron.bias)
    return Neuron(weights, bias)

  def map(self, f):
    """
    Creates a new neuron where the weights have been changed
    according to f.

    f:
      an unary operator that returns a new float, given a
      double.

    This function returns a new neuron where its weights have
    been altered by fn.
    """
    weights = [f(w) for w in self.weights]
    bias = fn(self.bias)
    return Neuron(weights, bias)

  count = property(lambda self: len(self.weights + 1))

class Layer:
  """
  This layer class represents a single layer of neurons in
  the ANN.
  """

  def __init__(self, neurons):
    """
    init with a list of neurons, where the number of weights
    are exactly equal.
    """
    self.neurons = neurons

  def evaluate(self, inputs):
    """
    This function evaluates an output list based on a list of
    inputs.

    Arguments:
      inputs:
        a list of inputs where the size is equal to the size
        of the individual neurons in the layer.

    Returns:
      a list of floats corresponding to the outputs of the network.
    """
    return [n.evaluate(inputs) for n in self.neurons]

  def combine(self, layer, f):
    """
    Combine takes a layer, and a function f, and creates another
    one where the weights have been combined according to the rules
    of f.

    Arguments:
      layer:
        A layer to combine with `self`.
      f:
        A binary function; it takes two weights (floats) and returns
        a new weight.
    """
    zipped = zip(self.neurons, layer.neurons)
    neurons = [a.combine(b, fn) for a,b in zipped]
    return Layer(neurons)

  def map(self, f):
    """
    Creates a new layer where all the weights have been changed
    according to f.

    Arguments:
      f:
        an unary operator that returns a new float, given a double.
    """
    return Layer([n.map(f) for n in self.neurons])

  count = property(lambda self: sum([n.count for n in self.neurons]))

class FeedForwardNetwork:
  """
  This class represents a feed forward network.
  - The first element is where the inputs are initially fed into.
  - The last element is a layer that has a single output neuron.
  - It's important that the first layer has a number of weights
    equal to the number of inputs.
  - Similarly, the second layer needs a number of weights equal
    to the number of neurons in layer 1, and so on.

  """

  def __init__(self, layers):
    """
    Here we create a new feed-forward network.
    The layers parameter is a list of layers.

    Arguments:
      layers:
        a list of layers, where the last layer must consist of a 
        single neuron.
    """
    self.layers = layers


  def evaluate(self, inputs):
    """
    Evaluate a list of inputs and returns a float.
    """
    for i in self.layers:
      outputs = i.evaluate(inputs)
    return outputs[0]

  def combine(self, network, f):
    """
    This takes another network, and a function f, and creates
    another layer where all the weight have been combined
    according to the rules of f.
    The network must have the same topology as self.

    Arguments:
      layer:
        a layer to combine with self.
      f:
        a binary function that takes two weights (floats)
        and returns a new weight.
    Return:
      a new network where the weights have been changed according
      to f, self, and network.
    """
    zipped = zip(self.layers, network.layers)
    layers = [a.combine(b,f) for a, b in zipped]
  return FeedForwardNetwork(layers)

  def map(self, f):
    """
    Creates a new network where the weights have been changed 
    according to f.

    Arguments:
      f:
        an unary operator that returns a new float, given
        a double.
    Return:
      a new network where its weights have been altered by f.
    """

    return FeedForwardNetwork([layer.map(f) for layer in self.layers])

  count = property(lambda self: sum([a.count for a in self.layers]))

class BlondieNetwork:
  """
  This represents a network inspired by Blondie24.
  """

  def __init__(self):
    self._net = self._create_network(lambda: random.uniform(-.2, .2))
    self._sigma = self._create_network(lambda: .5)
    self._score = 0
    self.table = {}

  def add_win(self):
    """
    adds a win to the score.
    """
    self._score += 1

  def add_loss(self):
    """
    adds a loss to the score.
    """
    self._score -= 2

  def reset(self):
    self._score = 0
    self._table = {}

  @staticmethod
  def _create_network(f):
    def create_layer(neurons, weights):
      weights = [[f() for i in range(weights)] for j in range(neurons)]
      return Layer([Neuron(w, f()) for w in weights])
    layers = [create_layer(40,32)]
    layers.append(create_layer(10, 40))
    layers.append(create_layer(1, 10))
    return FeedForwardNetwork(layers)

  def _evaluation_function(self):
    """
    This method returns a method that takes a checkers board
    as a parameter, and returns the network's evaluation of
    the board.
    """    

    def eval_board(board, colour):
      return self._net.evaluate(board.hash)
    return eval_board

  def save(self, file):
    f = open(file, 'wb')
    pickle.dump(self, f)
    f.close()

  @staticmethod
  def load(file):
    f = open(file, 'rb')
    b = pickle.load(f)
    f.close()
    return b

  def breed(self):
    """
    Returns a new BlondieNetwork according to the Blondie
    algorithm 
    """
    tau = 1 / math.sqrt(2 * math.sqrt(self._net.count))

    def sigma_map(weight):
      return weight * math.e ** (tau * random.gauss(0, 1))
    def neuron_map(a, b):
      return a + b * random.gauss(0, 1)

    new = BlondieNetwork()
    new._sigma = self._sigma.map(sigma_map)
    new._net = self._net.combine(new._sigma, neuron_map)
    return new

  score = property(lambda self: self._score)
  evaluator = property(_evaluation_function)

class Slowpoke:
  """
  Slowpoke AI
  """
  
  def __init__(self,colour,evaluator,depth,table):
    """
    Creates a new Slowpoke

    Arguments:
      evaluator:
        A function that takes a board and colour as an argument;
        and returns a value from [-1,1] where -1 means that the player
        has lost and 1 means the player has won.
      Depth:
        The depth of the alpha-beta seatch.
    """
    self._evaluator = evaluator
    self._depth = depth
    self._board = Board()
    # initiate the board.
    self._board.start()
    self._colour = colour
    self._jumper = None
    self._table = Table

  def oppoment_move(self, move):
    """
    This method acknowledges a move played by the oppoment.
    """
    self._board = move

  def choose_move(self,screen):
    """
    Chooses and returns a move.

    Arguments:
      screen:
        where to send the move to.
    Returns:
      a board object that represents selfs last move.
    """
    return False

  def _select_move(self):
    """
    Internal logic to find available moves.
    """
    # check if we're jumping or that the board has jumps
      # then you gotta select those jumps.
    # see the list of possible moves
      # iterate through the list of moves
        # if this move is a winning move, choose it
    # iteratively_search those moves.
    # get the index of those moves.
    # assign the moves somewhere.

  def _select_jumps(self):
    """
    Chooses the best jump available to the current board.
    """

    def assign_jump(inp):
      """
      Assigns a new board, and possibly a forced jump position
      """
      (jump,pos) = inp
      