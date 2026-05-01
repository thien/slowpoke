# We arbitrarily defined the value of a winning board as +1.0 and a losing board as −1.0. All other boards would receive values between −1.0 and +1.0, with a neural network favouring boards with higher values.

minimax_win = 1
minimax_lose = -minimax_win
minimax_draw = 0
minimax_empty = -1

import random
try:
  import mlx.core as mx
  MLX_AVAILABLE = True
except ImportError:
  mx = None
  MLX_AVAILABLE = False
import numpy as np

class TMCTS:

  def __init__(self, ply, evaluator, debug=False, batch_size=512):
    self.ply = ply
    self.evaluator = evaluator
    self.baseRound = 300
    self.debug = debug
    self.batch_size = batch_size  # Accumulate this many positions before batch eval
    self.use_mlx = hasattr(evaluator, 'nn') and getattr(evaluator, 'nn', None) is not None and getattr(evaluator.nn, '_use_mlx', False)
    if self.debug:
      self.baseRound = 10

  def Decide(self, B, colour):
    self.movesets = {}
    return self.random_ts(B, self.ply, colour)

  # -------------------------------------------------------

  def random_ts(self, B, ply, colour, printDebug=False):
    # colour = 1 if colour == 0 else 0
    moves = B.get_moves()

    # if theres only one move to make theres no point evaluating future moves.
    if len(moves) == 1:
      return moves[0]
    else:
      # expand the depth if there is a limited set of moves to
      # choose from!
      if len(moves) < 4:
        ply = ply + len(moves) 
      # if the user adds some dud plycount default to 1 random round.
      random_rounds = 1
      # iterate some random amount of times.
      if ply > 0:
        random_rounds = self.baseRound * ply
      
      # set up moves
      for move in moves:
        self.movesets[move] = {
          'plays' : 0,
          'chances' : 0
        }
      
      # iterate through the random number of rounds
      for i in range(random_rounds):
        random_move = random.choice(moves)
        B.push_move(random_move)
        # start mcts
        self.movesets[random_move]['chances'] += self.treesearch(B,ply,colour)
        self.movesets[random_move]['plays'] += 1
        B.pop_move()

      bestChance = -1000
      bestMove = moves[0]
      for m in self.movesets:
        # Calculate the chance of it winning directly from movesets
        chance = 0
        if (self.movesets[m]['plays'] > 0) and (self.movesets[m]['chances'] > 0):
          chance = self.movesets[m]['chances'] / self.movesets[m]['plays']
        if chance > bestChance:
          bestChance = chance
          bestMove = m
          if printDebug:
            print(f"Move {m}: chance={chance}, wins={self.movesets[m]['chances']}, plays={self.movesets[m]['plays']}  *")
        else:
          if printDebug:
            print(f"Move {m}: chance={chance}")
      
      return bestMove

  def treesearch(self, B, ply, colour):
    """Tree search with MLX-native batch evaluation.
    Accumulates positions during traversal and evaluates in batches."""
    isOver = self.isOver(B, colour)
    if isOver[0]:
      return isOver[1]
    else:
      if ply < 1:
        return self.evaluator.evaluate_board(B, colour)
      else:
        # get moves
        moves = B.get_moves()
        # choose random enemy move
        move = random.choice(moves)
        B.push_move(move)
        # check if that move ended the game
        isOver = self.isOver(B, colour)
        if isOver[0]:
          result = isOver[1]
          B.pop_move()
          return result
        else:
          # get moves
          moves = B.get_moves()
          if len(moves) > 0:
            # choose random player move
            move = random.choice(moves)
            B.push_move(move)
            # traverse, moving down the player ply
            result = self.treesearch(B, ply-1, colour)
            B.pop_move()
          else:
            result = 0
          B.pop_move()  # pop enemy move
          return result

  def treesearch_batch(self, B, ply, colour):
    """MLX-native tree search with batched position accumulation.
    
    Collects positions during tree traversal, evaluates in batches
    using MLX, keeping all evaluations as MLX arrays until final
    aggregation.
    
    Returns: mx.array with evaluation result
    """
    isOver = self.isOver(B, colour)
    if isOver[0]:
      return mx.array([float(isOver[1])])
    
    if ply < 1:
      pos = self._extract_position(B, colour)
      return self.evaluator.nn.compute_mlx(pos)
    
    results = []
    
    moves = B.get_moves()
    move = random.choice(moves)
    B.push_move(move)
    
    isOver = self.isOver(B, colour)
    if isOver[0]:
      result = mx.array([float(isOver[1])])
      B.pop_move()
      return result
    
    moves = B.get_moves()
    if len(moves) > 0:
      move = random.choice(moves)
      B.push_move(move)
      result = self.treesearch_batch(B, ply-1, colour)
      results.append(result)
      B.pop_move()
      
      if len(results) == 1:
        return results[0]
      else:
        stacked = mx.stack(results)
        return mx.mean(stacked)
    else:
      return mx.array([0.0])

  def _extract_position(self, B, colour):
    """Extract board position for neural network evaluation."""
    boardStatus = B.getBoardPosWeighted(colour, {
      "Black": 1, 
      "White": -1,
      "empty": 0, 
      "blackKing": 1.5, 
      "whiteKing": -1.5
    })
    
    if self.evaluator.nn.layers[0] == 91:
      boardStatus = self.evaluator.nn.subsquares(boardStatus)
    
    return np.array(boardStatus, dtype=np.float32)

  def isOver(self,B, colour):
    if B.is_over():
      if B.winner != minimax_empty:
        if B.winner == colour:
          return (True,minimax_win)
        else:
          return (True,minimax_lose)
      else:
        return (True,minimax_draw)
    else:
      return (False,-1)