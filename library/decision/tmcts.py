# We arbitrarily defined the value of a winning board as +1.0 and a losing board as −1.0. All other boards would receive values between −1.0 and +1.0, with a neural network favouring boards with higher values.

minimax_win = 1
minimax_lose = -minimax_win
minimax_draw = 0
minimax_empty = -1

import random
import numpy as np

try:
  import mlx.core as mx
  MLX_AVAILABLE = True
except ImportError:
  mx = None
  MLX_AVAILABLE = False

# Special marker for terminal values (negative index indicates terminal state)
TERMINAL_VALUE_MARKER = -1

class TMCTS:

  def __init__(self, ply, evaluator, debug=False, batch_size=512):
    self.ply = ply
    self.evaluator = evaluator
    self.baseRound = 300
    self.debug = debug
    self.batch_size = batch_size  # Accumulate this many positions before batch eval
    # Check if evaluator has MLX-capable neural network
    self.nn = getattr(evaluator, 'nn', None)
    self.use_mlx = (hasattr(evaluator, 'nn') and 
                    evaluator.nn is not None and 
                    getattr(evaluator.nn, '_use_mlx', False))
    # Batch accumulation state for cross-tree batching
    self._batch_positions = []
    self._position_to_result = {}  # Maps position index -> evaluation result
    # Counter for position indexing
    self._position_counter = 0
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
      
      # Clear batch accumulation for this decision
      self._batch_positions = []
      self._position_to_result = {}
      self._position_counter = 0
      
      # Track result info for each round: (result_or_pos_idx, move)
      self._round_results = []
      
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
        # start mcts - use batched tree search if MLX is available
        if self.use_mlx:
          result = self.treesearch_batch(B, ply, colour)
          self._round_results.append((result, random_move))
        else:
          result = self.treesearch(B, ply, colour)
          self.movesets[random_move]['chances'] += result
          self.movesets[random_move]['plays'] += 1
        B.pop_move()

      # After all rounds, flush batch and resolve all deferred evaluations
      if self.use_mlx and self._batch_positions:
        self.flush_batch()
        # Resolve all results
        for result, move in self._round_results:
          if isinstance(result, int) and result >= 0:
            # Position index - resolve from batch results
            value = self._position_to_result.get(result, 0.0)
          else:
            # Terminal value (float)
            value = float(result)
          self.movesets[move]['chances'] += value
          self.movesets[move]['plays'] += 1

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
    """Standard tree search with individual evaluations."""
    isOver = self.isOver(B, colour)
    if isOver[0]:
      return isOver[1]
    else:
      if ply < 1:
        # Support both callable evaluator and object with evaluate_board method
        if callable(self.evaluator):
          return self.evaluator(B, colour)
        else:
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
    
    Accumulates positions during tree traversal and evaluates in batches
    using MLX. Uses a deferred evaluation pattern where positions are
    collected, then batch-evaluated after the tree search completes.
    
    Returns: 
      - int (position index) for deferred neural network evaluations
      - float for terminal states (win/lose/draw)
    """
    isOver = self.isOver(B, colour)
    if isOver[0]:
      return float(isOver[1])
    
    if ply < 1:
      # Extract position and accumulate for batch evaluation
      pos = self._extract_position(B, colour)
      pos_idx = self._position_counter
      self._position_counter += 1
      self._batch_positions.append((pos_idx, pos))
      # Return position index as deferred marker
      return pos_idx
    
    moves = B.get_moves()
    move = random.choice(moves)
    B.push_move(move)
    
    isOver = self.isOver(B, colour)
    if isOver[0]:
      result = float(isOver[1])
      B.pop_move()
      return result
    
    moves = B.get_moves()
    if len(moves) > 0:
      move = random.choice(moves)
      B.push_move(move)
      result = self.treesearch_batch(B, ply-1, colour)
      B.pop_move()
      B.pop_move()  # Pop enemy move
      return result
    else:
      B.pop_move()  # Pop enemy move
      return 0.0

  def flush_batch(self):
    """Evaluate all accumulated positions in a single batch.
    
    Maps batch results back to individual positions using position indices.
    Stores results in _position_to_result for resolution.
    
    Returns: np.array with evaluation results
    """
    if not self._batch_positions:
      return np.array([0.0])
    
    # Sort by position index to maintain order
    sorted_positions = sorted(self._batch_positions, key=lambda x: x[0])
    indices = [p[0] for p in sorted_positions]
    position_arrays = [p[1] for p in sorted_positions]
    
    # Evaluate batch using MLX or fallback
    if self.use_mlx and self.nn is not None and hasattr(self.nn, 'compute_batch_mlx'):
      results = self.nn.compute_batch_mlx(position_arrays)
      # Convert mx.array to numpy if needed
      if hasattr(results, 'numpy'):
        results = np.array(results.numpy())
      elif not isinstance(results, np.ndarray):
        results = np.array([float(r) for r in results])
    else:
      # Fallback: evaluate individually using evaluator
      if callable(self.evaluator):
        results = np.array([self.evaluator(None, None) for _ in position_arrays], dtype=np.float32)
      else:
        results = np.array([self.evaluator.evaluate_board(None, None) for _ in position_arrays], dtype=np.float32)
    
    # Store results in lookup table for resolution
    for idx, result in zip(indices, results):
      self._position_to_result[idx] = float(result)
    
    # Clear positions
    self._batch_positions = []
    
    return results

  def _extract_position(self, B, colour):
    """Extract board position for neural network evaluation."""
    boardStatus = B.getBoardPosWeighted(colour, {
      "Black": 1, 
      "White": -1,
      "empty": 0,
      "blackKing": 1.5,
      "whiteKing": -1.5
    })
    
    # Handle both NeuralNetwork objects and other evaluators
    layer_size = None
    if self.nn is not None and hasattr(self.nn, 'layer_size'):
      layer_size = self.nn.layer_size[0]
    elif hasattr(self.evaluator, 'layer_size'):
      layer_size = self.evaluator.layer_size[0]
    
    if layer_size == 91:
      boardStatus = self.nn.subsquares(boardStatus) if self.nn else boardStatus
    
    return np.array(boardStatus, dtype=np.float32)

  def isOver(self, B, colour):
    if B.is_over():
      if B.winner != minimax_empty:
        if B.winner == colour:
          return (True, minimax_win)
        else:
          return (True, minimax_lose)
      else:
        return (True, minimax_draw)
    else:
      return (False, -1)