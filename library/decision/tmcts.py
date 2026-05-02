"""TMCTS — tree-based Monte Carlo tree search with UCB1 and batched NN eval."""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
  import mlx.core as mx
  MLX_AVAILABLE = True
except ImportError:
  mx = None
  MLX_AVAILABLE = False

minimax_win = 1
minimax_lose = -minimax_win
minimax_draw = 0
minimax_empty = -1

TERMINAL_VALUE_MARKER = -1


class TMCTS:
  """Tree-based Monte Carlo Tree Search with UCB1 selection and batched NN evaluation."""

  def __init__(self, ply: int, evaluator: Any, debug: bool = False, batch_size: int = 512) -> None:
    self.ply = ply
    self.evaluator = evaluator
    self.baseRound = 300
    self.debug = debug
    self.batch_size = batch_size  # Accumulate this many positions before batch eval
    self.ucb_exploration = 1.4  # C constant for UCB1 (sqrt(2) ≈ 1.414 is standard)
    # Progressive narrowing with Gumbel-Top-K
    self.progressive_narrowing = True   # Enable Gumbel-Top-K narrowing
    self.progressive_narrowing_k = 5    # Keep top K moves after Gumbel-Top-K
    self.gumbel_temperature = 0.5       # Higher = more exploration in narrowing
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
    
    # --- Tree reuse across turns: persistent node cache ---
    # Maps board position hash (bytes) -> MCTS value (cached evaluation result).
    # Survives across Decide() calls so evaluations from previous turns'
    # internal nodes are reused when the same position is encountered again.
    # Unlike a leaf-only NN cache, this stores the full MCTS playout average,
    # which is more accurate and captures transpositions within a single turn too.
    self._node_cache = {}
    self._max_cache_size = 50000
    self._cache_hits = 0  # Counts cache hits for benchmarking
    
    if self.debug:
      self.baseRound = 10

  def _select_move_ucb1(self, moves: List[int], C: Optional[float] = None) -> int:
    """Select move using UCB1 (Upper Confidence Bound).

    Balances exploration (trying under-explored moves) with
    exploitation (choosing moves with high win rates).

    UCB1 formula: score = win_rate + C * sqrt(ln(total_visits + 1) / visits)

    - Moves with 0 visits get infinite score (always explored first)
    - As visits increase, exploration bonus shrinks
    - As total_visits grows, exploration bonus grows slowly (sublinear)
    """
    if C is None:
      C = self.ucb_exploration
    total_visits = sum(self.movesets[m]['plays'] for m in moves)

    # Always explore unvisited moves first (infinite UCB score)
    unvisited = [m for m in moves if self.movesets[m]['plays'] == 0]
    if unvisited:
      return random.choice(unvisited)

    best_score = -float('inf')
    best_move = moves[0]
    for m in moves:
      wins = self.movesets[m]['chances']
      visits = self.movesets[m]['plays']
      if visits == 0:
        return m  # Safety: give infinite UCB score to unvisited
      win_rate = wins / visits
      exploration_bonus = C * math.sqrt(math.log(total_visits + 1) / visits)
      score = win_rate + exploration_bonus
      if score > best_score:
        best_score = score
        best_move = m
    return best_move

  def _resolve_batch_results(self) -> None:
    """Evaluate accumulated batch positions and resolve all deferred results into movesets.

    Called periodically during UCB1 rounds so the selection policy gets live feedback.
    Resolves all pending _round_results that have evaluations available.
    """
    if not self._batch_positions:
      return

    self.flush_batch()

    resolved_remaining = []
    for result, move in self._round_results:
      if isinstance(result, int) and result >= 0:
        value = self._position_to_result.get(result, 0.0)
      else:
        value = float(result)
      self.movesets[move]['chances'] += value
      self.movesets[move]['plays'] += 1
    self._round_results = []

  def _sample_gumbel(self, n: int, temperature: Optional[float] = None) -> List[float]:
    """Sample n values from Gumbel(0,1) distribution for Gumbel-Top-K.

    Gumbel noise is distributed as: g = -log(-log(U)) where U ~ Uniform(0,1).
    Adding Gumbel noise to scores and taking top-K is equivalent to sampling
    from the softmax distribution without replacement (Gumbel-Top-K trick).

    Args:
        n: Number of samples to generate.
        temperature: Scale factor (higher = more uniform, lower = more greedy).

    Returns:
        List of n Gumbel samples.
    """
    if temperature is None:
      temperature = self.gumbel_temperature
    # Generate uniform(0,1) avoiding exact 0 or 1
    uniforms = [random.random() for _ in range(n)]
    uniforms = [max(min(u, 0.9999999), 0.0000001) for u in uniforms]
    return [-math.log(-math.log(u)) * temperature for u in uniforms]

  def _evaluate_moves_batch(self, B: Any, moves: List[int], colour: int) -> Optional[List[Tuple[int, float]]]:
    """Get NN evaluations for all root moves in a single GPU batch.

    For each move: push, extract position, pop. Terminal moves get +-1.
    Evaluates all accumulated positions in one GPU call.

    Args:
        B: Board state.
        moves: List of candidate moves.
        colour: Current player colour.

    Returns:
        List of (move, score) tuples, or None if batch evaluation unavailable.
    """
    if not self.use_mlx or self.nn is None:
      return None
    
    batch_positions = []
    move_info = []  # (move, value_or_None)
    
    for move in moves:
      B.push_move(move)
      isOver = self.isOver(B, colour)
      if isOver[0]:
        move_info.append((move, float(isOver[1])))
      else:
        pos = self._extract_position(B, colour)
        batch_positions.append(pos)
        move_info.append((move, None))
      B.pop_move()
    
    # Evaluate batch on GPU
    if batch_positions:
      if hasattr(self.nn, 'compute_batch_mlx'):
        results = self.nn.compute_batch_mlx(batch_positions)
      elif hasattr(self.nn, 'compute_batch'):
        results = self.nn.compute_batch(batch_positions)
      else:
        return None
      
      if hasattr(results, 'numpy'):
        results = np.array(results.numpy())
      else:
        results = np.array([float(r) for r in results])
    else:
      results = []
    
    # Merge terminal and NN results
    scored_moves = []
    nn_idx = 0
    for move, value in move_info:
      if value is not None:
        scored_moves.append((move, value))
      else:
        scored_moves.append((move, float(results[nn_idx])))
        nn_idx += 1
    
    return scored_moves

  def Decide(self, B: Any, colour: int) -> int:
    """Return the best move found by TMCTS UCB1 search."""
    self.movesets = {}
    return self.random_ts(B, self.ply, colour)

  def decide(self, B: Any, colour: int) -> int:
    """Lowercase alias for Decide()."""
    return self.Decide(B, colour)

  # -------------------------------------------------------

  def random_ts(self, B: Any, ply: int, colour: int, printDebug: bool = False) -> int:
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
      
      # --- Progressive narrowing with Gumbel-Top-K ---
      # If we have many moves, use a quick NN batch eval to identify
      # the most promising ones, then only simulate those deeply.
      # Gumbel noise ensures every move still has a non-zero chance.
      narrowed_moves = moves
      if (self.progressive_narrowing and 
          len(moves) > self.progressive_narrowing_k and
          self.use_mlx):
        scored = self._evaluate_moves_batch(B, moves, colour)
        if scored:
          # Apply Gumbel-Top-K: add noise to NN scores, then sort
          gumbel = self._sample_gumbel(len(scored))
          noisy = [s + g for (_, s), g in zip(scored, gumbel)]
          # Keep top K by noisy score (descending)
          top_indices = sorted(
            range(len(noisy)), key=lambda i: noisy[i], reverse=True
          )[:self.progressive_narrowing_k]
          narrowed_moves = [scored[i][0] for i in top_indices]
      
      # Clear batch accumulation for this decision
      self._batch_positions = []
      self._position_to_result = {}
      self._position_counter = 0
      
      # Track result info for each round: (result_or_pos_idx, move)
      self._round_results = []
      
      # set up moves
      for move in narrowed_moves:
        self.movesets[move] = {
          'plays' : 0,
          'chances' : 0
        }
      
      # iterate through the random number of rounds
      for i in range(random_rounds):
        # Use UCB1 to select which move to explore
        # Prioritizes promising moves while ensuring all moves are tested
        random_move = self._select_move_ucb1(narrowed_moves)
        B.push_move(random_move)
        # start mcts - use batched tree search if MLX is available
        if self.use_mlx:
          result = self.treesearch_batch(B, ply, colour)
          self._round_results.append((result, random_move))
          # Periodically flush batch results so UCB1 gets live feedback
          # on move quality instead of waiting until all 3600 rounds finish
          if len(self._batch_positions) >= self.batch_size:
            self._resolve_batch_results()
        else:
          result = self.treesearch(B, ply, colour)
          self.movesets[random_move]['chances'] += result
          self.movesets[random_move]['plays'] += 1
        B.pop_move()

      # Flush any remaining batch results and resolve into movesets
      if self.use_mlx:
        self._resolve_batch_results()

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

  def treesearch(self, B: Any, ply: int, colour: int) -> float:
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

  def treesearch_batch(self, B: Any, ply: int, colour: int) -> Union[int, float]:
    """MLX-native tree search with batched position accumulation.

    Uses a persistent node cache for tree reuse across turns:
    - At every node (leaf or internal), checks cache first
    - Cached MCTS values from previous turns / pruned subtrees
      avoid redundant GPU evaluations and tree traversal
    - Transpositions within the same turn also benefit
    """
    isOver = self.isOver(B, colour)
    if isOver[0]:
      return float(isOver[1])
    
    # Extract position and check persistent node cache
    pos = self._extract_position(B, colour)
    pos_key = pos.tobytes()
    if pos_key in self._node_cache:
      self._cache_hits += 1
      return float(self._node_cache[pos_key])
    
    if ply < 1:
      # Leaf: accumulate position for batch evaluation
      pos_idx = self._position_counter
      self._position_counter += 1
      self._batch_positions.append((pos_idx, pos))
      return pos_idx
    
    # Internal node: opponent's random move, then player's random move
    moves = B.get_moves()
    move = random.choice(moves)
    B.push_move(move)
    
    isOver = self.isOver(B, colour)
    if isOver[0]:
      result = float(isOver[1])
      B.pop_move()
      # Cache terminal evaluation
      if len(self._node_cache) >= self._max_cache_size:
        self._node_cache.pop(next(iter(self._node_cache)))
      self._node_cache[pos_key] = result
      return result
    
    moves = B.get_moves()
    if len(moves) > 0:
      move = random.choice(moves)
      B.push_move(move)
      result = self.treesearch_batch(B, ply-1, colour)
      B.pop_move()
      B.pop_move()  # Pop enemy move
    else:
      B.pop_move()  # Pop enemy move
      result = 0.0
    
    # Cache the MCTS value for this internal node
    if len(self._node_cache) >= self._max_cache_size:
      self._node_cache.pop(next(iter(self._node_cache)))
    self._node_cache[pos_key] = result
    return result

  def flush_batch(self) -> np.ndarray:
    """Evaluate all accumulated positions in a single batch.

    Maps batch results back to individual positions using position indices.
    Stores results in _position_to_result for resolution.

    Returns:
        np.array with evaluation results.
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
    
    # Populate persistent node cache (tree reuse across turns)
    for idx, pos_array in zip(indices, position_arrays):
      pos_key = pos_array.tobytes()
      if len(self._node_cache) >= self._max_cache_size:
        # LRU-approximate eviction: remove one arbitrary entry
        self._node_cache.pop(
          next(iter(self._node_cache))
        )
      self._node_cache[pos_key] = self._position_to_result[idx]
    
    # Clear positions
    self._batch_positions = []
    
    return results

  def _extract_position(self, B: Any, colour: int) -> np.ndarray:
    """Extract board position for neural network evaluation."""
    boardStatus = B.getBoardPosWeighted(colour, {
      "Black": 1,
      "White": -1,
      "empty": 0,
      "blackKing": 1.5,
      "whiteKing": -1.5
    })

    if self.nn and hasattr(self.nn, 'layer_size') and self.nn.layer_size[0] == 91:
      boardStatus = self.nn.subsquares(boardStatus)

    return np.asarray(boardStatus, dtype=np.float32)

  def isOver(self, B: Any, colour: int) -> Tuple[bool, int]:
    if B.is_over(check_repetition=False):
      if B.winner != minimax_empty:
        if B.winner == colour:
          return (True, minimax_win)
        else:
          return (True, minimax_lose)
      else:
        return (True, minimax_draw)
    else:
      return (False, -1)