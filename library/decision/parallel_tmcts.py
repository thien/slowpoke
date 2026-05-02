"""
Parallel TMCTS - Multiple independent MCTS instances sharing a batch accumulator.

AlphaZero-style parallelization: Run N independent MCTS instances in parallel,
each accumulating positions during tree traversal, then batch-evaluate all
positions together for maximum GPU utilization.
"""

import random
import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    mx = None
    MLX_AVAILABLE = False

class SharedBatchAccumulator:
    """Thread-safe batch accumulator shared across multiple TMCTS instances.
    
    Also maintains a persistent evaluation cache for tree reuse across turns.
    """
    
    def __init__(self, batch_size: int = 512, auto_flush: bool = True):
        self._batch_positions: List[Tuple[int, np.ndarray]] = []
        self._position_to_result: Dict[int, float] = {}
        self._position_counter = 0
        self._batch_size = batch_size
        self._auto_flush = auto_flush
        self._total_positions = 0
        self._lock = Lock()  # Created fresh each time, not pickled
        
        # --- Tree reuse across turns: persistent node cache ---
        # Maps board position bytes -> MCTS value
        # Survives across decide() calls so evaluations from one turn's
        # internal nodes are reused when the same position is encountered
        # again (tree reuse + transpositions).
        self._node_cache: Dict[bytes, float] = {}
        self._max_cache_size = 50000
    
    def __getstate__(self):
        """Make the lock transient - it gets recreated on unpickling."""
        state = self.__dict__.copy()
        del state['_lock']
        return state
    
    def __setstate__(self, state):
        """Recreate the lock when unpickling."""
        self.__dict__.update(state)
        self._lock = Lock()
    
    def add_position(self, pos: np.ndarray) -> int:
        """Add a position to the batch, return its index.
        
        Returns an index which can be resolved via get_result().
        Checks persistent cache first to avoid redundant evaluations.
        """
        pos_key = pos.tobytes()
        with self._lock:
            # Check persistent node cache first (tree reuse across turns)
            if pos_key in self._node_cache:
                pos_idx = self._position_counter
                self._position_counter += 1
                # Store cached result directly so _run_instance can find it
                self._position_to_result[pos_idx] = self._node_cache[pos_key]
                return pos_idx
            
            pos_idx = self._position_counter
            self._position_counter += 1
            self._batch_positions.append((pos_idx, pos))
            self._total_positions += 1
            
            # Auto-flush if batch is full
            if self._auto_flush and len(self._batch_positions) >= self._batch_size:
                return -1  # Signal to flush
            return pos_idx
    
    def get_batch(self) -> Tuple[List[int], List[np.ndarray]]:
        """Get current batch contents. Must be followed by clear_batch()."""
        with self._lock:
            if not self._batch_positions:
                return [], []
            sorted_positions = sorted(self._batch_positions, key=lambda x: x[0])
            indices = [p[0] for p in sorted_positions]
            arrays = [p[1] for p in sorted_positions]
            return indices, arrays
    
    def clear_batch(self):
        """Clear the accumulated batch."""
        with self._lock:
            self._batch_positions = []

    def clear(self):
        """Reset the accumulator for a new root evaluation cycle.

        Clears batch positions, result lookup table, and position counter,
        but preserves the persistent node cache for tree reuse across turns.
        This is called after root-level batch evaluation so the main MCTS
        tree search starts with a clean slate.
        """
        with self._lock:
            self._batch_positions = []
            self._position_to_result = {}
            self._position_counter = 0
    
    def store_results(self, indices: List[int], results: np.ndarray):
        """Store evaluation results in the lookup table."""
        with self._lock:
            for idx, result in zip(indices, results):
                self._position_to_result[idx] = float(result)
    
    def get_result(self, pos_idx: int) -> float:
        """Get the evaluation result for a position."""
        with self._lock:
            return self._position_to_result.get(pos_idx, 0.0)
    
    def get_cached_value(self, pos_key: bytes) -> Optional[float]:
        """Check if a position has a cached MCTS value (thread-safe).
        
        Returns cached value or None if not cached.
        Used by _treesearch_batch for tree reuse across turns.
        """
        with self._lock:
            return self._node_cache.get(pos_key)
    
    def store_cached_value(self, pos_key: bytes, value: float):
        """Store an MCTS value in the persistent node cache (thread-safe).
        
        Used by _treesearch_batch to cache internal node results
        for tree reuse across turns.
        """
        with self._lock:
            if len(self._node_cache) >= self._max_cache_size:
                self._node_cache.pop(next(iter(self._node_cache)))
            self._node_cache[pos_key] = value
    
    def flush_and_evaluate(self, nn) -> Optional[np.ndarray]:
        """Flush batch and evaluate using neural network.
        
        Also populates the persistent evaluation cache so subsequent
        turns can reuse leaf evaluations via transposition.
        """
        indices, arrays = self.get_batch()
        if not arrays:
            return None
        
        # Compute cache keys before clearing
        cache_keys = [arr.tobytes() for arr in arrays]
        
        self.clear_batch()
        
        # Evaluate batch
        if nn is not None and hasattr(nn, 'compute_batch_mlx'):
            results = nn.compute_batch_mlx(arrays)
            if hasattr(results, 'numpy'):
                results = np.array(results.numpy())
            elif not isinstance(results, np.ndarray):
                results = np.array([float(r) for r in results])
        else:
            results = np.array([0.0] * len(arrays), dtype=np.float32)
        
        self.store_results(indices, results)
        
        # Populate persistent node cache (tree reuse across turns)
        with self._lock:
            for key, result in zip(cache_keys, results):
                if len(self._node_cache) >= self._max_cache_size:
                    self._node_cache.pop(next(iter(self._node_cache)))
                self._node_cache[key] = float(result)
        
        return results

class ParallelTMCTS:
    """Runs multiple TMCTS instances in parallel, sharing a batch accumulator.
    
    This achieves AlphaZero-style parallelization:
    - Multiple independent trees traversing simultaneously
    - All positions accumulated into a shared batch
    - Single batch evaluation per flush
    - Results distributed back to each tree
    """
    
    def __init__(self, ply: int, evaluator, num_parallel: int = 4, 
                 batch_size: int = 512, debug: bool = False, seed: int = None):
        self.ply = ply
        self.evaluator = evaluator
        self.num_parallel = num_parallel
        self.debug = debug
        self.seed = seed
        
        # Progressive narrowing with Gumbel-Top-K
        self.progressive_narrowing = True
        self.progressive_narrowing_k = 5
        self.gumbel_temperature = 0.5
        
        # Cache support (for tournament compatibility)
        self.enableCache = False
        self.cache = {}
        
        # Shared batch accumulator
        self.accumulator = SharedBatchAccumulator(batch_size=batch_size, auto_flush=False)
        
        # Neural network reference
        self.nn = getattr(evaluator, 'nn', None)
        self.use_mlx = (hasattr(evaluator, 'nn') and 
                        evaluator.nn is not None and 
                        getattr(evaluator.nn, '_use_mlx', False))
        
        # Base rounds per instance
        self.baseRound = 300
        if debug:
            self.baseRound = 10
    
    def __getstate__(self):
        """Make the object picklable by handling unpicklable attributes."""
        state = self.__dict__.copy()
        # accumulator has __getstate__ for its lock, so it should work
        return state
    
    def __setstate__(self, state):
        """Restore state when unpickling."""
        self.__dict__.update(state)
    
    def Decide(self, B, colour: int) -> Any:
        """Run parallel MCTS and return best move."""
        return self._decide_impl(B, colour)
    
    def _decide_impl(self, B, colour: int) -> Any:
        """Internal implementation of decide."""
        moves = B.get_moves()
        
        if len(moves) == 1:
            return moves[0]
        
        if len(moves) < 4:
            ply = self.ply + len(moves)
        else:
            ply = self.ply
        
        # Progressive narrowing: evaluate all root moves and select top-K
        narrowed_moves = moves
        if (self.progressive_narrowing and 
            self.use_mlx and 
            len(moves) > self.progressive_narrowing_k):
            scored = self._evaluate_root_moves(B, moves, colour)
            if scored:
                # Add Gumbel noise for stochastic exploration
                gumbel = self._sample_gumbel(len(scored))
                noisy = [s + gumbel[i] for i, (_, s) in enumerate(scored)]
                # Select top-K by noisy score
                top_k = sorted(range(len(noisy)), key=lambda i: noisy[i], reverse=True)[:self.progressive_narrowing_k]
                narrowed_moves = [scored[i][0] for i in top_k]
                if self.debug:
                    print(f"[ParallelTMCTS] Narrowed {len(moves)} -> {len(narrowed_moves)} root moves")
        
        random_rounds = self.baseRound * ply if ply > 0 else 1
        
        # Track move statistics across all instances
        move_stats: Dict[Any, Dict[str, float]] = {m: {'wins': 0.0, 'plays': 0} for m in narrowed_moves}
        
        # Run parallel instances with seeded random generators
        with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
            futures = []
            for i in range(self.num_parallel):
                # Each thread gets a deterministic seed derived from base seed
                thread_seed = (self.seed if self.seed is not None else 42) + i
                future = executor.submit(
                    self._run_instance, B, ply, colour, random_rounds, move_stats, thread_seed, narrowed_moves
                )
                futures.append(future)
            
            # Wait for all instances to complete
            for future in as_completed(futures):
                pass  # Results already updated in move_stats
        
        # Flush any remaining positions
        self.accumulator.flush_and_evaluate(self.nn)
        
        # Select best move (deterministic: sort by move value for tie-breaking)
        best_move = None
        best_chance = -float('inf')
        
        for m in sorted(narrowed_moves):  # Sort moves for deterministic tie-breaking
            stats = move_stats[m]
            if stats['plays'] > 0:
                chance = stats['wins'] / stats['plays']
            else:
                chance = 0.0
            
            if chance > best_chance:
                best_chance = chance
                best_move = m
        
        return best_move
    
    def _sample_gumbel(self, n, temperature=None):
        """Sample n values from Gumbel(0,1) distribution."""
        import random as _random
        if temperature is None:
            temperature = self.gumbel_temperature
        return [-math.log(-math.log(_random.random())) * temperature for _ in range(n)]
    
    def _evaluate_root_moves(self, B, moves, colour):
        """Evaluate all root moves via NN to get scores for narrowing.
        
        For each move: push, extract position, pop, accumulate in batch.
        Returns list of (move, score) tuples, or None if unavailable.
        """
        if not self.use_mlx or self.nn is None:
            return None
        
        move_info = []  # (move, value_or_None)
        
        for move in moves:
            B.push_move(move)
            isOver = self._isOver(B, colour)
            if isOver[0]:
                move_info.append((move, float(isOver[1])))
            else:
                # Extract and accumulate for batch evaluation
                pos = self._extract_position(B, colour)
                idx = self.accumulator.add_position(pos)
                move_info.append((move, idx))
            B.pop_move()
        
        # Flush and evaluate all accumulated positions
        self.accumulator.flush_and_evaluate(self.nn)
        
        # Collect results
        scored = []
        for move, value_or_idx in move_info:
            if isinstance(value_or_idx, float):
                scored.append((move, value_or_idx))
            else:
                result = self.accumulator.get_result(value_or_idx)
                scored.append((move, float(result)))
        
        self.accumulator.clear()
        return scored
    
    def _run_instance(self, B, ply: int, colour: int, rounds: int, 
                      move_stats: Dict[Any, Dict[str, float]], thread_seed: int,
                      narrowed_moves=None):
        """Run a single MCTS instance with its own seeded random generator."""
        rng = random.Random(thread_seed)  # Thread-local RNG
        for _ in range(rounds):
            # Use narrowed moves if provided, otherwise get all moves
            if narrowed_moves is not None:
                move = rng.choice(narrowed_moves)
            else:
                move = rng.choice(B.get_moves())
            B.push_move(move)
            
            # Run batch tree search
            result = self._treesearch_batch(B, ply, colour, rng)
            
            # Resolve result
            if isinstance(result, int) and result >= 0:
                value = self.accumulator.get_result(result)
            else:
                value = float(result)
            
            # Update stats (thread-safe dict access)
            move_stats[move]['wins'] += value
            move_stats[move]['plays'] += 1
            
            B.pop_move()
    
    def _treesearch_batch(self, B, ply: int, colour: int, rng):
        """MLX-native tree search with shared batch accumulator.
        
        Uses persistent node cache for tree reuse across turns:
        - Every node (leaf or internal) checks cache first
        - Cached values from previous turns/pruned subtrees are reused
        - Results cached for future turns within the same game
        """
        isOver = self._isOver(B, colour)
        if isOver[0]:
            return float(isOver[1])
        
        # Extract position and check persistent node cache
        pos = self._extract_position(B, colour)
        pos_key = pos.tobytes()
        cached = self.accumulator.get_cached_value(pos_key)
        if cached is not None:
            return float(cached)
        
        if ply < 1:
            # Leaf: accumulate position for batch evaluation
            return self.accumulator.add_position(pos)
        
        # Get moves and choose random enemy move
        moves = B.get_moves()
        if not moves:
            return 0.0
        
        move = rng.choice(moves)
        B.push_move(move)
        
        # Check if that move ended the game
        isOver = self._isOver(B, colour)
        if isOver[0]:
            result = float(isOver[1])
            B.pop_move()
            # Cache terminal evaluation
            self.accumulator.store_cached_value(pos_key, result)
            return result
        
        # Get moves for next level
        moves = B.get_moves()
        if moves:
            # Choose random player move
            move = rng.choice(moves)
            B.push_move(move)
            # Traverse, moving down the player ply
            result = self._treesearch_batch(B, ply - 1, colour, rng)
            B.pop_move()
        
        B.pop_move()  # Pop enemy move
        
        # Cache the MCTS value for this internal node
        self.accumulator.store_cached_value(pos_key, result)
        return result
    
    def _extract_position(self, B, colour: int) -> np.ndarray:
        """Extract board position for neural network evaluation."""
        boardStatus = B.getBoardPosWeighted(colour, {
            "Black": 1, "White": -1, "empty": 0,
            "blackKing": 1.5, "whiteKing": -1.5
        })
        
        layer_size = None
        if self.nn and hasattr(self.nn, 'layer_size'):
            layer_size = self.nn.layer_size[0]
        elif hasattr(self.evaluator, 'layer_size'):
            layer_size = self.evaluator.layer_size[0]
        
        if layer_size == 91:
            boardStatus = self.nn.subsquares(boardStatus) if self.nn else boardStatus
        
        return np.array(boardStatus, dtype=np.float32)
    
    def _isOver(self, B, colour: int) -> Tuple[bool, int]:
        minimax_win = 1
        minimax_lose = -minimax_win
        minimax_draw = 0
        minimax_empty = -1
        
        if B.is_over():
            if B.winner != minimax_empty:
                return (True, minimax_win if B.winner == colour else minimax_lose)
            return (True, minimax_draw)
        return (False, -1)
    
    def move_function(self, board, colour):
        """Wrapper for decide() to match Slowpoke interface."""
        return self.decide(board, colour)


# Convenience function to compare single vs parallel
def benchmark_parallel_tmcts(B, colour: int, ply: int, evaluator, 
                              num_parallel: int = 4, rounds_per_instance: int = 100, seed: int = 42):
    """Benchmark single vs parallel TMCTS."""
    import time
    from .tmcts import TMCTS
    
    # Single instance
    single = TMCTS(ply, evaluator, debug=True)
    single.baseRound = rounds_per_instance
    
    start = time.time()
    move_single = single.Decide(B, colour)
    time_single = time.time() - start
    
    # Parallel instance with same seed
    parallel = ParallelTMCTS(ply, evaluator, num_parallel=num_parallel, debug=True, seed=seed)
    parallel.baseRound = rounds_per_instance
    
    start = time.time()
    move_parallel = parallel.decide(B, colour)
    time_parallel = time.time() - start
    
    print(f"Single TMCTS: {time_single:.3f}s")
    print(f"Parallel TMCTS ({num_parallel}x): {time_parallel:.3f}s")
    print(f"Speedup: {time_single/time_parallel:.2f}x")
    print(f"Same move: {move_single == move_parallel} (expected False - single uses global random)")
    print(f"Parallel with same seed: deterministic (see test_deterministic_with_seed)")
    
    return move_single, move_parallel, time_single, time_parallel