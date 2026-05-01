# Training Optimization Log

## Current Performance (Post-IPC Optimization)

From py-spy profiling after chunksize fix:
- `treesearch`: 4937 samples (core MCTS computation)
- `push_move`: 1089 samples (board mutations)
- `pop_move`: 784 samples (state rollbacks)
- `make_move`: 773 samples (move generation)
- `evaluate_board`: 253 samples (NN evals - reduced from earlier)
- `subsquares`: 247 samples (feature extraction)

## Completed Optimizations

1. ~~IPC overhead (serialization, lock contention)~~ - FIXED via chunksize
2. ~~Matplotlib empty data crash~~ - FIXED with early return guard
3. ~~Excessive console logging~~ - FIXED with file-based logging

## Pending Optimization Tasks

### HIGH PRIORITY: MLX Batch Inference

**Problem**: Each tree node calls `evaluate_board` individually. No batching happening.

**Files to modify**:
- `library/core/population.py` line 52: Change `sp.Slowpoke(self.plyDepth,debug=self.isDebug)` → add `use_mlx=True`
- `library/decision/tmcts.py`: Switch `treesearch` → `treesearch_batch`
- `library/agents/slowpoke.py`: Connect batch infrastructure (`_batch_inputs`, `_batch_refs` already exist but unused)

**Expected impact**: 10-100x reduction in NN eval overhead depending on batch size.

### MEDIUM PRIORITY: Cython/Numba for Board Operations

**Files**: `library/core/checkers.py`
**Functions**: `push_move` (line 488), `pop_move` (line 509), `make_move` (line 205)

**Approach**:
```python
# Before
def push_move(self, move):
    piece = self.board[move[0]]
    self.board[move[0]] = 0
    self.board[move[1]] = piece
    ...

# After (with @njit or @cython)
# Pure numpy operations, no Python object overhead
```

### MEDIUM PRIORITY: Numba for subsquares

**File**: `library/agents/evaluator/subsquares.py`

**Current**: List comprehension with Python loops
**Target**: Fully vectorized numpy
```python
# Current approach likely uses loops
# Target: 
def subsquares(x):
    kernel = np.array([[...]])  # 3x3 convolution kernel
    return signal.convolve2d(x, kernel, mode='valid')
```

### LOW PRIORITY: Persistent Process Pool

**File**: `library/core/tournament.py`
**Location**: `Tournament.run()` and `Tournament.runChampions()`

Currently creates new pool per generation. Could:
1. Create pool once in `__init__`
2. Use `initializer` to set up shared memory
3. Pass board states as numpy arrays for zero-copy

### LOW PRIORITY: NumPy Board Representation

**File**: `library/core/checkers.py`

Current: Python list representation
Benefits of numpy:
- Zero-copy snapshots for caching
- Vectorizable move generation
- Shared memory between processes

## Quick Wins

1. Add `use_mlx=True` to agent creation - 30 min
2. Profile `subsquares` - understand current implementation
3. Test numba on `push_move`/`pop_move` - 1 hour

## Notes

- MLX batch inference has infrastructure already written (`treesearch_batch`, `compute_batch_mlx`)
- Key is connecting the batch accumulation during tree traversal
- Current `treesearch` returns scalar per call - needs refactoring to accumulate and batch