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

### HIGH PRIORITY: MLX Batch Inference - COMPLETED

**Problem**: Each tree node calls `evaluate_board` individually. No batching happening.

**Solution Implemented**:
1. Added `use_mlx=True` to Slowpoke initialization in `population.py` line 52
2. Implemented `treesearch_batch()` in `tmcts.py` with position accumulation during tree traversal
3. Fixed `compute_batch_mlx()` in `neural.py` for MLX array handling
4. Fixed `CheckerBoard.__slots__` to include `is_over_called` for test compatibility

**Impact**: **696.9x speedup** for batched neural network evaluation on M2 Ultra GPU!

**Benchmark Results**:
```
Batch Size   Time (ms)    Per Eval (us)
|--------------------------------------------|
1            14.11         14112.71
100          0.41          4.09
5000         2.11          0.42

MCTS Workload (25000 positions):
Individual: 5837ms
Batched: 8.4ms
Speedup: 696.9x
```

### Bug Fixes - COMPLETED

1. Fixed `tournament.py` gameWorker - empty cache sampling issue
2. Fixed `tmcts.py` - use `layer_size[0]` instead of `layers[0]` for subsquares check
3. **CRITICAL FIX**: Implemented position-to-result mapping for batch evaluation
   - Added `_position_counter` to track position indices during tree traversal
   - Added `_position_to_result` dictionary to map position indices to evaluation results
   - Added `_round_results` list in `random_ts()` to track (pos_idx, move) pairs
   - Implemented `flush_batch()` to evaluate accumulated positions and store results
   - Implemented `resolve_position()` to retrieve deferred evaluation results
   - After batch evaluation, results are resolved and added to move statistics
4. All 45 tests passing

### Move Stack Optimization - COMPLETED

**Problem**: `push_move` and `pop_move` use dict-based history with list copying overhead.

**Solution Implemented**:
- Replaced dict-based history with tuple-based history in `push_move()`
- Updated `pop_move()` to use tuple indexing and convert back to lists
- Includes all state needed for undo: active, passive, forward, backward, pieces, empty, jump, mandatoryJumps, noEatCount, multipleJumpStack, turnCount, moves_len, altMoveStack_len

**Impact**: ~44,000 moves/sec throughput (measured in tests)

**Benchmark Results**:
```
Individual push/pop: 1000 moves in 0.023s = 0.023ms per move
Throughput: 44398 moves/sec
```

### Logging Configuration - RESTORED

**Change**: Restored console output for training progress.

**Solution Implemented**:
- Updated `_setup_logging()` docstring to reflect dual output
- Modified `displayStatusInfo()` to always print to console (not just on generation boundaries)
- Status info now appears in terminal during training runs

**How to Use**:
```bash
# Run light simulation with console output
python library/train.py light

# Run tests
python -m pytest -v
```

## Files Modified

- `library/core/population.py` - Added `use_mlx=True` to Slowpoke initialization
- `library/decision/tmcts.py` - Implemented `treesearch_batch()` with position indexing and deferred evaluation
- `library/agents/evaluator/neural.py` - Fixed `compute_batch_mlx()` for MLX arrays
- `library/core/checkers.py` - Fixed `__slots__` with `is_over_called`
- `library/core/tournament.py` - Fixed empty cache sampling in gameWorker

## Test Status

- All 65 tests passing (45 original + 20 new TMCTS tests)
- Training simulation runs successfully with MLX enabled
- MLX batch evaluation verified working

## New Tests Added

- `library/tests/test_tmcts_batch.py` - TMCTS batch evaluation tests
- `library/tests/test_tmcts_e2e.py` - End-to-end AI comparison tests

## How to Use

```bash
# Run light simulation with MLX batch inference
python library/train.py light

# Run tests
python -m pytest -v
```

## Technical Details: Batch Evaluation Fix

The key insight was that `treesearch_batch()` needed to defer evaluation results until after the entire tree search completes. Here's how it works:

1. **Position Indexing**: Each position extracted during tree traversal gets a unique index from `_position_counter`
2. **Deferred Resolution**: Instead of returning the evaluation value immediately, `treesearch_batch()` returns the position index
3. **Result Tracking**: `_round_results` tracks which position index corresponds to which move
4. **Batch Evaluation**: After all rounds complete, `flush_batch()` evaluates all positions in one GPU call
5. **Result Mapping**: The batch results are stored in `_position_to_result` dictionary
6. **Final Resolution**: Each round's result is resolved from the dictionary and added to move statistics

This pattern allows the neural network to evaluate all accumulated positions in a single batch, achieving the 696.9x speedup on M2 Ultra GPU.