"""
Tests for move stack optimization - correctness and performance benchmarks.
"""

import numpy as np
import pytest
import time
import sys
sys.path.insert(0, 'library')

from core.checkers import CheckerBoard
from decision.tmcts import TMCTS
from agents.evaluator.neural import NeuralNetwork


def test_move_stack_correctness():
    """Test that batched move operations produce correct results."""
    B = CheckerBoard()
    moves = B.get_moves()
    
    # Test individual push/pop
    B1 = CheckerBoard()
    original_state = (B1.active, B1.passive, B1.pieces.copy(), B1.forward.copy(), B1.backward.copy())
    for _ in range(5):
        move = np.random.choice(moves)
        B1.push_move(move)
    for _ in range(5):
        B1.pop_move()
    
    # Should be back to original state
    assert B1.active == original_state[0]
    assert B1.passive == original_state[1]
    assert B1.pieces == original_state[2]
    assert B1.forward == original_state[3]
    assert B1.backward == original_state[4]


def test_move_stack_performance():
    """Benchmark move stack operations."""
    B = CheckerBoard()
    moves = B.get_moves()
    
    # Benchmark individual push/pop
    n = 1000
    start = time.perf_counter()
    for _ in range(n):
        move = np.random.choice(moves)
        B.push_move(move)
    for _ in range(n):
        B.pop_move()
    elapsed_individual = time.perf_counter() - start
    
    print(f"\nIndividual push/pop: {n} moves in {elapsed_individual:.3f}s = {elapsed_individual/n*1000:.3f}ms per move")
    print(f"Throughput: {n/elapsed_individual:.0f} moves/sec")


def test_tmcts_with_batched_moves():
    """Test TMCTS with batched move operations."""
    nn = NeuralNetwork([32, 40, 10, 1], use_mlx=False)
    evaluator = type('Evaluator', (), {
        'nn': nn,
        'evaluate_board': lambda self, B, colour: nn.compute(B.getBoardPosWeighted(colour, {
            "Black": 1, "White": -1, "empty": 0, "blackKing": 1.5, "whiteKing": -1.5
        }))
    })()
    
    agent = TMCTS(ply=3, evaluator=evaluator, debug=True)
    B = CheckerBoard()
    
    # Run decision
    move = agent.Decide(B, 0)
    assert move in B.get_moves()


if __name__ == "__main__":
    test_move_stack_correctness()
    test_move_stack_performance()
    test_tmcts_with_batched_moves()
    print("\nAll tests passed!")