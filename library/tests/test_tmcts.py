"""Comprehensive tests for TMCTS tree-based MCTS decision module."""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision import tmcts
from decision.tmcts import TMCTS, TERMINAL_VALUE_MARKER
from core import checkers
import numpy as np


Black, White = 0, 1


class ConstantEvaluator:
    """Evaluator that returns a constant value."""
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self, board, colour):
        return self.value

    def evaluate_board(self, board, colour):
        return self.value


class TestTMCTSInit(unittest.TestCase):
    """Test TMCTS initialisation."""

    def test_ply_stored(self):
        """Ply depth should be stored."""
        tc = TMCTS(ply=3, evaluator=ConstantEvaluator())
        self.assertEqual(tc.ply, 3)

    def test_evaluator_stored(self):
        """Evaluator should be stored."""
        ev = ConstantEvaluator()
        tc = TMCTS(ply=1, evaluator=ev)
        self.assertIs(tc.evaluator, ev)

    def test_default_base_round(self):
        """Default base round should be 300."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator())
        self.assertEqual(tc.baseRound, 300)

    def test_default_ucb_exploration(self):
        """Default UCB exploration constant should be 1.4."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator())
        self.assertEqual(tc.ucb_exploration, 1.4)

    def test_progressive_narrowing_enabled(self):
        """Progressive narrowing should be enabled by default."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator())
        self.assertTrue(tc.progressive_narrowing)

    def test_progressive_narrowing_k(self):
        """Default narrowing K should be 5."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator())
        self.assertEqual(tc.progressive_narrowing_k, 5)

    def test_gumbel_temperature(self):
        """Default gumbel temperature should be 0.5."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator())
        self.assertEqual(tc.gumbel_temperature, 0.5)

    def test_debug_reduces_base_round(self):
        """Debug mode should reduce base round to 10."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator(), debug=True)
        self.assertEqual(tc.baseRound, 10)

    def test_batch_size_default(self):
        """Default batch size should be 512."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator())
        self.assertEqual(tc.batch_size, 512)

    def test_custom_batch_size(self):
        """Custom batch size should be respected."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator(), batch_size=128)
        self.assertEqual(tc.batch_size, 128)

    def test_node_cache_initialized(self):
        """Node cache should be a dict."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator())
        self.assertEqual(tc._node_cache, {})

    def test_max_cache_size(self):
        """Max cache size should be 50000."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator())
        self.assertEqual(tc._max_cache_size, 50000)


class TestTMCTSIsOver(unittest.TestCase):
    """Test TMCTS terminal state detection."""

    def setUp(self):
        self.tc = TMCTS(ply=1, evaluator=ConstantEvaluator())

    def test_not_over_initial(self):
        """Initial board should not be over."""
        B = checkers.CheckerBoard()
        result = self.tc.isOver(B, Black)
        self.assertFalse(result[0])
        self.assertEqual(result[1], -1)

    def test_terminal_win(self):
        """Winning terminal state should return (True, 1)."""
        from unittest import mock
        with mock.patch.object(checkers.CheckerBoard, 'is_over', return_value=True):
            B = checkers.CheckerBoard()
            B.winner = Black
            result = self.tc.isOver(B, Black)
            self.assertTrue(result[0])
            self.assertEqual(result[1], tmcts.minimax_win)

    def test_terminal_loss(self):
        """Losing terminal state should return (True, -1)."""
        from unittest import mock
        with mock.patch.object(checkers.CheckerBoard, 'is_over', return_value=True):
            B = checkers.CheckerBoard()
            B.winner = White
            result = self.tc.isOver(B, Black)
            self.assertTrue(result[0])
            self.assertEqual(result[1], tmcts.minimax_lose)

    def test_terminal_draw(self):
        """Draw terminal state should return (True, 0)."""
        from unittest import mock
        with mock.patch.object(checkers.CheckerBoard, 'is_over', return_value=True):
            B = checkers.CheckerBoard()
            B.winner = -1
            result = self.tc.isOver(B, Black)
            self.assertTrue(result[0])
            self.assertEqual(result[1], tmcts.minimax_draw)


class TestTMCTSGumbel(unittest.TestCase):
    """Test Gumbel distribution sampling."""

    def setUp(self):
        self.tc = TMCTS(ply=1, evaluator=ConstantEvaluator())

    def test_sample_returns_correct_count(self):
        """Should return exactly n samples."""
        samples = self.tc._sample_gumbel(5)
        self.assertEqual(len(samples), 5)

    def test_sample_returns_floats(self):
        """All samples should be floats."""
        samples = self.tc._sample_gumbel(10)
        for s in samples:
            self.assertIsInstance(s, float)

    def test_sample_different_calls_different(self):
        """Consecutive calls should produce different samples."""
        s1 = self.tc._sample_gumbel(50)
        s2 = self.tc._sample_gumbel(50)
        self.assertNotEqual(s1, s2)

    def test_temperature_scales(self):
        """Higher temperature should scale results."""
        s_low = self.tc._sample_gumbel(100, temperature=0.1)
        s_high = self.tc._sample_gumbel(100, temperature=2.0)
        # Std dev should be larger for higher temperature
        self.assertGreater(np.std(s_high), np.std(s_low))

    def test_zero_temperature(self):
        """Zero temperature should return all zeros."""
        samples = self.tc._sample_gumbel(5, temperature=0.0)
        for s in samples:
            self.assertEqual(s, 0.0)

    def test_uses_instance_temperature(self):
        """Uses self.gumbel_temperature when not specified."""
        self.tc.gumbel_temperature = 0.0
        samples = self.tc._sample_gumbel(5)
        for s in samples:
            self.assertEqual(s, 0.0)


class TestTMCTSUCB1(unittest.TestCase):
    """Test UCB1 move selection."""

    def setUp(self):
        self.tc = TMCTS(ply=1, evaluator=ConstantEvaluator())

    def test_select_unvisited_first(self):
        """Unvisited moves should be selected first."""
        moves = [1, 2, 3]
        self.tc.movesets = {m: {'plays': 0, 'chances': 0} for m in moves}
        self.tc.movesets[1] = {'plays': 5, 'chances': 3}
        # Move 1 has visits, moves 2 and 3 don't
        selected = self.tc._select_move_ucb1(moves)
        self.assertIn(selected, [2, 3])

    def test_selects_highest_ucb(self):
        """Move with highest UCB should be selected when all visited."""
        moves = [1, 2]
        self.tc.movesets = {
            1: {'plays': 10, 'chances': 9},   # 90% win rate
            2: {'plays': 10, 'chances': 5},   # 50% win rate
        }
        selected = self.tc._select_move_ucb1(moves, C=0.0)  # No exploration bonus
        self.assertEqual(selected, 1)

    def test_exploration_bonus(self):
        """Lower-visited moves should get exploration bonus."""
        moves = [1, 2]
        self.tc.movesets = {
            1: {'plays': 100, 'chances': 60},  # 60% win, many visits
            2: {'plays': 2, 'chances': 1},     # 50% win, few visits
        }
        # With high C, move 2 gets large exploration bonus
        selected = self.tc._select_move_ucb1(moves, C=10.0)
        self.assertEqual(selected, 2)

    def test_single_move(self):
        """Single move should always be selected."""
        self.tc.movesets = {1: {'plays': 5, 'chances': 3}}
        selected = self.tc._select_move_ucb1([1])
        self.assertEqual(selected, 1)

    def test_returns_random_from_unvisited(self):
        """Multiple unvisited moves should return one at random."""
        moves = [1, 2, 3]
        self.tc.movesets = {m: {'plays': 0, 'chances': 0} for m in moves}
        selections = set()
        for _ in range(30):
            selections.add(self.tc._select_move_ucb1(moves))
        # All three should be selected at some point
        self.assertEqual(selections, {1, 2, 3})


class TestTMCTSDecide(unittest.TestCase):
    """Test TMCTS decision making."""

    def test_returns_valid_move(self):
        """Decide should return a valid move."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator(0.0), debug=True)
        B = checkers.CheckerBoard()
        move = tc.Decide(B, Black)
        self.assertIn(move, B.get_moves())

    def test_returns_integer(self):
        """Returned move should be an integer."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator(0.0), debug=True)
        B = checkers.CheckerBoard()
        move = tc.Decide(B, Black)
        self.assertIsInstance(move, int)

    def test_single_move_returns_immediately(self):
        """Single-move state should return without search."""
        B = checkers.CheckerBoard()
        tc = TMCTS(ply=5, evaluator=ConstantEvaluator(0.0))
        for _ in range(60):
            moves = B.get_moves()
            if len(moves) == 1:
                break
            B.make_move(moves[0])
        if len(B.get_moves()) == 1:
            move = tc.Decide(B, B.active)
            self.assertEqual(move, B.get_moves()[0])

    def test_decide_alias(self):
        """decide() lowercase alias should work."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator(0.0), debug=True)
        B = checkers.CheckerBoard()
        move = tc.decide(B, Black)
        self.assertIn(move, B.get_moves())

    def test_movesets_created(self):
        """After Decide, movesets should be populated."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator(0.0), debug=True)
        B = checkers.CheckerBoard()
        tc.Decide(B, Black)
        self.assertIsInstance(tc.movesets, dict)
        self.assertTrue(len(tc.movesets) > 0)


class TestTMCTSTreeSearch(unittest.TestCase):
    """Test TMCTS tree search functions."""

    def test_treesearch_returns_numeric(self):
        """treesearch should return a numeric value."""
        tc = TMCTS(ply=2, evaluator=ConstantEvaluator(0.5))
        B = checkers.CheckerBoard()
        result = tc.treesearch(B, 1, Black)
        self.assertIsInstance(result, (int, float))

    def test_treesearch_no_ply(self):
        """treesearch at ply=0 should call evaluator."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator(0.3))
        B = checkers.CheckerBoard()
        result = tc.treesearch(B, 0, Black)
        self.assertEqual(result, 0.3)

    def test_treesearch_calls_evaluator_board(self):
        """treesearch should call evaluator.evaluate_board if available."""
        class ObjEvaluator:
            def evaluate_board(self, board, colour):
                return 0.7
        tc = TMCTS(ply=1, evaluator=ObjEvaluator())
        B = checkers.CheckerBoard()
        result = tc.treesearch(B, 0, Black)
        self.assertEqual(result, 0.7)

    def test_treesearch_single_ply(self):
        """treesearch at ply=1 should work."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator(0.5))
        B = checkers.CheckerBoard()
        result = tc.treesearch(B, 1, Black)
        self.assertIsInstance(result, (int, float))

    def test_treesearch_terminal(self):
        """Terminal state in treesearch should return immediately."""
        from unittest import mock
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator(0.5))
        B = checkers.CheckerBoard()
        with mock.patch.object(checkers.CheckerBoard, 'is_over', return_value=True):
            B.winner = Black
            result = tc.treesearch(B, 1, Black)
            self.assertEqual(result, tmcts.minimax_win)

    def test_treesearch_restores_board(self):
        """Board state should be restored after treesearch."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator(0.5))
        B = checkers.CheckerBoard()
        original_moves = B.get_moves()
        tc.treesearch(B, 1, Black)
        self.assertEqual(B.get_moves(), original_moves)


class TestTMCTSPositionExtraction(unittest.TestCase):
    """Test position extraction for NN evaluation."""

    def test_extract_returns_ndarray(self):
        """Extracted position should be a numpy array."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator())
        B = checkers.CheckerBoard()
        pos = tc._extract_position(B, Black)
        self.assertIsInstance(pos, np.ndarray)

    def test_extract_32_element(self):
        """Default (32-input) extraction should return 32 elements."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator())
        B = checkers.CheckerBoard()
        pos = tc._extract_position(B, Black)
        self.assertEqual(len(pos), 32)

    def test_extract_dtype_float32(self):
        """Extracted position should be float32."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator())
        B = checkers.CheckerBoard()
        pos = tc._extract_position(B, Black)
        self.assertEqual(pos.dtype, np.float32)


class TestTMCTSFlushBatch(unittest.TestCase):
    """Test batch flushing."""

    def test_flush_empty(self):
        """Flushing an empty batch should return array([0.0])."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator(0.0))
        result = tc.flush_batch()
        self.assertIsInstance(result, np.ndarray)

    def test_flush_with_positions(self):
        """Flushing with positions should evaluate them."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator(0.0))
        B = checkers.CheckerBoard()
        for _ in range(3):
            pos = tc._extract_position(B, Black)
            idx = tc._position_counter
            tc._position_counter += 1
            tc._batch_positions.append((idx, pos))
        result = tc.flush_batch()
        self.assertTrue(len(result) > 0)

    def test_flush_clears_positions(self):
        """After flush, batch positions should be cleared."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator(0.0))
        B = checkers.CheckerBoard()
        pos = tc._extract_position(B, Black)
        tc._batch_positions.append((0, pos))
        tc.flush_batch()
        self.assertEqual(tc._batch_positions, [])

    def test_flush_populates_position_to_result(self):
        """After flush, position results should be stored."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator(0.0))
        B = checkers.CheckerBoard()
        pos = tc._extract_position(B, Black)
        tc._batch_positions.append((0, pos))
        tc.flush_batch()
        self.assertIn(0, tc._position_to_result)

    def test_flush_populates_node_cache(self):
        """After flush, node cache should be populated."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator(0.0))
        B = checkers.CheckerBoard()
        pos = tc._extract_position(B, Black)
        tc._batch_positions.append((0, pos))
        tc.flush_batch()
        self.assertTrue(len(tc._node_cache) > 0)


class TestTMCTSNodeCache(unittest.TestCase):
    """Test tree reuse node cache."""

    def test_cache_hits_counted(self):
        """Cache hits should be counted."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator(0.0))
        B = checkers.CheckerBoard()
        # First call populates cache
        tc.treesearch_batch(B, 0, Black)
        tc.flush_batch()
        # Second call should hit cache
        tc.treesearch_batch(B, 0, Black)
        self.assertGreater(tc._cache_hits, 0)

    def test_cache_size_limited(self):
        """Cache should not exceed max size."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator(0.0))
        tc._max_cache_size = 5
        B = checkers.CheckerBoard()
        # Add more positions than cache can hold
        for i in range(10):
            pos = tc._extract_position(B, Black)
            tc._batch_positions.append((i, pos))
        tc.flush_batch()
        self.assertLessEqual(len(tc._node_cache), 5)


class TestTMCTSProgressiveNarrowingFlag(unittest.TestCase):
    """Test progressive narrowing feature flags."""

    def test_disabled_does_not_narrow(self):
        """When disabled and MLX off, keep all moves."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator())
        tc.progressive_narrowing = False
        tc.use_mlx = False
        B = checkers.CheckerBoard()
        moves = B.get_moves()
        # _evaluate_moves_batch returns None when use_mlx is False
        result = tc._evaluate_moves_batch(B, moves, Black)
        self.assertIsNone(result)

    def test_evaluate_moves_needs_mlx(self):
        """_evaluate_moves_batch returns None without MLX."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator())
        tc.use_mlx = False
        B = checkers.CheckerBoard()
        moves = B.get_moves()
        result = tc._evaluate_moves_batch(B, moves, Black)
        self.assertIsNone(result)

    def test_mlx_disabled_by_default(self):
        """MLX should be disabled by default for ConstantEvaluator."""
        tc = TMCTS(ply=1, evaluator=ConstantEvaluator())
        self.assertFalse(tc.use_mlx)


class TestTMCTSTerminalMarker(unittest.TestCase):
    """Test terminal value marker constant."""

    def test_marker_is_minus_one(self):
        """TERMINAL_VALUE_MARKER should be -1."""
        self.assertEqual(TERMINAL_VALUE_MARKER, -1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
