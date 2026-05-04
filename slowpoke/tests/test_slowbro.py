"""Tests for the Slowbro agent."""

import unittest

from slowpoke.agents.slowbro import Slowbro
from slowpoke.agents.agent import Agent
from slowpoke.core import checkers
import numpy as np

Black, White = 0, 1


class TestSlowbroInit(unittest.TestCase):
    """Test Slowbro initialisation."""

    def test_default_ply(self):
        """Default ply should be 4."""
        bot = Slowbro(use_mlx=False)
        self.assertEqual(bot.ply, 4)

    def test_custom_ply(self):
        """Custom ply should be respected."""
        bot = Slowbro(ply_depth=6, use_mlx=False)
        self.assertEqual(bot.ply, 6)

    def test_default_layers(self):
        """Default layers should be [32, 40, 10, 1]."""
        bot = Slowbro(use_mlx=False)
        self.assertEqual(bot.layers, [32, 40, 10, 1])

    def test_custom_layers(self):
        """Custom layers should be respected."""
        bot = Slowbro(layers=[32, 20, 10, 1], use_mlx=False)
        self.assertEqual(bot.layers, [32, 20, 10, 1])

    def test_nn_created(self):
        """Neural network should be created."""
        bot = Slowbro(use_mlx=False)
        self.assertIsNotNone(bot.nn)

    def test_nn_has_32_inputs(self):
        """NN should have 32 inputs (fused architecture)."""
        bot = Slowbro(use_mlx=False)
        self.assertEqual(bot.nn.layer_size[0], 32)

    def test_nn_has_4_layers(self):
        """NN should have 4 layers by default."""
        bot = Slowbro(use_mlx=False)
        self.assertEqual(bot.nn.num_layers, 4)

    def test_cache_enabled_by_default(self):
        """Cache should be enabled by default."""
        bot = Slowbro(use_mlx=False)
        self.assertTrue(bot.enable_cache)

    def test_cache_is_dict(self):
        """Cache should start as empty dict."""
        bot = Slowbro(use_mlx=False)
        self.assertEqual(bot.cache, {})

    def test_default_decision_function_is_tmcts(self):
        """Default decision function should be serial TMCTS."""
        from slowpoke.search.tmcts import TMCTS

        bot = Slowbro(use_mlx=False)
        self.assertIsInstance(bot.decision_function, TMCTS)

    def test_parallel_decision_function(self):
        """When use_parallel=True, decision function should be ParallelTMCTS."""
        from slowpoke.search.parallel_tmcts import ParallelTMCTS

        bot = Slowbro(use_mlx=False, use_parallel=True, num_parallel=2)
        self.assertIsInstance(bot.decision_function, ParallelTMCTS)

    def test_parallel_thread_count(self):
        """Parallel TMCTS should use specified thread count."""
        bot = Slowbro(use_mlx=False, use_parallel=True, num_parallel=3)
        self.assertEqual(bot.decision_function.num_parallel, 3)

    def test_debug_mode(self):
        """Debug mode should be stored."""
        bot = Slowbro(use_mlx=False, debug=True)
        self.assertTrue(bot.debug)


class TestSlowbroPieceWeights(unittest.TestCase):
    """Test Slowbro piece weights."""

    def setUp(self):
        self.bot = Slowbro(use_mlx=False)

    def test_black_weight(self):
        """Black piece weight should be 1."""
        self.assertEqual(self.bot.pieceWeights["Black"], 1)

    def test_white_weight(self):
        """White piece weight should be -1."""
        self.assertEqual(self.bot.pieceWeights["White"], -1)

    def test_empty_weight(self):
        """Empty square weight should be 0."""
        self.assertEqual(self.bot.pieceWeights["empty"], 0)

    def test_black_king_weight(self):
        """Black king weight should be 1.5."""
        self.assertEqual(self.bot.pieceWeights["blackKing"], 1.5)

    def test_white_king_weight(self):
        """White king weight should be -1.5."""
        self.assertEqual(self.bot.pieceWeights["whiteKing"], -1.5)


class TestSlowbroEvaluateBoard(unittest.TestCase):
    """Test Slowbro board evaluation."""

    def setUp(self):
        self.bot = Slowbro(use_mlx=False)

    def test_evaluate_returns_float(self):
        """Evaluation should return a float."""
        B = checkers.CheckerBoard()
        result = self.bot.evaluate_board(B, Black)
        self.assertIsInstance(result, float)

    def test_evaluate_initial_board(self):
        """Initial board evaluation should be within valid range."""
        B = checkers.CheckerBoard()
        result = self.bot.evaluate_board(B, Black)
        self.assertGreaterEqual(result, -1.0)
        self.assertLessEqual(result, 1.0)

    def test_evaluate_caches_result(self):
        """Repeated evaluations should use cache."""
        B = checkers.CheckerBoard()
        result1 = self.bot.evaluate_board(B, Black)
        result2 = self.bot.evaluate_board(B, Black)
        self.assertEqual(result1, result2)

    def test_evaluate_win_returns_one(self):
        """Winning position should return 1.0."""
        from unittest import mock

        with mock.patch.object(checkers.CheckerBoard, "is_over", return_value=True):
            B = checkers.CheckerBoard()
            B.winner = Black
            result = self.bot.evaluate_board(B, Black)
            self.assertEqual(result, 1.0)

    def test_evaluate_loss_returns_minus_one(self):
        """Losing position should return -1.0."""
        from unittest import mock

        with mock.patch.object(checkers.CheckerBoard, "is_over", return_value=True):
            B = checkers.CheckerBoard()
            B.winner = White
            result = self.bot.evaluate_board(B, Black)
            self.assertEqual(result, -1.0)

    def test_evaluate_draw_returns_zero(self):
        """Draw position should return 0.0."""
        from unittest import mock

        with mock.patch.object(checkers.CheckerBoard, "is_over", return_value=True):
            B = checkers.CheckerBoard()
            B.winner = -1
            result = self.bot.evaluate_board(B, Black)
            self.assertEqual(result, 0.0)


class TestSlowbroLoadWeights(unittest.TestCase):
    """Test Slowbro weight loading."""

    def test_load_weights_32_input(self):
        """Loading 32-input weights should work directly."""
        bot = Slowbro(use_mlx=False)
        coeffs = bot.nn.get_all_coefficients().copy()
        bot.load_weights(coeffs)
        np.testing.assert_allclose(bot.nn.get_all_coefficients(), coeffs, rtol=1e-5)

    def test_load_weights_preserves_architecture(self):
        """Loading new weights should not change architecture."""
        bot = Slowbro(use_mlx=False)
        coeffs = np.ones(bot.nn.len_coefficients, dtype=np.float32)
        bot.load_weights(coeffs)
        self.assertEqual(bot.nn.layer_size, [32, 40, 10, 1])

    def test_load_legacy_91_weights(self):
        """Loading legacy [91,40,10,1] weights should fuse them."""
        legacy_count = 91 * 40 + 40 + 40 * 10 + 10 + 10 * 1 + 1
        legacy_nn_weights = np.random.random(legacy_count).astype(np.float32)
        bot = Slowbro(use_mlx=False, layers=[32, 40, 10, 1])
        bot.load_weights(legacy_nn_weights)
        # After fusion, NN should have 32 inputs
        self.assertEqual(bot.nn.layer_size[0], 32)
        # Evaluation should work
        B = checkers.CheckerBoard()
        result = bot.evaluate_board(B, Black)
        self.assertIsInstance(result, float)


class TestSlowbroMoveFunction(unittest.TestCase):
    """Test Slowbro move function."""

    def test_make_valid_move(self):
        """Slowbro should make a valid move."""
        bot = Slowbro(use_mlx=False, ply_depth=1)
        agent = Agent(bot)
        B = checkers.CheckerBoard()
        move = agent.make_move(B, Black)
        self.assertIn(move, B.get_moves())

    def test_move_function_returns_int(self):
        """Move should be an integer."""
        bot = Slowbro(use_mlx=False, ply_depth=1)
        agent = Agent(bot)
        B = checkers.CheckerBoard()
        move = agent.make_move(B, Black)
        self.assertIsInstance(move, int)


if __name__ == "__main__":
    unittest.main(verbosity=2)
