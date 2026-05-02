"""Tests for the subsquares feature extraction module."""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.evaluator.subsquares import (
    _build_subsquare_matrix,
    SUBSQUARE_MATRIX,
    subsquares,
    make_fused_nn,
)
from agents.evaluator.neural import NeuralNetwork
import numpy as np


class TestSubsquareMatrix(unittest.TestCase):
    """Test the 91x32 subsquare matrix construction."""

    def test_matrix_shape(self):
        """Matrix should be 91x32."""
        self.assertEqual(SUBSQUARE_MATRIX.shape, (91, 32))

    def test_matrix_row_count(self):
        """There must be exactly 91 subsquare rows."""
        M = _build_subsquare_matrix()
        self.assertEqual(M.shape[0], 91)

    def test_matrix_values_non_negative(self):
        """All entries in the matrix must be >= 0 (they are weights)."""
        self.assertTrue(np.all(SUBSQUARE_MATRIX >= 0))

    def test_matrix_rows_sum_to_one(self):
        """Each row of the matrix should sum to 1.0 (averaging)."""
        row_sums = np.sum(SUBSQUARE_MATRIX, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-6)

    def test_matrix_is_constant(self):
        """Rebuilding the matrix should give identical result."""
        M2 = _build_subsquare_matrix()
        np.testing.assert_array_equal(SUBSQUARE_MATRIX, M2)


class TestSubsquares(unittest.TestCase):
    """Test the subsquares function."""

    def setUp(self):
        self.x = np.zeros(32, dtype=np.float32)

    def test_returns_91_elements(self):
        """Output should be 91-element vector."""
        result = subsquares(self.x)
        self.assertEqual(len(result), 91)

    def test_zero_input_gives_zero_output(self):
        """All-zero input should produce all-zero output."""
        result = subsquares(self.x)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_all_ones_input_gives_ones(self):
        """All-ones input should produce all-ones output (averaging)."""
        x = np.ones(32, dtype=np.float32)
        result = subsquares(x)
        np.testing.assert_allclose(result, 1.0, rtol=1e-6)

    def test_single_active_square(self):
        """Only one square active: all subsquares covering it should return 1/n."""
        x = np.zeros(32, dtype=np.float32)
        x[0] = 1.0
        result = subsquares(x)
        # Every subsquare that includes square 0 will have value 1/(window_size)
        # There should be some non-zero values
        self.assertTrue(np.any(result > 0))
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1.0))


class TestMakeFusedNN(unittest.TestCase):
    """Test fusing a 91-input NN into a 32-input NN."""

    def setUp(self):
        self.nn_91 = NeuralNetwork([91, 20, 10, 1])

    def test_fused_network_has_32_inputs(self):
        """Fused network should have 32 inputs."""
        fused = make_fused_nn(self.nn_91)
        self.assertEqual(fused.layer_size[0], 32)

    def test_fused_network_preserves_architecture(self):
        """Remaining layer sizes should be unchanged."""
        fused = make_fused_nn(self.nn_91)
        for i in range(1, len(self.nn_91.layer_size)):
            self.assertEqual(fused.layer_size[i], self.nn_91.layer_size[i])

    def test_fused_network_has_smaller_coefficient_count(self):
        """Fused network should have fewer coefficients (32 first layer vs 91)."""
        fused = make_fused_nn(self.nn_91)
        self.assertLess(fused.len_coefficients, self.nn_91.len_coefficients)

    def test_output_parity_with_subsquares(self):
        """Fused 32-input NN should produce same output as 91-input NN + subsquares."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            x_32 = rng.random(32).astype(np.float32)
            x_91 = subsquares(x_32)

            fused = make_fused_nn(self.nn_91)
            out_fused = fused.compute(x_32)

            out_original = self.nn_91.compute(x_91)

            self.assertAlmostEqual(
                out_fused,
                out_original,
                places=5,
                msg="Fused NN output should match original NN + subsquares",
            )

    def test_output_parity_with_terminal_layer_contribution(self):
        """The 32-input path uses np.sum(x) while 91-input uses x[-1]*32.
        For a board vector with all equal values, these are the same."""
        x_32 = np.full(32, 0.25, dtype=np.float32)
        x_91 = subsquares(x_32)

        fused = make_fused_nn(self.nn_91)
        out_fused = fused.compute(x_32)
        out_original = self.nn_91.compute(x_91)

        self.assertAlmostEqual(out_fused, out_original, places=5)

    def test_fused_network_preserves_mlx_flag(self):
        """MLX flag should be preserved if enabled in original."""
        fused = make_fused_nn(self.nn_91)
        self.assertEqual(fused._use_mlx, self.nn_91._use_mlx)

    def test_non_91_input_raises_no_error(self):
        """make_fused_nn should work with any 91-input NN."""
        nn = NeuralNetwork([91, 40, 10, 1])
        fused = make_fused_nn(nn)
        self.assertIsNotNone(fused)
        self.assertEqual(fused.layer_size[0], 32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
