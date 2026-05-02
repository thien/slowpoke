"""Comprehensive tests for the NeuralNetwork evaluator."""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.evaluator.neural import NeuralNetwork
import numpy as np


class TestNeuralNetworkInit(unittest.TestCase):
    """Test neural network initialisation."""

    def test_default_layer_list(self):
        """Default should be [32, 40, 10, 1]."""
        nn = NeuralNetwork()
        self.assertEqual(nn.layer_size, [32, 40, 10, 1])

    def test_custom_layer_list(self):
        """Custom layer sizes should be respected."""
        nn = NeuralNetwork(layer_list=[16, 8, 4, 1])
        self.assertEqual(nn.layer_size, [16, 8, 4, 1])

    def test_layer_count_property(self):
        """num_layers should equal len(layer_size)."""
        nn = NeuralNetwork([32, 20, 10, 1])
        self.assertEqual(nn.num_layers, 4)

    def test_hidden_layer_count_property(self):
        """num_hidden_layers should be num_layers - 2."""
        nn = NeuralNetwork([32, 20, 10, 1])
        self.assertEqual(nn.num_hidden_layers, 2)

    def test_init_creates_layers(self):
        """init_layers should be called during __init__."""
        nn = NeuralNetwork([32, 20, 10, 1])
        self.assertEqual(len(nn.layers), 4)
        self.assertEqual(len(nn.layers[0]), 32)

    def test_init_creates_weights(self):
        """init_weights should be called during __init__."""
        nn = NeuralNetwork([32, 20, 10, 1])
        self.assertEqual(len(nn.weights), 3)

    def test_init_creates_biases(self):
        """init_biases should be called during __init__."""
        nn = NeuralNetwork([32, 20, 10, 1])
        self.assertEqual(len(nn.biases), 3)

    def test_init_sets_coefficient_count(self):
        """len_coefficients should be correctly set."""
        nn = NeuralNetwork([32, 40, 10, 1])
        expected = 32 * 40 + 40 + 40 * 10 + 10 + 10 * 1 + 1
        self.assertEqual(nn.len_coefficients, expected)

    def test_91_input_detection(self):
        """layer_size[0] == 91 should set _input_size_91."""
        nn = NeuralNetwork([91, 40, 10, 1])
        self.assertTrue(nn._input_size_91)

    def test_32_input_detection(self):
        """layer_size[0] == 32 should not set _input_size_91."""
        nn = NeuralNetwork([32, 40, 10, 1])
        self.assertFalse(nn._input_size_91)


class TestNeuralNetworkWeights(unittest.TestCase):
    """Test weight and bias operations."""

    def test_weight_shapes(self):
        """Each weight matrix should match layer sizes."""
        nn = NeuralNetwork([32, 20, 10, 1])
        self.assertEqual(nn.weights[0].shape, (32, 20))
        self.assertEqual(nn.weights[1].shape, (20, 10))
        self.assertEqual(nn.weights[2].shape, (10, 1))

    def test_bias_shapes(self):
        """Each bias vector should match output layer size."""
        nn = NeuralNetwork([32, 20, 10, 1])
        self.assertEqual(len(nn.biases[0]), 20)
        self.assertEqual(len(nn.biases[1]), 10)
        self.assertEqual(len(nn.biases[2]), 1)

    def test_get_all_coefficients_returns_flat_array(self):
        """get_all_coefficients should return a 1D array."""
        nn = NeuralNetwork([32, 20, 10, 1])
        coeffs = nn.get_all_coefficients()
        self.assertEqual(len(coeffs.shape), 1)

    def test_get_all_coefficients_length_matches(self):
        """Length of flattened coefficients should match len_coefficients."""
        nn = NeuralNetwork([32, 20, 10, 1])
        coeffs = nn.get_all_coefficients()
        self.assertEqual(len(coeffs), nn.len_coefficients)

    def test_load_coefficients_restores_state(self):
        """Loading coefficients should restore the exact same state."""
        nn = NeuralNetwork([32, 20, 10, 1])
        coeffs = nn.get_all_coefficients()

        nn2 = NeuralNetwork([32, 20, 10, 1])
        nn2.load_coefficients(coeffs)

        np.testing.assert_allclose(
            nn.get_all_coefficients(), nn2.get_all_coefficients(), rtol=1e-5
        )

    def test_load_coefficients_updates_weights(self):
        """Loading should update weight values."""
        nn = NeuralNetwork([32, 5, 1])
        nn.weights[0].copy()
        new_coeffs = nn.get_all_coefficients().copy()
        new_coeffs[: 32 * 5] = 0.5  # Set first layer weights to 0.5
        nn.load_coefficients(new_coeffs)
        self.assertTrue(np.all(nn.weights[0] == 0.5))

    def test_load_invalid_coefficient_count_raises(self):
        """Loading wrong number of coefficients should raise ValueError."""
        nn = NeuralNetwork([32, 20, 10, 1])
        wrong = np.zeros(50)
        with self.assertRaises(ValueError):
            nn.load_coefficients(wrong)

    def test_weights_normalised_range(self):
        """Weights should be normalised to approximate [-0.2, 0.2]."""
        nn = NeuralNetwork([32, 100, 1])
        self.assertTrue(np.all(nn.weights[0] >= -0.21))
        self.assertTrue(np.all(nn.weights[0] <= 0.21))


class TestNeuralNetworkCompute(unittest.TestCase):
    """Test forward pass computation."""

    def test_compute_returns_float_for_single_output(self):
        """Output layer with 1 node should return Python float."""
        nn = NeuralNetwork([32, 20, 10, 1])
        x = np.random.random(32).astype(np.float32)
        result = nn.compute(x)
        self.assertIsInstance(result, float)

    def test_compute_returns_array_for_multi_output(self):
        """Output layer with >1 nodes should return array."""
        nn = NeuralNetwork([32, 20, 10, 3])
        x = np.random.random(32).astype(np.float32)
        result = nn.compute(x)
        self.assertEqual(len(result), 3)

    def test_compute_32_input_shape(self):
        """32-input network should accept 32-element vectors."""
        nn = NeuralNetwork([32, 20, 10, 1])
        x = np.random.random(32).astype(np.float32)
        result = nn.compute(x)
        self.assertIsInstance(result, float)

    def test_compute_91_input_shape(self):
        """91-input network should accept 91-element vectors."""
        nn = NeuralNetwork([91, 40, 10, 1])
        x = np.random.random(91).astype(np.float32)
        result = nn.compute(x)
        self.assertIsInstance(result, float)

    def test_compute_terminal_contribution_32(self):
        """32-input path adds np.sum(x) as terminal layer contribution."""
        nn = NeuralNetwork([32, 40, 10, 1])
        x = np.full(32, 2.0, dtype=np.float32)
        # With all weights = 0, the contribution is just 0 + np.sum(x) = 64.0
        for w in nn.weights:
            w.fill(0)
        for b in nn.biases:
            b.fill(0)
        result = nn.compute(x)
        self.assertAlmostEqual(result, 64.0, places=4)

    def test_compute_terminal_contribution_91(self):
        """91-input path adds x[-1]*32 as terminal layer contribution."""
        nn = NeuralNetwork([91, 40, 10, 1])
        x = np.full(91, 2.0, dtype=np.float32)
        for w in nn.weights:
            w.fill(0)
        for b in nn.biases:
            b.fill(0)
        # x[-1] = 2.0, so contribution = 2.0 * 32 = 64.0
        result = nn.compute(x)
        self.assertAlmostEqual(result, 64.0, places=4)

    def test_compute_deterministic(self):
        """Same input should always produce same output."""
        nn = NeuralNetwork([32, 20, 10, 1])
        x = np.random.random(32).astype(np.float32)
        r1 = nn.compute(x)
        r2 = nn.compute(x)
        self.assertEqual(r1, r2)


class TestNeuralNetworkActivations(unittest.TestCase):
    """Test activation functions."""

    def test_tanh(self):
        """tanh should match numpy."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = np.tanh(x)
        np.testing.assert_allclose(NeuralNetwork.tanh(x), expected)

    def test_relu_positive(self):
        """ReLU should pass positive values through."""
        x = np.array([1.0, 2.0, 3.0])
        result = NeuralNetwork.relu(x.copy())
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_relu_negative_clamped(self):
        """ReLU should clamp negative values to 0."""
        x = np.array([-1.0, 0.0, 3.0])
        result = NeuralNetwork.relu(x.copy())
        np.testing.assert_array_equal(result, [0.0, 0.0, 3.0])

    def test_crelu_negative_clamped(self):
        """CReLU should clamp values below -1 to -1."""
        x = np.array([-2.0, -1.0, 0.0, 1.0])
        result = NeuralNetwork.crelu(x.copy())
        np.testing.assert_array_equal(result, [-1.0, -1.0, 0.0, 1.0])

    def test_softmax_sums_to_one(self):
        """Softmax output should sum to 1.0."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = NeuralNetwork.softmax(x)
        self.assertAlmostEqual(np.sum(result), 1.0)

    def test_softmax_all_non_negative(self):
        """All softmax outputs should be non-negative."""
        x = np.array([-100, 0, 100])
        result = NeuralNetwork.softmax(x)
        self.assertTrue(np.all(result >= 0))

    def test_nonlinear_function_delegates_to_tanh(self):
        """nonlinear_function should return tanh of input."""
        val = np.array([-0.5, 0.0, 0.5])
        nn = NeuralNetwork([32, 1])
        result = nn.nonlinear_function(val)
        expected = NeuralNetwork.tanh(val)
        np.testing.assert_allclose(result, expected)

    def test_normalise_vectors_range(self):
        """normalise_vectors should map [0,1] to [-0.2, 0.2]."""
        v = np.array([0.0, 0.5, 1.0])
        result = NeuralNetwork.normalise_vectors(v)
        expected = (v - 0.5) * 0.4
        np.testing.assert_allclose(result, expected)


class TestNeuralNetworkComputeBatch(unittest.TestCase):
    """Test batch computation."""

    def test_compute_batch_returns_correct_count(self):
        """Batch output should have one result per input."""
        nn = NeuralNetwork([32, 20, 10, 1])
        inputs = [np.random.random(32).astype(np.float32) for _ in range(5)]
        results = nn.compute_batch(inputs)
        self.assertEqual(len(results), 5)

    def test_compute_batch_matches_individual(self):
        """Batch results should match individual compute calls."""
        nn = NeuralNetwork([32, 20, 10, 1])
        inputs = [np.random.random(32).astype(np.float32) for _ in range(3)]
        batch_results = nn.compute_batch(inputs)
        for i, x in enumerate(inputs):
            individual = nn.compute(x)
            self.assertAlmostEqual(batch_results[i], individual, places=5)

    def test_compute_batch_empty(self):
        """Empty batch should return empty array."""
        nn = NeuralNetwork([32, 1])
        results = nn.compute_batch([])
        self.assertEqual(len(results), 0)

    def test_compute_batch_with_subsquares_input(self):
        """Batch should work with 91-input networks too."""
        nn = NeuralNetwork([91, 40, 10, 1])
        inputs = [np.random.random(91).astype(np.float32) for _ in range(3)]
        results = nn.compute_batch(inputs)
        self.assertEqual(len(results), 3)
        for i, x in enumerate(inputs):
            self.assertAlmostEqual(results[i], nn.compute(x), places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
