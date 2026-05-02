"""Subsquare feature extraction for checkers board positions."""

import numpy as np

BOARD = np.array([
    [-1,  0, -1,  1, -1,  2, -1,  3],
    [ 4, -1,  5, -1,  6, -1,  7, -1],
    [-1,  8, -1,  9, -1, 10, -1, 11],
    [12, -1, 13, -1, 14, -1, 15, -1],
    [-1, 16, -1, 17, -1, 18, -1, 19],
    [20, -1, 21, -1, 22, -1, 23, -1],
    [-1, 24, -1, 25, -1, 26, -1, 27],
    [28, -1, 29, -1, 30, -1, 31, -1],
], dtype=np.int32)


def _build_subsquare_matrix() -> np.ndarray:
    """Precompute the 91x32 matrix that maps 32 board squares to 91 subsquare features."""
    M = np.zeros((91, 32), dtype=np.float32)
    row = 0
    for kernel in range(3, 9):
        for j in range(0, 8 - kernel + 1):
            for i in range(0, 8 - kernel + 1):
                indices = []
                for ky in range(kernel):
                    for kx in range(kernel):
                        val = BOARD[j + ky, i + kx]
                        if val != -1:
                            indices.append(val)
                indices.sort()
                scale = 1.0 / len(indices)
                for idx in indices:
                    M[row, idx] = scale
                row += 1
    assert row == 91, f"Expected 91 subsquares, got {row}"
    return M


SUBSQUARE_MATRIX: np.ndarray = _build_subsquare_matrix()


def subsquares(x: np.ndarray) -> np.ndarray:
    """Calculate 91-element subsquare feature vector from a 32-element board array.

    Each subsquare is the average of board values within a kxk window (k=3..8).
    Uses a single BLAS matrix-vector multiply instead of 600+ scalar Python ops.

    Args:
        x: 32-element numpy array (board position representation).

    Returns:
        91-element numpy array (subsquare features).
    """
    return SUBSQUARE_MATRIX @ x


def make_fused_nn(nn_91: "NeuralNetwork") -> "NeuralNetwork":
    """Convert a [91, 40, 10, 1] NeuralNetwork to a [32, 40, 10, 1] fused network.

    Fuses the subsquares matrix into the first layer weights:
      W1_fused = M.T @ W1_orig

    Args:
        nn_91: NeuralNetwork with layer_size[0] == 91.

    Returns:
        NeuralNetwork with layer_size = [32, 40, 10, 1],
        MLX state preserved if enabled.
    """
    from agents.evaluator.neural import NeuralNetwork

    layer_32 = [32] + nn_91.layer_size[1:]
    nn_32 = NeuralNetwork(layer_32, use_mlx=nn_91._use_mlx)

    nn_32.weights[0] = (SUBSQUARE_MATRIX.T @ nn_91.weights[0]).astype(np.float32)

    for i in range(1, len(nn_91.weights)):
        nn_32.weights[i] = nn_91.weights[i].copy()
    for i in range(len(nn_91.biases)):
        nn_32.biases[i] = nn_91.biases[i].copy()

    total = sum(w.size for w in nn_32.weights) + sum(b.size for b in nn_32.biases)
    nn_32.lenCoefficents = total

    if nn_32._use_mlx:
        nn_32._init_mlx_weights()

    return nn_32
