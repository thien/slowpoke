import numpy as np

"""
Board model for generating subsquare index sets.

The checkers board (8×8, only dark squares playable):

  [-1,  0, -1,  1, -1,  2, -1,  3],
  [ 4, -1,  5, -1,  6, -1,  7, -1],
  [-1,  8, -1,  9, -1, 10, -1, 11],
  [12, -1, 13, -1, 14, -1, 15, -1],
  [-1, 16, -1, 17, -1, 18, -1, 19],
  [20, -1, 21, -1, 22, -1, 23, -1],
  [-1, 24, -1, 25, -1, 26, -1, 27],
  [28, -1, 29, -1, 30, -1, 31, -1]

Each subsquare = average of x[indices] for a k×k window (k=3..8).
91 subsquares total: 36 (3×3 windows) + 25 (4×4) + 16 (5×5) + 9 (6×6) + 4 (7×7) + 1 (8×8).
"""

# Build the 91×32 subsquare matrix programmatically
def _build_subsquare_matrix():
    """Precompute the 91×32 matrix that maps 32 board squares → 91 subsquare features."""
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

    M = np.zeros((91, 32), dtype=np.float32)
    row = 0
    for kernel in range(3, 9):  # 3×3 through 8×8
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

SUBSQUARE_MATRIX = _build_subsquare_matrix()

def subsquares(x):
    """
    Calculate 91-element subsquare feature vector from a 32-element board array.

    Each subsquare is the average of board values within a k×k window (k=3..8).
    Uses a single BLAS matrix-vector multiply instead of 600+ scalar Python ops.

    Args:
        x: 32-element numpy array (board position representation)

    Returns:
        91-element numpy array (subsquare features)
    """
    return SUBSQUARE_MATRIX @ x


def make_fused_nn(nn_91):
    """
    Convert a [91, 40, 10, 1] NeuralNetwork to a [32, 40, 10, 1] fused network.

    Fuses the subsquares matrix into the first layer weights:
      W1_fused = M.T @ W1_orig

    This eliminates the subsquares computation entirely — the [32] input goes
    directly into the NN's first layer, producing the same output as
    subsquares(x) → nn_original.compute().

    The NN output is bit-identical because:
      - x_91[-1] * 32  (91-case input contribution) == np.sum(x_32)  (32-case)
      - The 91-case always uses the else branch when x.size != 91 ✓

    Args:
        nn_91: NeuralNetwork with layer_size[0] == 91

    Returns:
        NeuralNetwork with layer_size = [32, 40, 10, 1],
        MLX state preserved if enabled.
    """
    # Defer import to avoid circular dependency
    from agents.evaluator.neural import NeuralNetwork

    layer_32 = [32] + nn_91.layer_size[1:]
    nn_32 = NeuralNetwork(layer_32, use_mlx=nn_91._use_mlx)

    # Fuse first layer: W1_fused = M.T @ W1_orig
    nn_32.weights[0] = (SUBSQUARE_MATRIX.T @ nn_91.weights[0]).astype(np.float32)

    # Copy remaining layers unchanged
    for i in range(1, len(nn_91.weights)):
        nn_32.weights[i] = nn_91.weights[i].copy()
    for i in range(len(nn_91.biases)):
        nn_32.biases[i] = nn_91.biases[i].copy()

    # Recompute coefficient length (it's smaller now for [32] architecture)
    total = sum(w.size for w in nn_32.weights) + sum(b.size for b in nn_32.biases)
    nn_32.lenCoefficents = total

    # Reinitialize MLX state with fused weights
    if nn_32._use_mlx:
        nn_32._init_mlx_weights()

    return nn_32
