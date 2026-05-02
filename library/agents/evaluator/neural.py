from __future__ import annotations

import math
from typing import Any, List, Optional, Union

import numpy as np

try:
    import agents.evaluator.subsquares as subsquares
except ImportError:
    from library.agents.evaluator import subsquares

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    mx = None
    MLX_AVAILABLE = False


def showVector(v: np.ndarray, dec: int) -> None:
    fmt = "%." + str(dec) + "f"  # like %.4f
    for i in range(len(v)):
        x = v[i]
        if x >= 0.0:
            print(" ", end="")
        print(fmt % x + "  ", end="")


class NeuralNetwork:
    __slots__ = (
        "layer_size",
        "num_layers",
        "num_hidden_layers",
        "layers",
        "weights",
        "biases",
        "len_coefficients",
        "rebuildCoefficents",
        "rnd",
        "ravel",
        "_use_mlx",
        "_mx_weights",
        "_mx_biases",
        "_mx_compiled_forward",
        "_last_input_size",
        "_input_size_91",
    )

    def __init__(
        self, layer_list: Optional[List[int]] = None, use_mlx: bool = False
    ) -> None:
        self.layer_size = layer_list if layer_list is not None else [32, 40, 10, 1]
        self.num_layers = len(self.layer_size)
        self.num_hidden_layers = self.num_layers - 2
        self.layers = []
        self.weights = []
        self.biases = []
        self.len_coefficients = 0
        self.rebuildCoefficents = None
        self.rnd = np.random.seed()
        self._use_mlx = use_mlx and MLX_AVAILABLE
        self._mx_weights = None
        self._mx_biases = None
        self._mx_compiled_forward = None
        self._last_input_size = 0
        self._input_size_91 = self.layer_size[0] == 91
        # initiate layers
        self.init_layers()
        self.init_weights()
        self.init_biases()

        # Initialize MLX weights if requested
        if self._use_mlx:
            self._init_mlx_weights()

    def _init_mlx_weights(self) -> None:
        """Convert numpy weights to MLX arrays for GPU evaluation."""
        if not MLX_AVAILABLE or mx is None:
            self._use_mlx = False
            return

        self._mx_weights = [mx.array(w.astype(np.float32)) for w in self.weights]
        self._mx_biases = [mx.array(b.astype(np.float32)) for b in self.biases]

    def init_layers(self) -> None:
        for i in self.layer_size:
            nodes = np.zeros(shape=[i], dtype=np.float32)
            self.layers.append(nodes)

    def init_weights(self) -> None:
        for i in range(self.num_layers - 1):
            inputNodes = self.layer_size[i]
            outputNodes = self.layer_size[i + 1]
            self.len_coefficients += inputNodes * outputNodes
            weights = np.random.random_sample([inputNodes, outputNodes])
            weights = self.normalise_vectors(weights)
            self.weights.append(weights)

    def init_biases(self) -> None:
        for i in range(self.num_layers - 1):
            biasNodes = self.layer_size[i + 1]
            self.len_coefficients += biasNodes
            biases = np.random.random_sample(biasNodes)
            biases = self.normalise_vectors(biases)
            self.biases.append(biases)

    def get_all_coefficients(self) -> np.ndarray:
        """Optimised: collect all weights and biases in one pass."""
        arrays = []
        for w in self.weights:
            arrays.append(np.ravel(w))
        for b in self.biases:
            arrays.append(np.ravel(b))
        return np.concatenate(arrays)

    def load_coefficients(self, ravelled: np.ndarray) -> bool:
        if len(ravelled) != self.len_coefficients:
            raise ValueError("The number of coefficents do not match.")
        # calculate number of weights to split array from
        totalNumWeights = 0
        for i in self.weights:
            totalNumWeights += i.shape[0] * i.shape[1]

        # rebuild weights
        weights = ravelled[:totalNumWeights]

        weight_inc = 0
        for i in range(len(self.weights)):
            # get the dimensions of i
            resolution = self.weights[i].shape[0] * self.weights[i].shape[1]
            sub_weight = weights[weight_inc : weight_inc + resolution]
            # Reshape to (input_nodes, output_nodes) - store as ndarray, not matrix
            self.weights[i] = sub_weight.reshape(self.weights[i].shape).astype(
                np.float32
            )
            weight_inc += resolution

        # rebuild biases
        biases = ravelled[totalNumWeights:]

        biases_inc = 0
        for i in range(len(self.biases)):
            resolution = self.biases[i].shape[0]
            sub_biases = biases[biases_inc : biases_inc + resolution]
            biases_inc += resolution
            self.biases[i] = sub_biases.astype(np.float32)

        # Sync MLX weights if using MLX
        if self._use_mlx:
            self._init_mlx_weights()

        return True

    def compute(self, x: np.ndarray) -> Union[float, np.ndarray]:
        """
        Optimised forward pass through the neural network.
        Returns Python float for compatibility with existing code.
        Fully vectorised - no loops over neurons.
        """
        current = x

        # Forward pass through all hidden layers
        for n in range(self.num_layers - 2):
            # Vectorised: matrix multiply + bias in one step
            current = np.tanh(self.weights[n].T.dot(current) + self.biases[n])

        # Final layer
        current = self.weights[-1].T.dot(current) + self.biases[-1]

        # Add input contribution for terminal layer (special heuristic)
        if x.size == 91:
            current = current + x[-1] * 32
        else:
            current = current + np.sum(x)

        # Return scalar value (Python float for compatibility)
        return float(current[0]) if current.size == 1 else current

    def compute_mlx(self, x: np.ndarray) -> Any:
        """
        MLX-accelerated forward pass that returns mx.array.
        For native MLX tree search - keeps values on GPU.
        """
        if not self._use_mlx or not MLX_AVAILABLE:
            # Fallback to numpy then convert
            result = self.compute(x)
            return np.array([result], dtype=np.float32)

        # Convert input to MLX array
        x_arr = np.asarray(x, dtype=np.float32)
        mx_x = mx.array(x_arr)

        # JIT-compiled forward pass
        if self._mx_compiled_forward is None:

            def forward_fn(inputs):
                current = inputs
                for n in range(self.num_layers - 2):
                    current = mx.tanh(
                        mx.matmul(current, self._mx_weights[n]) + self._mx_biases[n]
                    )
                current = mx.matmul(current, self._mx_weights[-1]) + self._mx_biases[-1]
                return current

            self._mx_compiled_forward = mx.compile(forward_fn)

        result = self._mx_compiled_forward(mx_x[None, :])  # Add batch dimension
        mx.eval(result)

        # Add input contribution (MLX version)
        if x_arr.size == 91:
            result = result + x_arr[-1] * 32
        else:
            result = result + mx.sum(mx_x)

        return result[0]  # Return scalar mx.array

    def compute_batch_mlx(self, batch_inputs: List[np.ndarray]) -> Any:
        """
        Batched MLX evaluation that returns mx.array.
        For native tree search with accumulated positions.
        """
        if not self._use_mlx or not MLX_AVAILABLE or len(batch_inputs) == 0:
            # Fallback
            return np.array([self.compute(x) for x in batch_inputs], dtype=np.float32)

        # Stack inputs as batch (N, input_size)
        batch_np = np.array(batch_inputs, dtype=np.float32)
        mx_batch = mx.array(batch_np)

        # Vectorized batch forward pass
        current = mx_batch
        for n in range(self.num_layers - 2):
            current = mx.tanh(
                mx.matmul(current, self._mx_weights[n]) + self._mx_biases[n]
            )
        current = mx.matmul(current, self._mx_weights[-1]) + self._mx_biases[-1]

        mx.eval(current)

        # Add input contribution in MLX
        if len(batch_inputs[0]) == 91:
            # x[-1] contribution for each input
            sums = mx.array([float(x[-1] * 32) for x in batch_inputs])
            current = current + sums[:, None]
        else:
            # Vectorized sum for non-91 case
            batch_arr = mx.array(batch_np)
            sums = mx.sum(batch_arr, axis=1)
            current = current + sums[:, None]

        return current.flatten()  # Returns mx.array

    def compute_batch(self, batch_inputs: List[np.ndarray]) -> np.ndarray:
        """
        Batch evaluation returning NumPy array for backward compatibility.
        """
        if not self._use_mlx or not MLX_AVAILABLE:
            return np.array([self.compute(x) for x in batch_inputs])

        mx_results = self.compute_batch_mlx(batch_inputs)
        return np.array(mx_results)

    @staticmethod
    def subsquares(x: np.ndarray) -> np.ndarray:
        """Calculate subsquare features from 32-element board vector."""
        return subsquares.subsquares(x)

    @staticmethod
    def normalise_vectors(vector: np.ndarray) -> np.ndarray:
        """Normalise to a range from -0.2 to 0.2."""
        return (vector - 0.5) * 0.4

    def nonlinear_function(self, val: np.ndarray) -> np.ndarray:
        """Apply the nonlinear function (tanh) to values."""
        return self.tanh(val)

    @staticmethod
    def tanh(val: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent activation."""
        return np.tanh(val)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation: clamp negatives to 0."""
        x = x.copy()
        x[x < 0] = 0
        return x

    @staticmethod
    def crelu(x: np.ndarray) -> np.ndarray:
        """Capped ReLU: clamp values below -1 to -1."""
        x = x.copy()
        x[x < -1] = -1
        return x

    @staticmethod
    def softmax(oSums: np.ndarray) -> np.ndarray:
        """
        Function to softmax output values.
        """
        result = np.zeros(shape=[len(oSums)], dtype=np.float32)
        m = max(oSums)
        divisor = 0.0
        for k in range(len(oSums)):
            divisor += math.exp(oSums[k] - m)
        for k in range(len(result)):
            result[k] = math.exp(oSums[k] - m) / divisor
        return result


if __name__ == "__main__":
    # Insert checkerboard.
    x = np.array(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        ],
        dtype=np.float32,
    )

    # standard neural network
    inputs = [32, 40, 10, 1]
    nn = NeuralNetwork(inputs)

    # subsquare neural network
    subsq = [91, 40, 10, 1]
    nn2 = NeuralNetwork(subsq)

    import datetime

    # print("Regular Neural Network")
    start = datetime.datetime.now().timestamp()

    yValues = nn.compute(x)
    print("RNN:", yValues)
    end = datetime.datetime.now().timestamp() - start
    # print("RNN Time:",end)

    x = nn.subsquares(x)

    # print("Subsquare Processed Neural Network")
    mu = datetime.datetime.now().timestamp()

    # print(x.size)
    yValues = nn2.compute(x)
    print("SNN:", yValues)
    end2 = datetime.datetime.now().timestamp() - start
    # print("SNN Time:",end2)

    # print("\nOutput values are: ")
    # showVector(yValues, 4)

    print("Time Multiplier:", end2 / end)
