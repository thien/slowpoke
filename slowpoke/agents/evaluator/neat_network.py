"""NEAT forward pass — evaluates a Genome on an input vector.

Includes CompiledNEAT: compiles a Genome into sequential MLX operations
for GPU-accelerated evaluation (single and batched).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from slowpoke.agents.evaluator.genome import Genome
from slowpoke.core.constants import MLX_AVAILABLE, mx


class CompiledNEAT:
    """A Genome compiled to sequential (batched) MLX operations.

    Topology is fixed at compile time. Each non-input node becomes one
    sequential step: gather sources → multiply by weights → sum → tanh.
    For batch evaluation, each step processes all batch items simultaneously.
    """

    __slots__ = (
        "_input_count",
        "_hidden_output_count",
        "_max_conns",
        "_np_biases",
        "_np_srcs",
        "_np_ws",
        "_mx_biases",
        "_mx_srcs",
        "_mx_ws",
    )

    def __init__(self) -> None:
        self._input_count: int = 0
        self._hidden_output_count: int = 0
        self._max_conns: int = 0
        self._np_biases: np.ndarray = np.array([], dtype=np.float32)
        self._np_srcs: List[np.ndarray] = []
        self._np_ws: List[np.ndarray] = []
        self._mx_biases: Optional[Any] = None
        self._mx_srcs: Optional[Any] = None
        self._mx_ws: Optional[Any] = None

    def compile(self, genome: Genome) -> None:
        """Compile a Genome into fixed sequential evaluation structure.

        Args:
            genome: A Genome instance with nodes and connections.
        """
        enabled = [c for c in genome.connections.values() if c.enabled]

        # Map input node IDs → positions 0..N-1
        input_ids = sorted(nid for nid, n in genome.nodes.items() if n.kind == "input")
        self._input_count = len(input_ids)
        input_map = {nid: i for i, nid in enumerate(input_ids)}

        # Non-input nodes in topological order (by node ID, guaranteed
        # feed-forward by NEAT's lower→higher constraint)
        non_input = sorted(nid for nid, n in genome.nodes.items() if n.kind != "input")
        self._hidden_output_count = len(non_input)
        non_input_pos = {nid: i for i, nid in enumerate(non_input)}

        # Build per-node source indices and weights
        biases: List[float] = []
        srcs_list: List[List[int]] = []
        ws_list: List[List[float]] = []

        for nid in non_input:
            node = genome.nodes[nid]
            biases.append(node.bias)
            conns = [c for c in enabled if c.to_node == nid]
            srcs: List[int] = []
            vals: List[float] = []
            for c in conns:
                if c.from_node in input_map:
                    srcs.append(input_map[c.from_node])
                else:
                    pos = non_input_pos.get(c.from_node)
                    if pos is not None:
                        srcs.append(self._input_count + pos)
                    else:
                        continue  # disabled upstream node
                vals.append(c.weight)
            srcs_list.append(srcs)
            ws_list.append(vals)

        # Pad to max_conns for rectangular MLX tensors
        self._max_conns = max((len(s) for s in srcs_list), default=0)

        # NumPy buffers (for fallback)
        self._np_biases = np.array(biases, dtype=np.float32)
        self._np_srcs = [
            np.array(s + [0] * (self._max_conns - len(s)), dtype=np.int32)
            for s in srcs_list
        ]
        self._np_ws = [
            np.array(w + [0.0] * (self._max_conns - len(w)), dtype=np.float32)
            for w in ws_list
        ]

        # MLX buffers
        self._mx_biases = None
        self._mx_srcs = None
        self._mx_ws = None
        if MLX_AVAILABLE and mx is not None:
            self._mx_biases = mx.array(biases, mx.float32)
            padded = np.array(
                [s + [0] * (self._max_conns - len(s)) for s in srcs_list],
                dtype=np.int32,
            )
            self._mx_srcs = mx.array(padded, mx.int32)
            padded_ws = np.array(
                [w + [0.0] * (self._max_conns - len(w)) for w in ws_list],
                dtype=np.float32,
            )
            self._mx_ws = mx.array(padded_ws, mx.float32)

    # ── Single evaluation ──

    def evaluate(self, inputs: np.ndarray) -> float:
        """Evaluate the compiled genome on a single input vector.

        Args:
            inputs: 32-element float32 array.

        Returns:
            Single float output.
        """
        if self._mx_biases is not None and MLX_AVAILABLE:
            return self._evaluate_mlx(inputs)
        return self._evaluate_numpy(inputs)

    def _evaluate_numpy(self, inputs: np.ndarray) -> float:
        total_nodes = self._input_count + self._hidden_output_count
        acts = np.zeros(total_nodes, dtype=np.float32)
        acts[: self._input_count] = inputs
        for i in range(self._hidden_output_count):
            if self._max_conns == 0:
                acts[self._input_count + i] = np.tanh(self._np_biases[i])
            else:
                sel = acts[self._np_srcs[i]]
                val = np.tanh(np.sum(sel * self._np_ws[i]) + self._np_biases[i])
                acts[self._input_count + i] = val
        return float(acts[-1])

    def _evaluate_mlx(self, inputs: np.ndarray) -> float:
        total_nodes = self._input_count + self._hidden_output_count
        acts = mx.zeros(total_nodes, mx.float32)
        acts[: self._input_count] = mx.array(inputs, mx.float32)
        for i in range(self._hidden_output_count):
            if self._max_conns == 0:
                val = mx.tanh(self._mx_biases[i])
            else:
                gathered = mx.take(acts, self._mx_srcs[i])
                val = mx.tanh(mx.sum(gathered * self._mx_ws[i]) + self._mx_biases[i])
            acts[self._input_count + i] = val
        mx.eval(acts)
        return float(acts[-1].item())

    # ── Batch evaluation ──

    def evaluate_batch_mlx(self, batch_inputs: np.ndarray) -> Any:
        """Evaluate the compiled genome on a batch of inputs using MLX.

        Args:
            batch_inputs: [batch_size, input_count] float32 array.

        Returns:
            mx.array of shape [batch_size] with output values.
        """
        if self._mx_srcs is None or not MLX_AVAILABLE:
            # Fallback: evaluate each item individually
            return mx.array(
                [self._evaluate_numpy(x) for x in batch_inputs],
                mx.float32,
            )

        batch = mx.array(batch_inputs, mx.float32)
        B = batch.shape[0]
        total_nodes = self._input_count + self._hidden_output_count
        acts = mx.zeros((B, total_nodes), mx.float32)
        acts[:, : self._input_count] = batch

        for i in range(self._hidden_output_count):
            if self._max_conns == 0:
                val = mx.tanh(mx.broadcast_to(self._mx_biases[i], (B,)))
            else:
                # acts: [B, total]; src_indices[i]: [max_conns]
                gathered = mx.take(acts, self._mx_srcs[i], axis=1)
                # gathered: [B, max_conns]
                weighted = mx.sum(gathered * self._mx_ws[i], axis=-1)
                val = mx.tanh(weighted + self._mx_biases[i])
            acts[:, self._input_count + i] = val

        mx.eval(acts)
        return acts[:, -1]  # [B]


class NEATNetwork:
    """Forward pass through a Genome, matching NeuralNetwork.compute() interface.

    Relies on Genome.build_cache() for topology precomputation to avoid
    rebuilding adjacency structures on every evaluation.
    """

    @staticmethod
    def compute(genome: Genome, inputs: np.ndarray) -> float:
        """Evaluate genome on a flat input array.

        Args:
            genome: A Genome instance (cache will be built lazily if absent).
            inputs: 32-element float32 array (board position).

        Returns:
            Single float output value, in [-1, 1].
        """
        if genome._cache is None:
            genome.build_cache()
        cache = genome._cache

        activations: Dict[int, float] = {}

        # Assign inputs from cached input_ids
        input_ids = cache["input_ids"]
        if not input_ids:
            return 0.0
        for i, nid in enumerate(input_ids):
            activations[nid] = float(inputs[i]) if i < len(inputs) else 0.0

        # Process hidden + output nodes in topological order
        incoming = cache["incoming"]
        genome_nodes = genome.nodes
        for nid in cache["hidden_and_output"]:
            node = genome_nodes[nid]
            conns = incoming.get(nid, [])
            if not conns:
                activations[nid] = np.tanh(node.bias)
                continue
            total = node.bias
            for c in conns:
                total += activations.get(c.from_node, 0.0) * c.weight
            activations[nid] = np.tanh(total)

        # Return output node value
        output_ids = cache["output_ids"]
        return float(activations.get(output_ids[0], 0.0)) if output_ids else 0.0

    @staticmethod
    def compute_batch(genome: Genome, batch_inputs: List[np.ndarray]) -> np.ndarray:
        """Evaluate genome on a batch of inputs.

        Args:
            genome: A Genome instance.
            batch_inputs: List of 32-element input arrays.

        Returns:
            Array of output values.
        """
        return np.array(
            [NEATNetwork.compute(genome, x) for x in batch_inputs],
            dtype=np.float32,
        )
