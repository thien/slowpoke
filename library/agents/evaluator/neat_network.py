"""NEAT forward pass — evaluates a Genome on an input vector."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from agents.evaluator.genome import Genome


class NEATNetwork:
    """Forward pass through a Genome, matching NeuralNetwork.compute() interface."""

    @staticmethod
    def compute(genome: Genome, inputs: np.ndarray) -> float:
        """Evaluate genome on a flat input array.

        Args:
            genome: A Genome instance.
            inputs: 32-element float32 array (board position).

        Returns:
            Single float output value, in [-1, 1].
        """
        # Map node_id → activation value
        activations: Dict[int, float] = {}

        # Find input nodes (sorted by id for deterministic input mapping)
        input_ids = sorted(nid for nid, n in genome.nodes.items() if n.kind == "input")
        if not input_ids:
            return 0.0

        # Assign inputs
        for i, nid in enumerate(input_ids):
            if i < len(inputs):
                activations[nid] = float(inputs[i])
            else:
                activations[nid] = 0.0

        # Collect non-input nodes
        hidden_and_output = [
            nid for nid, n in genome.nodes.items() if n.kind != "input"
        ]

        # Topological sort: order by node ID (a simple DAG heuristic)
        # For feed-forward networks, processing lower IDs first works
        # because we only allow connections from lower→higher IDs.
        sorted_nodes = sorted(hidden_and_output)

        # Build adjacency for quick lookup
        enabled_connections = [c for c in genome.connections.values() if c.enabled]
        # Group by target node
        incoming: Dict[int, List] = {}
        for c in enabled_connections:
            incoming.setdefault(c.to_node, []).append(c)

        # Process in topological order
        for nid in sorted_nodes:
            node = genome.nodes[nid]
            conns = incoming.get(nid, [])
            if not conns:
                # No incoming connections → use bias as activation
                activations[nid] = np.tanh(node.bias)
                continue

            total = node.bias
            for c in conns:
                source_val = activations.get(c.from_node, 0.0)
                total += source_val * c.weight

            activations[nid] = np.tanh(total)

        # Return output node value
        output_ids = sorted(
            nid for nid, n in genome.nodes.items() if n.kind == "output"
        )
        if not output_ids:
            return 0.0

        return float(activations.get(output_ids[0], 0.0))

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
