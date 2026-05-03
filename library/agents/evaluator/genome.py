"""NEAT Genome: nodes, connections, innovation tracking, mutations, crossover."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ── Global innovation counter ──────────────────────────────────────────────────
# Tracks every structural mutation across the entire run.
# Persists across generations so innovation numbers are globally unique.
_next_innovation: int = 0


def _reset_innovation_counter(start: int = 0) -> None:
    global _next_innovation
    _next_innovation = start


def _next_innovation_id(history: Dict[Tuple[int, int], int], from_id: int, to_id: int) -> int:
    """Return a global innovation ID for a connection (from→to).

    Uses a per-run history dict to deduplicate: if the same (from,to)
    was already created in this run, reuses its innovation number.
    """
    global _next_innovation
    key = (from_id, to_id)
    if key in history:
        return history[key]
    nid = _next_innovation
    _next_innovation += 1
    history[key] = nid
    return nid


# ── Node gene ──────────────────────────────────────────────────────────────────

class NodeGene:
    """A single node in the NEAT network."""

    __slots__ = ("node_id", "kind", "bias")

    def __init__(self, node_id: int, kind: str, bias: float = 0.0) -> None:
        self.node_id = node_id
        self.kind = kind  # 'input' | 'hidden' | 'output'
        self.bias = bias


# ── Connection gene ────────────────────────────────────────────────────────────

class ConnectionGene:
    """A single connection between two nodes."""

    __slots__ = ("innovation", "from_node", "to_node", "weight", "enabled")

    def __init__(
        self,
        innovation: int,
        from_node: int,
        to_node: int,
        weight: float = 1.0,
        enabled: bool = True,
    ) -> None:
        self.innovation = innovation
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.enabled = enabled

    def copy(self) -> ConnectionGene:
        return ConnectionGene(
            self.innovation,
            self.from_node,
            self.to_node,
            self.weight,
            self.enabled,
        )


# ── Genome ─────────────────────────────────────────────────────────────────────

class Genome:
    """A full NEAT genome: collection of node genes and connection genes."""

    __slots__ = ("nodes", "connections", "fitness", "innovation_history")

    def __init__(self) -> None:
        self.nodes: Dict[int, NodeGene] = {}
        self.connections: Dict[int, ConnectionGene] = {}  # keyed by innovation
        self.fitness: float = 0.0
        # Local innovation history for this genome's lineage
        self.innovation_history: Dict[Tuple[int, int], int] = {}

    # ── Factory methods ──

    @classmethod
    def minimal(cls, num_inputs: int = 32, num_hidden: int = 4, num_outputs: int = 1) -> Genome:
        """Create a minimal genome with inputs → hidden layer → output.

        All connections are fully connected between adjacent layers.
        Biases are initialised near zero.
        """
        g = cls()
        nid = 0

        # Input nodes
        for i in range(num_inputs):
            g.nodes[nid] = NodeGene(nid, "input", bias=0.0)
            nid += 1
        input_end = nid

        # Hidden nodes
        hidden_start = nid
        for _ in range(num_hidden):
            g.nodes[nid] = NodeGene(nid, "hidden", bias=random.gauss(0, 0.1))
            nid += 1
        hidden_end = nid

        # Output node
        output_id = nid
        g.nodes[output_id] = NodeGene(output_id, "output", bias=random.gauss(0, 0.1))
        nid += 1

        # Connections: input → hidden (fully connected)
        for i in range(input_end):
            for h in range(hidden_start, hidden_end):
                innov = _next_innovation_id(g.innovation_history, i, h)
                w = random.gauss(0, 0.5)
                g.connections[innov] = ConnectionGene(innov, i, h, weight=w)

        # Connections: hidden → output (fully connected)
        for h in range(hidden_start, hidden_end):
            innov = _next_innovation_id(g.innovation_history, h, output_id)
            w = random.gauss(0, 0.5)
            g.connections[innov] = ConnectionGene(innov, h, output_id, weight=w)

        return g

    # ── Mutation ──

    def mutate_weights(self, tau: float, p_weight: float = 0.8, p_bias: float = 0.2) -> None:
        """Perturb connection weights and node biases.

        Args:
            tau: Scale factor for log-normal perturbation.
            p_weight: Per-connection probability of weight change.
            p_bias: Per-node probability of bias change.
        """
        for conn in self.connections.values():
            if random.random() < p_weight:
                multiplier = math.exp(tau * random.uniform(-1, 1))
                conn.weight = np.clip(conn.weight * multiplier, -3.0, 3.0)

        for node in self.nodes.values():
            if random.random() < p_bias:
                node.bias += random.gauss(0, tau)

    def mutate_add_node(self) -> bool:
        """Split an existing connection by inserting a new hidden node.

        The old connection is disabled, and two new connections are created:
        from→new_node (weight=1.0) and new_node→to (weight=old_weight).
        This preserves the network output function.

        Returns:
            True if a node was added.
        """
        enabled = [c for c in self.connections.values() if c.enabled]
        if not enabled:
            return False

        target = random.choice(enabled)
        new_id = max(self.nodes.keys()) + 1 if self.nodes else 0

        # Disable the old connection
        target.enabled = False

        # Create new hidden node
        self.nodes[new_id] = NodeGene(new_id, "hidden", bias=0.0)

        # Connection from→new_node
        innov_a = _next_innovation_id(self.innovation_history, target.from_node, new_id)
        self.connections[innov_a] = ConnectionGene(
            innov_a, target.from_node, new_id, weight=1.0
        )

        # Connection new_node→to
        innov_b = _next_innovation_id(self.innovation_history, new_id, target.to_node)
        self.connections[innov_b] = ConnectionGene(
            innov_b, new_id, target.to_node, weight=target.weight
        )

        return True

    def mutate_add_connection(self) -> bool:
        """Add a new connection between two previously unconnected nodes.

        Only creates feed-forward connections (from lower-ID group to higher-ID group),
        preventing cycles.

        Returns:
            True if a connection was added.
        """
        existing = set((c.from_node, c.to_node) for c in self.connections.values())
        candidates = []

        node_ids = sorted(self.nodes.keys())
        for i, a in enumerate(node_ids):
            for b in node_ids[i + 1:]:
                kind_a = self.nodes[a].kind
                kind_b = self.nodes[b].kind
                # Only feed-forward: earlier → later node
                # Prevent output→something and something→input
                if kind_a == "output" or kind_b == "input":
                    continue
                if (a, b) not in existing:
                    candidates.append((a, b))

        if not candidates:
            return False

        from_id, to_id = random.choice(candidates)
        innov = _next_innovation_id(self.innovation_history, from_id, to_id)
        self.connections[innov] = ConnectionGene(
            innov, from_id, to_id, weight=random.gauss(0, 0.5)
        )
        return True

    def mutate(self, tau: float) -> None:
        """Apply all mutation operators with default probabilities."""
        self.mutate_weights(tau)
        if random.random() < 0.03:
            self.mutate_add_node()
        if random.random() < 0.05:
            self.mutate_add_connection()

    # ── Crossover ──

    @staticmethod
    def crossover(parent_a: Genome, parent_b: Genome) -> Genome:
        """Create a child genome via NEAT crossover.

        Matching genes (same innovation number) are inherited randomly.
        Disjoint and excess genes come from the fitter parent.
        If fitness is equal, both parents contribute disjoint/excess.
        """
        child = Genome()

        # Copy all nodes from both parents (deduplicate by node_id)
        all_node_ids: Set[int] = set()
        for nid in parent_a.nodes:
            all_node_ids.add(nid)
        for nid in parent_b.nodes:
            all_node_ids.add(nid)
        for nid in all_node_ids:
            if nid in parent_a.nodes:
                child.nodes[nid] = NodeGene(
                    nid,
                    parent_a.nodes[nid].kind,
                    parent_a.nodes[nid].bias,
                )
            else:
                child.nodes[nid] = NodeGene(
                    nid,
                    parent_b.nodes[nid].kind,
                    parent_b.nodes[nid].bias,
                )

        # Determine the fitter parent for disjoint/excess handling
        fitter = parent_a if parent_a.fitness >= parent_b.fitness else parent_b
        weaker = parent_b if parent_a.fitness >= parent_b.fitness else parent_a
        fitter_innovs = set(fitter.connections.keys())
        weaker_innovs = set(weaker.connections.keys())

        matching = fitter_innovs & weaker_innovs
        disjoint_excess = fitter_innovs - weaker_innovs

        for innov in matching:
            # Randomly pick from either parent
            source = random.choice([fitter, weaker])
            child.connections[innov] = source.connections[innov].copy()

        for innov in disjoint_excess:
            child.connections[innov] = fitter.connections[innov].copy()

        # Copy innovation history from fitter parent
        child.innovation_history = dict(fitter.innovation_history)

        return child

    # ── Distance (for future speciation) ──

    def distance(self, other: Genome, c1: float = 1.0, c2: float = 1.0, c3: float = 0.4) -> float:
        """Compute genomic distance between two genomes.

        Uses NEAT's standard formula:
            distance = (c1 * E)/N + (c2 * D)/N + c3 * avg_weight_diff

        Where E = excess genes, D = disjoint genes, N = genome size.
        """
        self_innovs = set(self.connections.keys())
        other_innovs = set(other.connections.keys())
        max_innov = max(max(self_innovs, default=0), max(other_innovs, default=0))

        matching = self_innovs & other_innovs
        disjoint = (self_innovs ^ other_innovs) - {i for i in (self_innovs ^ other_innovs) if i > max_innov}
        excess = {i for i in (self_innovs ^ other_innovs) if i > max_innov}

        N = max(len(self.connections), len(other.connections), 1)
        E = len(excess)
        D = len(disjoint)

        weight_diff = 0.0
        for innov in matching:
            weight_diff += abs(
                self.connections[innov].weight - other.connections[innov].weight
            )
        avg_weight = weight_diff / max(len(matching), 1)

        return (c1 * E) / N + (c2 * D) / N + c3 * avg_weight

    # ── Serialisation ──

    def to_dict(self) -> dict:
        """Serialize genome to a JSON-compatible dict."""
        return {
            "nodes": [
                {"node_id": n.node_id, "kind": n.kind, "bias": n.bias}
                for n in self.nodes.values()
            ],
            "connections": [
                {
                    "innovation": c.innovation,
                    "from_node": c.from_node,
                    "to_node": c.to_node,
                    "weight": c.weight,
                    "enabled": c.enabled,
                }
                for c in self.connections.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Genome:
        """Deserialize genome from a dict."""
        g = cls()
        for nd in data["nodes"]:
            g.nodes[nd["node_id"]] = NodeGene(nd["node_id"], nd["kind"], nd["bias"])
        for cd in data["connections"]:
            g.connections[cd["innovation"]] = ConnectionGene(
                cd["innovation"],
                cd["from_node"],
                cd["to_node"],
                cd["weight"],
                cd["enabled"],
            )
        return g

    def node_count(self) -> int:
        """Return total number of nodes."""
        return len(self.nodes)

    def connection_count(self) -> int:
        """Return total number of connections (including disabled)."""
        return len(self.connections)
