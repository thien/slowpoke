"""Neuroevolution strategies: StandardGA (fixed-topology) and NEAT (topology-evolving)."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from slowpoke.agents.evaluator.genome import Genome
from slowpoke.agents.evaluator.neural import NeuralNetwork


class EvolutionMethod:
    """Base class for neuroevolution strategies.

    Subclasses define how agents are created, crossed over, mutated,
    and how their weights/genomes are serialised.
    """

    def __init__(self, population) -> None:
        self.population = population

    def generate_bot(self, ply_depth: int, debug: bool) -> Any:
        """Create a bot for a new agent."""
        raise NotImplementedError

    def crossover(
        self, parent_a_id: int, parent_b_id: int, child1_id: int, child2_id: int
    ) -> Tuple[int, int]:
        """Cross over two parents to produce two children."""
        raise NotImplementedError

    def mutate(self, agent_id: int) -> Tuple[int, Any]:
        """Mutate the given agent. Returns (agent_id, mutation_result)."""
        raise NotImplementedError

    def set_weights(self, agent_id: int, weights: Any) -> None:
        """Apply weights (or genome) to the given agent."""
        raise NotImplementedError

    def get_weights(self, agent_id: int) -> Any:
        """Retrieve weights (or genome) from the given agent."""
        raise NotImplementedError

    def load_mutation_result(self, agent_id: int, result: Any) -> None:
        """Load a mutation result back into the agent."""
        raise NotImplementedError


class StandardGA(EvolutionMethod):
    """Standard genetic algorithm with fixed-topology weight evolution."""

    def generate_bot(self, ply_depth: int, debug: bool) -> Any:
        import slowpoke.agents.slowbro as sb

        return sb.Slowbro(
            ply_depth=ply_depth,
            use_mlx=True,
            use_parallel=self.population.use_parallel_mcts,
            num_parallel=self.population.parallel_threads,
            debug=debug,
        )

    def crossover(
        self, parent_a_id: int, parent_b_id: int, child1_id: int, child2_id: int
    ) -> Tuple[int, int]:
        pop = self.population
        mother_w = pop.get_weights(parent_a_id)
        father_w = pop.get_weights(parent_b_id)
        num_w = pop.num_weights

        if pop.crossover_method == 0:
            for _ in range(10):
                idx = np.random.randint(0, num_w)
                mother_w[idx], father_w[idx] = father_w[idx], mother_w[idx]
            pop.set_weights(child1_id, mother_w)
            pop.set_weights(child2_id, father_w)
        else:
            i1 = np.random.randint(0, num_w)
            i2 = np.random.randint(0, num_w)
            if i1 > i2:
                i1, i2 = i2, i1
            c1 = np.concatenate([father_w[:i1], mother_w[i1:i2], father_w[i2:]])
            c2 = np.concatenate([mother_w[:i1], father_w[i1:i2], mother_w[i2:]])
            pop.set_weights(child1_id, c1)
            pop.set_weights(child2_id, c2)

        return (child1_id, child2_id)

    def mutate(self, agent_id: int) -> Tuple[int, Any]:
        pop = self.population
        weights = pop.get_weights(agent_id)
        move_cache = pop.get_move_cache(agent_id)

        if pop.safe_mutations and len(move_cache) > 100:
            weights = pop.safe_mutation(agent_id)
            pop.players[agent_id].bot.cache = {}
        else:
            mult = np.random.random_sample([pop.num_weights])
            mult = pop.tau * mult
            mult = np.exp(mult)
            weights = weights * mult
            weights = np.clip(weights, -1, 1)
            pop.set_weights(agent_id, weights)

        return (agent_id, weights)

    def set_weights(self, agent_id: int, weights: Any) -> None:
        self.population.players[agent_id].bot.nn.load_coefficients(weights)

    def get_weights(self, agent_id: int) -> Any:
        return self.population.players[agent_id].bot.nn.get_all_coefficients()

    def load_mutation_result(self, agent_id: int, result: Any) -> None:
        self.population.players[agent_id].bot.nn.load_coefficients(result)


class NEATEvolution(EvolutionMethod):
    """NEAT: topology and weight evolution via genomes."""

    def generate_bot(self, ply_depth: int, debug: bool) -> Any:
        import slowpoke.agents.slowbro as sb

        nn = NeuralNetwork(layer_list=[32, 1], use_mlx=True, mode="neat")
        bot = sb.Slowbro(
            ply_depth=ply_depth,
            use_mlx=True,
            use_parallel=self.population.use_parallel_mcts,
            num_parallel=self.population.parallel_threads,
            debug=debug,
        )
        bot.nn = nn
        return bot

    def crossover(
        self, parent_a_id: int, parent_b_id: int, child1_id: int, child2_id: int
    ) -> Tuple[int, int]:
        pop = self.population
        ga = pop.players[parent_a_id].bot.nn._genome
        gb = pop.players[parent_b_id].bot.nn._genome
        if ga is not None and gb is not None:
            ca = Genome.crossover(ga, gb)
            cb = Genome.crossover(gb, ga)
            pop.players[child1_id].bot.nn._genome = ca
            pop.players[child2_id].bot.nn._genome = cb
        return (child1_id, child2_id)

    def mutate(self, agent_id: int) -> Tuple[int, Any]:
        pop = self.population
        genome = pop.players[agent_id].bot.nn._genome
        if genome is not None:
            genome.mutate(pop.tau)
        return (agent_id, genome.to_dict() if genome else {})

    def set_weights(self, agent_id: int, weights: Any) -> None:
        if isinstance(weights, dict):
            self.population.players[agent_id].bot.nn._genome = Genome.from_dict(weights)

    def get_weights(self, agent_id: int) -> Any:
        g = self.population.players[agent_id].bot.nn._genome
        return g.to_dict() if g is not None else {}

    def load_mutation_result(self, agent_id: int, result: Any) -> None:
        if isinstance(result, dict):
            self.population.players[agent_id].bot.nn._genome = Genome.from_dict(result)
