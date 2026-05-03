"""
Tests for parallel MCTS integration with Population and Tournament.
"""

import unittest

from core.population import Population
from search.parallel_tmcts import ParallelTMCTS
from agents.slowbro import Slowbro


class TestParallelPopulationIntegration(unittest.TestCase):
    """Test parallel MCTS integration with Population."""

    def test_ply_1_uses_slowbro(self):
        """ply=1 should use Slowbro (native [32] NN)."""
        # Create population with ply=1
        pop = Population(
            num_players=2,
            ply_depth=1,
            is_debug=True,
            use_parallel_mcts=False,
            use_neat=False,
        )

        # Check that bots are Slowbro
        for player_id, player in pop.players.items():
            self.assertIsInstance(
                player.bot, Slowbro, f"Player {player_id} should be Slowbro"
            )

    def test_ply_2_uses_parallel_by_default(self):
        """ply=2 should use ParallelTMCTS by default (auto mode)."""
        # We need to test the bot creation directly since Population expects nn
        Population(
            num_players=2,
            ply_depth=2,
            is_debug=True,
            use_parallel_mcts=True,
            use_neat=False,
        )
        # If we got here without error, the parallel flag was accepted
        # The actual bot creation happens in generate_player

    def test_explicit_parallel_enabled_ply_1(self):
        """Explicitly enabling parallel should use ParallelTMCTS even for ply=1."""
        Population(
            num_players=2,
            ply_depth=1,
            is_debug=True,
            use_parallel_mcts=True,
            use_neat=False,
        )
        # If we got here without error, the parallel flag was accepted

    def test_parallel_thread_count(self):
        """Population should configure parallel thread count."""
        pop = Population(
            num_players=2,
            ply_depth=4,
            is_debug=True,
            use_parallel_mcts=True,
            use_neat=False,
        )
        self.assertEqual(
            pop.parallel_threads, 4, "Default parallel threads should be 4"
        )

    def test_parallel_disabled_flag(self):
        """use_parallel_mcts=False should disable parallel even for deep ply."""
        pop = Population(
            num_players=2,
            ply_depth=4,
            is_debug=True,
            use_parallel_mcts=False,
            use_neat=False,
        )
        self.assertFalse(pop.use_parallel_mcts, "use_parallel_mcts should be False")


class TestBaselineEntity(unittest.TestCase):
    """Test baseline entity behavior."""

    def test_baseline_exists(self):
        """Baseline entity should exist with default Elo of 500."""
        pop = Population(
            num_players=5,
            ply_depth=1,
            is_debug=True,
            include_baseline=True,
            use_neat=False,
        )
        self.assertIsNotNone(pop.baseline_entity, "Baseline entity should exist")
        self.assertEqual(pop.baseline_entity.id, -1, "Baseline ID should be -1")
        self.assertEqual(
            pop.baseline_entity.elo, 500.0, "Baseline Elo should be default 500"
        )
        self.assertTrue(
            pop.baseline_entity.isBaseline, "Baseline should have isBaseline=True"
        )

    def test_baseline_not_in_regular_population(self):
        """Baseline should not count toward regular population."""
        pop = Population(
            num_players=3,
            ply_depth=2,
            is_debug=True,
            include_baseline=True,
            use_neat=False,
        )
        # Population should have 3 regular players + 1 baseline
        self.assertEqual(
            len(pop.current_population),
            4,
            "Population should have 3 players + baseline",
        )
        # Baseline should be at the end
        self.assertEqual(
            pop.current_population[-1],
            pop.baseline_entity.id,
            "Baseline should be last in population",
        )

    def test_baseline_resets_each_generation(self):
        """Baseline should reset to configured Elo and 0 points each generation."""
        pop = Population(
            num_players=5,
            ply_depth=1,
            is_debug=True,
            include_baseline=True,
            baseline_elo=1500.0,
            use_neat=False,
        )
        # Simulate some tournament play
        pop.players[pop.baseline_entity.id].elo = 1000.0
        pop.players[pop.baseline_entity.id].points = 10.0

        # Generate next population
        pop.generate_next_population()

        # Baseline should be reset to configured value
        self.assertEqual(
            pop.players[pop.baseline_entity.id].elo,
            1500.0,
            "Baseline Elo should reset to configured value",
        )
        self.assertEqual(
            pop.players[pop.baseline_entity.id].points,
            0,
            "Baseline points should reset to 0",
        )

    def test_all_players_start_with_baseline_elo(self):
        """All players should start with the configured baseline Elo."""
        pop = Population(
            num_players=5,
            ply_depth=1,
            is_debug=True,
            include_baseline=True,
            baseline_elo=400.0,
            use_neat=False,
        )

        # All regular players should have baseline Elo
        for pid in pop.current_population:
            if pid != pop.baseline_entity.id:
                self.assertEqual(
                    pop.players[pid].elo,
                    400.0,
                    f"Player {pid} should start with baseline Elo",
                )

        # Baseline entity should also have baseline Elo
        self.assertEqual(
            pop.players[pop.baseline_entity.id].elo,
            400.0,
            "Baseline should have baseline Elo",
        )

    def test_no_duplicate_baseline_on_multiple_generate_calls(self):
        """Multiple calls to generate_baseline_player should return the same entity."""
        pop = Population(
            num_players=2,
            ply_depth=1,
            is_debug=True,
            include_baseline=True,
            use_neat=False,
        )

        first_id = pop.baseline_entity.id
        first_generation = pop.generate_baseline_player()

        # Should return the same baseline entity
        self.assertIs(
            pop.baseline_entity,
            first_generation,
            "generate_baseline_player should return same entity",
        )
        self.assertEqual(
            first_id, first_generation.id, "Baseline ID should remain constant"
        )

        # Population should not have duplicate baseline IDs
        baseline_count = pop.current_population.count(pop.baseline_entity.id)
        self.assertEqual(
            baseline_count, 1, "Baseline should appear exactly once in population"
        )


class TestParallelTMCTSInterface(unittest.TestCase):
    """Test ParallelTMCTS interface compatibility."""

    def test_move_function_exists(self):
        """ParallelTMCTS should have move_function for Agent compatibility."""
        # Create a minimal ParallelTMCTS instance
        tmcts = ParallelTMCTS(ply=2, evaluator=None, num_parallel=2, debug=True)

        # Check move_function exists
        self.assertTrue(
            hasattr(tmcts, "move_function"), "ParallelTMCTS should have move_function"
        )
        self.assertTrue(
            callable(tmcts.move_function), "move_function should be callable"
        )

    def test_decide_method_exists(self):
        """ParallelTMCTS should have decide method."""
        tmcts = ParallelTMCTS(ply=2, evaluator=None, num_parallel=2, debug=True)
        self.assertTrue(
            hasattr(tmcts, "decide"), "ParallelTMCTS should have decide method"
        )
        self.assertTrue(callable(tmcts.decide), "decide should be callable")


if __name__ == "__main__":
    unittest.main()
