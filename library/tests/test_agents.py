"""
Test each agent implementation to verify they work correctly.
"""

import unittest

import pytest

from core import checkers
from agents.slowpoke import Slowpoke
from agents.agent import Agent
from core.game import tournament_match

Black, White, empty = 0, 1, -1


class TestSlowpokeAgent(unittest.TestCase):
    """Test Slowpoke agent functionality."""

    def test_agent_initialization(self):
        """Test that Slowpoke agent initializes correctly."""
        bot = Slowpoke(ply_depth=2, use_mlx=False)
        agent = Agent(bot)
        self.assertIsNotNone(agent.bot)
        self.assertEqual(agent.bot.ply, 2)

    def test_agent_make_move(self):
        """Test that agent can make a move."""
        bot = Slowpoke(ply_depth=1, use_mlx=False)
        agent = Agent(bot)
        B = checkers.CheckerBoard()
        move = agent.make_move(B, Black)
        self.assertIn(move, B.get_moves())

    def test_agent_with_mlx(self):
        """Test that agent can be created with MLX enabled."""
        bot = Slowpoke(ply_depth=1, use_mlx=True)
        agent = Agent(bot)
        self.assertTrue(agent.bot.use_mlx)

    def test_agent_evaluate_board(self):
        """Test that agent can evaluate a board."""
        bot = Slowpoke(ply_depth=1, use_mlx=False)
        agent = Agent(bot)
        B = checkers.CheckerBoard()
        value = agent.bot.evaluate_board(B, Black)
        self.assertIsInstance(value, (int, float))


class TestAgentVsAgent(unittest.TestCase):
    """Test agent-to-agent gameplay."""

    def test_two_agents_play_game(self):
        """Test that two agents can play a game."""
        bot1 = Slowpoke(ply_depth=1, use_mlx=False)
        bot2 = Slowpoke(ply_depth=1, use_mlx=False)
        p1 = Agent(bot1)
        p2 = Agent(bot2)

        result = tournament_match(p1, p2, 0, False, False)
        self.assertIn("Winner", result)

    def test_mlx_agents_play_game(self):
        """Test that two MLX-enabled agents can play a game."""
        bot1 = Slowpoke(ply_depth=1, use_mlx=True)
        bot2 = Slowpoke(ply_depth=1, use_mlx=True)
        p1 = Agent(bot1)
        p2 = Agent(bot2)

        result = tournament_match(p1, p2, 0, False, False)
        self.assertIn("Winner", result)


class TestGameOutcomes(unittest.TestCase):
    """Games between agents of different strengths must produce wins/losses."""

    @pytest.mark.slow
    def test_asymmetric_agents_produce_wins_and_losses(self):
        """A higher-ply agent should win at least some games against a lower-ply one.

        Regression test: the Rust player-switch change broke game outcomes entirely
        (every game ended in a draw). This verifies the player-switch logic is correct.

        Both agents use the same random NN weights — ply depth is the only difference.
        """
        from agents.slowbro import Slowbro

        # Use same random weights for both — only ply differs
        bot_strong = Slowbro(ply_depth=6, use_mlx=False, use_parallel=False)
        weak_weights = bot_strong.nn.get_all_coefficients().copy()
        bot_weak = Slowbro(ply_depth=1, use_mlx=False, use_parallel=False)
        bot_weak.nn.load_coefficients(weak_weights)

        p_strong = Agent(bot_strong)
        p_weak = Agent(bot_weak)

        strong_wins = 0
        weak_wins = 0
        draws = 0
        n_games = 30

        for i in range(n_games):
            if i % 2 == 0:
                result = tournament_match(p_strong, p_weak, i, False, False)
            else:
                result = tournament_match(p_weak, p_strong, i, False, False)
            w = result["Winner"]
            if w == -1:
                draws += 1
            elif i % 2 == 0 and w == 0 or i % 2 == 1 and w == 1:
                strong_wins += 1
            else:
                weak_wins += 1

        self.assertEqual(
            strong_wins + weak_wins + draws,
            n_games,
            f"All {n_games} games should complete, got {strong_wins}+{weak_wins}+{draws}",
        )
        self.assertGreater(
            strong_wins,
            0,
            f"Deep-ply agent should win at least 1/{n_games} games (won {strong_wins})",
        )
        self.assertGreater(
            draws + weak_wins + strong_wins,
            draws,
            "At least one game should end in a win or loss, not just draws",
        )


class TestOnixAgent(unittest.TestCase):
    """Test Onix heuristic-based agent."""

    def test_onix_initialization(self):
        from agents.onix import Onix

        bot = Onix(ply_depth=2)
        self.assertEqual(bot.ply, 2)

    def test_onix_can_make_move(self):
        from agents.onix import Onix

        bot = Onix(ply_depth=1)
        agent = Agent(bot)
        B = checkers.CheckerBoard()
        move = agent.make_move(B, 0)
        self.assertIn(move, B.get_moves())

    def test_onix_plays_game(self):
        from agents.onix import Onix

        bot = Onix(ply_depth=1)
        agent = Agent(bot)
        B = checkers.CheckerBoard()
        for _ in range(60):
            move = agent.make_move(B, B.active)
            B.make_move(move)
            if B.is_over():
                break
        # Game ran without crashing — passes regardless of outcome

    @pytest.mark.slow
    def test_onix_outcomes(self):
        """Onix(6) should beat Onix(1) most games (same heuristics, deeper search)."""
        from agents.onix import Onix

        strong = Onix(ply_depth=6)
        weak = Onix(ply_depth=1)
        p_strong = Agent(strong)
        p_weak = Agent(weak)

        strong_wins = 0
        weak_wins = 0
        draws = 0
        n = 30

        for i in range(n):
            if i % 2 == 0:
                result = tournament_match(p_strong, p_weak, i, False, False)
            else:
                result = tournament_match(p_weak, p_strong, i, False, False)
            w = result["Winner"]
            if w == -1:
                draws += 1
            elif i % 2 == 0 and w == 0 or i % 2 == 1 and w == 1:
                strong_wins += 1
            else:
                weak_wins += 1

        self.assertEqual(strong_wins + weak_wins + draws, n)
        self.assertGreater(strong_wins, 0)

    @pytest.mark.slow
    def test_onix_vs_slowbro(self):
        """Onix should at least draw against a random-weight Slowbro."""
        from agents.onix import Onix
        from agents.slowbro import Slowbro

        onix = Onix(ply_depth=3)
        slowbro = Slowbro(ply_depth=3, use_mlx=False, use_parallel=False)
        p_onix = Agent(onix)
        p_slowbro = Agent(slowbro)

        onix_wins = 0
        draws = 0
        n = 10

        for i in range(n):
            if i % 2 == 0:
                result = tournament_match(p_onix, p_slowbro, i, False, False)
            else:
                result = tournament_match(p_slowbro, p_onix, i, False, False)
            w = result["Winner"]
            if w == -1:
                draws += 1
            elif i % 2 == 0 and w == 0 or i % 2 == 1 and w == 1:
                onix_wins += 1

        # Onix should at least draw some games (not lose every time)
        self.assertGreaterEqual(onix_wins + draws, n // 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
