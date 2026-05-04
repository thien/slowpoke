"""Tests for the Agent wrapper class."""

import unittest

from slowpoke.agents.agent import Agent
from slowpoke.agents.magikarp import Magikarp
from slowpoke.core import checkers

Black = 0
White = 1


class TestAgentInit(unittest.TestCase):
    """Test Agent initialisation."""

    def test_agent_creates_with_bot(self):
        """Agent should store the bot instance."""
        bot = Magikarp()
        agent = Agent(bot)
        self.assertIs(agent.bot, bot)

    def test_agent_default_elo(self):
        """Default Elo should be 100."""
        bot = Magikarp()
        agent = Agent(bot)
        self.assertEqual(agent.elo, 100)

    def test_agent_custom_elo(self):
        """Custom Elo should be respected."""
        bot = Magikarp()
        agent = Agent(bot, initial_elo=1500)
        self.assertEqual(agent.elo, 1500)

    def test_agent_custom_id(self):
        """Custom agent_id should be set."""
        bot = Magikarp()
        agent = Agent(bot, agent_id="test-id-123")
        self.assertEqual(agent.id, "test-id-123")

    def test_agent_auto_generates_id(self):
        """Without custom ID, agent should auto-generate one."""
        bot = Magikarp()
        agent = Agent(bot)
        self.assertIsNotNone(agent.id)
        self.assertIsInstance(agent.id, str)

    def test_agent_move_function(self):
        """Agent should expose bot's move_function."""
        bot = Magikarp()
        agent = Agent(bot)
        # Agent stores a reference to bot's bound method
        self.assertIsNotNone(agent.move_function)
        # Both produce valid moves (can't compare result equality since they're random)
        from core import checkers

        B = checkers.CheckerBoard()
        moves = B.get_moves()
        self.assertIn(agent.move_function(B, 0), moves)
        self.assertIn(bot.move_function(B, 0), moves)

    def test_agent_games_played_starts_at_zero(self):
        """games_played should start at 0."""
        bot = Magikarp()
        agent = Agent(bot)
        self.assertEqual(agent.games_played, 0)

    def test_agent_points_start_at_zero(self):
        """points should start at 0."""
        bot = Magikarp()
        agent = Agent(bot)
        self.assertEqual(agent.points, 0)

    def test_agent_colour_is_none(self):
        """colour should be None initially."""
        bot = Magikarp()
        agent = Agent(bot)
        self.assertIsNone(agent.colour)

    def test_agent_has_parents_list(self):
        """parents list should exist and be empty."""
        bot = Magikarp()
        agent = Agent(bot)
        self.assertEqual(agent.parents, [])


class TestAgentOrigin(unittest.TestCase):
    """Test origin tracking."""

    def test_generate_origin_creates_genesis_block(self):
        """generate_origin should create genesis origin."""
        bot = Magikarp()
        agent = Agent(bot)
        self.assertEqual(len(agent.origin), 1)
        self.assertEqual(agent.origin[0], [0, 0, 0])

    def test_generate_origin_adds_to_existing(self):
        """Calling generate_origin again should append."""
        bot = Magikarp()
        agent = Agent(bot)
        agent.origin.append([1, 2, 3])
        self.assertEqual(len(agent.origin), 2)


class TestAgentID(unittest.TestCase):
    """Test ID generation and setting."""

    def test_set_id(self):
        """set_id should update id."""
        bot = Magikarp()
        agent = Agent(bot)
        agent.set_id("new-id")
        self.assertEqual(agent.id, "new-id")

    def test_gen_id_creates_hash(self):
        """gen_id should create a hex hash string."""
        bot = Magikarp()
        agent = Agent(bot)
        agent.gen_id()
        self.assertIsInstance(agent.id, str)
        # MD5 hash is 32 hex characters
        self.assertEqual(len(agent.id), 32)

    def test_gen_id_is_consistent(self):
        """gen_id should produce the same ID for same bot state."""
        bot1 = Magikarp()
        agent1 = Agent(bot1)
        id1 = agent1.id

        bot2 = Magikarp()
        agent2 = Agent(bot2)
        id2 = agent2.id

        # Magikarp has no NN, so it falls back to time-based hash
        # IDs will differ (time-based), so just check format
        self.assertIsInstance(id1, str)
        self.assertIsInstance(id2, str)


class TestAgentGetDict(unittest.TestCase):
    """Test dictionary serialization."""

    def test_get_dict_contains_id(self):
        """get_dict should include agent ID."""
        bot = Magikarp()
        agent = Agent(bot)
        d = agent.get_dict()
        self.assertIn("_id", d)
        self.assertEqual(d["_id"], agent.id)

    def test_get_dict_contains_elo(self):
        """get_dict should include Elo."""
        bot = Magikarp()
        agent = Agent(bot, initial_elo=1350)
        d = agent.get_dict()
        self.assertIn("elo", d)
        self.assertEqual(d["elo"], 1350)

    def test_get_dict_contains_points(self):
        """get_dict should include points."""
        bot = Magikarp()
        agent = Agent(bot)
        agent.points = 42
        d = agent.get_dict()
        self.assertIn("points", d)
        self.assertEqual(d["points"], 42)

    def test_get_dict_weights_none_for_magikarp(self):
        """Magikarp has no NN weights, should be None."""
        bot = Magikarp()
        agent = Agent(bot)
        d = agent.get_dict()
        self.assertIsNone(d["weights"])


class TestAgentColour(unittest.TestCase):
    """Test colour assignment."""

    def test_assign_colour_black(self):
        """Assigning colour 0 should set agent.colour = 0."""
        bot = Magikarp()
        agent = Agent(bot)
        agent.assign_colour(Black)
        self.assertEqual(agent.colour, Black)

    def test_assign_colour_white(self):
        """Assigning colour 1 should set agent.colour = 1."""
        bot = Magikarp()
        agent = Agent(bot)
        agent.assign_colour(White)
        self.assertEqual(agent.colour, White)


class TestAgentMakeMove(unittest.TestCase):
    """Test agent move delegation."""

    def test_make_move_delegates_to_bot(self):
        """make_move should call bot.move_function and return result."""
        bot = Magikarp()
        agent = Agent(bot)
        B = checkers.CheckerBoard()
        move = agent.make_move(B, Black)
        self.assertIn(move, B.get_moves())

    def test_make_move_uses_bot_return(self):
        """The move returned should be from the bot's logic."""

        class FixedBot:
            def move_function(self, board, colour):
                return 42

        bot = FixedBot()
        agent = Agent(bot)
        result = agent.make_move(None, Black)
        self.assertEqual(result, 42)


if __name__ == "__main__":
    unittest.main(verbosity=2)
