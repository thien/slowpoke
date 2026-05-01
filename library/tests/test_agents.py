"""
Test each agent implementation to verify they work correctly.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import checkers
from agents.evaluator.neural import NeuralNetwork
from agents.slowpoke import Slowpoke
from agents.agent import Agent
from core.game import tournamentMatch
import numpy as np

Black, White, empty = 0, 1, -1

class TestSlowpokeAgent(unittest.TestCase):
    """Test Slowpoke agent functionality."""
    
    def test_agent_initialization(self):
        """Test that Slowpoke agent initializes correctly."""
        bot = Slowpoke(plyDepth=2, use_mlx=False)
        agent = Agent(bot)
        self.assertIsNotNone(agent.bot)
        self.assertEqual(agent.bot.ply, 2)
    
    def test_agent_make_move(self):
        """Test that agent can make a move."""
        bot = Slowpoke(plyDepth=1, use_mlx=False)
        agent = Agent(bot)
        B = checkers.CheckerBoard()
        move = agent.make_move(B, Black)
        self.assertIn(move, B.get_moves())
    
    def test_agent_with_mlx(self):
        """Test that agent can be created with MLX enabled."""
        bot = Slowpoke(plyDepth=1, use_mlx=True)
        agent = Agent(bot)
        self.assertTrue(agent.bot.use_mlx)
    
    def test_agent_evaluate_board(self):
        """Test that agent can evaluate a board."""
        bot = Slowpoke(plyDepth=1, use_mlx=False)
        agent = Agent(bot)
        B = checkers.CheckerBoard()
        value = agent.bot.evaluate_board(B, Black)
        self.assertIsInstance(value, (int, float))

class TestAgentVsAgent(unittest.TestCase):
    """Test agent-to-agent gameplay."""
    
    def test_two_agents_play_game(self):
        """Test that two agents can play a game."""
        bot1 = Slowpoke(plyDepth=1, use_mlx=False)
        bot2 = Slowpoke(plyDepth=1, use_mlx=False)
        p1 = Agent(bot1)
        p2 = Agent(bot2)
        
        result = tournamentMatch(p1, p2, 0, False, False)
        self.assertIn('Winner', result)
    
    def test_mlx_agents_play_game(self):
        """Test that two MLX-enabled agents can play a game."""
        bot1 = Slowpoke(plyDepth=1, use_mlx=True)
        bot2 = Slowpoke(plyDepth=1, use_mlx=True)
        p1 = Agent(bot1)
        p2 = Agent(bot2)
        
        result = tournamentMatch(p1, p2, 0, False, False)
        self.assertIn('Winner', result)

if __name__ == '__main__':
    unittest.main(verbosity=2)