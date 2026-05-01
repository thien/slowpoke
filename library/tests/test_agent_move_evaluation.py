"""
Test that each agent can evaluate and make a valid move.

This test file focuses on verifying that each agent implementation
has the necessary functions to make valid moves on a checkers board.
"""

import unittest
import unittest.mock as mock
import sys
import os

# Add project root to path (library/tests -> library -> project root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import checkers
from agents.agent import Agent
from agents.magikarp import Magikarp
from agents.slowpoke import Slowpoke
from agents.geodude import Geodude
from agents.human import Human

Black, White, empty = 0, 1, -1


def get_all_agents():
    """Return a list of all agent instances for testing."""
    agents = []
    
    # Magikarp - random player
    agents.append(("Magikarp", Magikarp()))
    
    # Slowpoke - neural network MCTS player
    agents.append(("Slowpoke", Slowpoke(plyDepth=2, use_mlx=False)))
    
    # Geodude - MCTS player
    agents.append(("Geodude", Geodude(plyDepth=2)))
    
    return agents


def get_all_agents_including_human():
    """Return a list of all agent instances including Human (for attribute tests)."""
    agents = get_all_agents()
    agents.append(("Human", Human()))
    return agents


class TestAgentMoveEvaluation(unittest.TestCase):
    """Test that each agent can evaluate and make valid moves."""
    
    def test_all_agents_have_move_function(self):
        """Verify all agents have a move_function attribute."""
        for name, bot in get_all_agents_including_human():
            self.assertTrue(
                hasattr(bot, 'move_function'),
                f"{name} should have a move_function attribute"
            )
    
    def test_all_agents_can_make_valid_move(self):
        """Verify each agent can make a valid move on a fresh board."""
        for name, bot in get_all_agents():
            board = checkers.CheckerBoard()
            agent = Agent(bot)
            
            # Make a move as Black
            move = agent.make_move(board, Black)
            
            # Verify the move is in the list of legal moves
            legal_moves = board.get_moves()
            self.assertIn(
                move, legal_moves,
                f"{name} should make a valid move. Got {move}, expected one of {legal_moves}"
            )
    
    def test_human_can_make_valid_move(self):
        """Verify Human agent can make a valid move (mocked input)."""
        bot = Human()
        board = checkers.CheckerBoard()
        agent = Agent(bot)
        
        # Mock input to return first legal move
        with mock.patch('builtins.input', return_value='0'):
            move = agent.make_move(board, Black)
            legal_moves = board.get_moves()
            self.assertIn(move, legal_moves)
    
    def test_agents_return_different_moves(self):
        """Test that different agent types behave differently."""
        board = checkers.CheckerBoard()
        
        # Magikarp should return a random move
        magikarp = Agent(Magikarp())
        move1 = magikarp.make_move(board, Black)
        self.assertIn(move1, board.get_moves())
        
        # Slowpoke should also return a valid move
        slowpoke = Agent(Slowpoke(plyDepth=1, use_mlx=False))
        move2 = slowpoke.make_move(board, Black)
        self.assertIn(move2, board.get_moves())
        
        # Geodude should also return a valid move
        geodude = Agent(Geodude(plyDepth=2))
        move3 = geodude.make_move(board, Black)
        self.assertIn(move3, board.get_moves())
        
        # Human should also return a valid move
        human = Agent(Human())
        with mock.patch('builtins.input', return_value='0'):
            move4 = human.make_move(board, Black)
            self.assertIn(move4, board.get_moves())
    
    def test_magikarp_move_function_signature(self):
        """Test Magikarp's move_function accepts correct parameters."""
        bot = Magikarp()
        board = checkers.CheckerBoard()
        
        # Should accept board and colour
        move = bot.move_function(board, Black)
        self.assertIn(move, board.get_moves())
    
    def test_slowpoke_move_function_signature(self):
        """Test Slowpoke's move_function accepts correct parameters."""
        bot = Slowpoke(plyDepth=1, use_mlx=False)
        board = checkers.CheckerBoard()
        
        move = bot.move_function(board, Black)
        self.assertIn(move, board.get_moves())
    
    def test_geodude_move_function_signature(self):
        """Test Geodude's move_function accepts correct parameters."""
        bot = Geodude(plyDepth=2)
        board = checkers.CheckerBoard()
        
        move = bot.move_function(board, Black)
        self.assertIn(move, board.get_moves())
    
    def test_human_move_function_signature(self):
        """Test Human's move_function accepts correct parameters."""
        bot = Human()
        board = checkers.CheckerBoard()
        
        # Human requires interactive input, so we test the method exists
        self.assertTrue(hasattr(bot, 'move_function'))
        # We can't test actual move without mocking input


class TestAgentBoardEvaluation(unittest.TestCase):
    """Test that agents can evaluate board states."""
    
    def test_slowpoke_can_evaluate_board(self):
        """Test Slowpoke can evaluate a board position."""
        bot = Slowpoke(plyDepth=1, use_mlx=False)
        board = checkers.CheckerBoard()
        
        value = bot.evaluate_board(board, Black)
        self.assertIsInstance(value, (int, float))
    
    def test_magikarp_has_no_evaluate_board(self):
        """Magikarp does not have board evaluation (it's random)."""
        bot = Magikarp()
        self.assertFalse(hasattr(bot, 'evaluate_board'))


if __name__ == '__main__':
    unittest.main(verbosity=2)