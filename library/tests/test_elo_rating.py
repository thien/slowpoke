"""
Tests for Elo rating calculations in the population system.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.population import EloRating

class TestEloRating(unittest.TestCase):
    """Test Elo rating system calculations."""
    
    def setUp(self):
        self.elo = EloRating(k_factor=32, initial_rating=1200)
    
    def test_expected_score_equal_ratings(self):
        """When ratings are equal, expected score should be 0.5."""
        expected = self.elo.expected_score(1200, 1200)
        self.assertAlmostEqual(expected, 0.5, places=6)
    
    def test_expected_score_higher_rating(self):
        """Higher rated player should have higher expected score."""
        expected = self.elo.expected_score(1500, 1200)
        self.assertGreater(expected, 0.5)
        self.assertAlmostEqual(expected, 0.849, places=2)
    
    def test_expected_score_lower_rating(self):
        """Lower rated player should have lower expected score."""
        expected = self.elo.expected_score(1200, 1500)
        self.assertLess(expected, 0.5)
        self.assertAlmostEqual(expected, 0.151, places=2)
    
    def test_expected_score_100_point_difference(self):
        """100 point difference should give ~64% win chance."""
        expected = self.elo.expected_score(1300, 1200)
        self.assertAlmostEqual(expected, 0.64, places=2)
    
    def test_expected_score_200_point_difference(self):
        """200 point difference should give ~76% win chance."""
        expected = self.elo.expected_score(1400, 1200)
        self.assertAlmostEqual(expected, 0.76, places=2)
    
    def test_update_rating_win(self):
        """Winner should gain rating, loser should lose."""
        winner_rating = 1200
        loser_rating = 1500
        
        new_winner = self.elo.update_rating(winner_rating, loser_rating, 1.0)
        new_loser = self.elo.update_rating(loser_rating, winner_rating, 0.0)
        
        self.assertGreater(new_winner, winner_rating)
        self.assertLess(new_loser, loser_rating)
    
    def test_update_rating_loss(self):
        """Loser should lose rating points."""
        rating = 1200
        opponent = 1500
        
        new_rating = self.elo.update_rating(rating, opponent, 0.0)
        
        self.assertLess(new_rating, rating)
    
    def test_update_rating_draw(self):
        """Draw should result in small rating changes."""
        rating_a = 1200
        rating_b = 1200
        
        new_a = self.elo.update_rating(rating_a, rating_b, 0.5)
        new_b = self.elo.update_rating(rating_b, rating_a, 0.5)
        
        # Ratings should remain unchanged for equal ratings and draw
        self.assertAlmostEqual(new_a, rating_a, places=6)
        self.assertAlmostEqual(new_b, rating_b, places=6)
    
    def test_k_factor_effect(self):
        """Higher K-factor should cause larger rating changes."""
        low_k = EloRating(k_factor=10, initial_rating=1200)
        high_k = EloRating(k_factor=32, initial_rating=1200)
        
        new_low = low_k.update_rating(1200, 1500, 1.0)
        new_high = high_k.update_rating(1200, 1500, 1.0)
        
        # High K should result in bigger change
        self.assertGreater(new_high, new_low)
    
    def test_anchor_player_stability(self):
        """Anchor player (high rating) should have stable ratings with K=32."""
        anchor = 2000
        challenger = 1200
        
        # Challenger beats anchor - anchor loses rating
        new_anchor = self.elo.update_rating(anchor, challenger, 0.0)
        
        # With K=32, anchor loses ~32 points (expected was ~0.998)
        anchor_change = anchor - new_anchor
        self.assertLess(anchor_change, 35)  # Reasonable for K=32
        self.assertGreater(anchor_change, 25)  # Significant change with K=32

class TestEloRatingKFactorVariants(unittest.TestCase):
    """Test different K-factor configurations."""
    
    def test_k32_new_player(self):
        """K=32 for new players (default)."""
        elo = EloRating(k_factor=32)
        # Upset win for lower-rated player
        new_rating = elo.update_rating(1200, 1500, 1.0)
        self.assertGreater(new_rating, 1227)
    
    def test_k24_intermediate(self):
        """K=24 for intermediate players."""
        elo = EloRating(k_factor=24)
        new_rating = elo.update_rating(1200, 1500, 1.0)
        self.assertLess(new_rating, 1227)  # Smaller change than K=32
        self.assertGreater(new_rating, 1220)
    
    def test_k10_established(self):
        """K=10 for established players."""
        elo = EloRating(k_factor=10)
        new_rating = elo.update_rating(1200, 1500, 1.0)
        self.assertLess(new_rating, 1216)  # Even smaller change
        self.assertGreater(new_rating, 1208)
    
    def test_dynamic_k_factor_new_player(self):
        """New player (<10 games) should use K=32."""
        elo = EloRating()
        # Player with 5 games played
        new_rating = elo.update_rating(1200, 1500, 1.0, games_played=5)
        self.assertGreater(new_rating, 1227)  # K=32 applied
    
    def test_dynamic_k_factor_intermediate(self):
        """Intermediate player (10-50 games) should use K=24."""
        elo = EloRating()
        # Player with 20 games played
        new_rating = elo.update_rating(1200, 1500, 1.0, games_played=20)
        self.assertLess(new_rating, 1227)  # K=24 applied
        self.assertGreater(new_rating, 1220)
    
    def test_dynamic_k_factor_established(self):
        """Established player (>50 games) should use K=10."""
        elo = EloRating()
        # Player with 100 games played
        new_rating = elo.update_rating(1200, 1500, 1.0, games_played=100)
        self.assertLess(new_rating, 1216)  # K=10 applied
        self.assertGreater(new_rating, 1208)

class TestEloIntegration(unittest.TestCase):
    """Integration tests for Elo rating with Population."""
    
    def test_populate_has_elo_system(self):
        """Population should have an elo_system."""
        from core.population import Population
        pop = Population(numberOfPlayers=5, plyDepth=1)
        self.assertIsNotNone(pop.elo_system)
    
    def test_print_population_includes_elo(self):
        """printCurrentPopulationByPoints should include Elo ratings."""
        from core.population import Population
        pop = Population(numberOfPlayers=5, plyDepth=1)
        
        # Set known Elo ratings
        for i, pid in enumerate(pop.currentPopulation):
            pop.players[pid].elo = 1200 + i * 100
            pop.players[pid].points = i * 2
        
        output = pop.printCurrentPopulationByPoints()
        # Check that output contains Elo
        self.assertIn("Elo:", output)
        # Check that output contains points
        self.assertIn("Pts:", output)
    
    def test_print_population_by_elo_includes_points(self):
        """printCurrentPopulationByElo should include points."""
        from core.population import Population
        pop = Population(numberOfPlayers=5, plyDepth=1)
        
        # Set known Elo ratings
        for i, pid in enumerate(pop.currentPopulation):
            pop.players[pid].elo = 1200 + i * 100
            pop.players[pid].points = i * 2
        
        output = pop.printCurrentPopulationByElo()
        # Check that output contains Elo
        self.assertIn("Elo:", output)
        # Check that output contains points
        self.assertIn("Pts:", output)
        # Check that it's sorted by Elo (highest first)
        lines = output.strip().split('\n')[1:]  # Skip header
        elos = [float(line.split('\t')[1].split(':')[1]) for line in lines]
        self.assertEqual(elos, sorted(elos, reverse=True))
    
    def test_agent_has_elo_attribute(self):
        """Agent should have elo attribute initialized to 1200."""
        import agents.agent as agent
        import agents.slowpoke as sp
        bot = sp.Slowpoke(plyDepth=1, use_mlx=False)
        human = agent.Agent(bot)
        self.assertEqual(human.elo, 1200)
    
    def test_agent_has_games_played_attribute(self):
        """Agent should have games_played attribute initialized to 0."""
        import agents.agent as agent
        import agents.slowpoke as sp
        bot = sp.Slowpoke(plyDepth=1, use_mlx=False)
        human = agent.Agent(bot)
        self.assertEqual(human.games_played, 0)
    
    def test_offspring_elo_inheritance(self):
        """Offspring should inherit mean Elo from parents for crossover children."""
        from core.population import Population
        pop = Population(numberOfPlayers=10, plyDepth=1)
        
        # Set known Elo ratings for first two players
        pop.players[pop.currentPopulation[0]].elo = 1400
        pop.players[pop.currentPopulation[1]].elo = 1200
        
        # Generate next population
        pop.generateNextPopulation()
        
        # First 2 offspring (crossover) should have mean Elo of 1300
        mean_elo = (1400 + 1200) / 2
        for i in range(2):
            offspring_id = pop.currentPopulation[i]
            self.assertAlmostEqual(pop.players[offspring_id].elo, mean_elo, places=2)
        
        # Next 2 offspring (copies) should have parent's Elo
        self.assertEqual(pop.players[pop.currentPopulation[2]].elo, 1400)
        self.assertEqual(pop.players[pop.currentPopulation[3]].elo, 1200)
    
    def test_allocate_points_updates_elo(self):
        """allocatePoints should update Elo ratings after games."""
        from core.population import Population
        pop = Population(numberOfPlayers=3, plyDepth=1)
        
        # Set known Elo ratings
        black_id = pop.currentPopulation[0]
        white_id = pop.currentPopulation[1]
        pop.players[black_id].elo = 1200
        pop.players[white_id].elo = 1200
        
        # Black wins
        pop.allocatePoints({"Winner": 0}, black_id, white_id)  # Black wins
        self.assertGreater(pop.players[black_id].elo, 1200)
        self.assertLess(pop.players[white_id].elo, 1200)
        
        # Reset for draw test
        pop.players[black_id].elo = 1200
        pop.players[white_id].elo = 1200
        
        # Draw
        pop.allocatePoints({"Winner": -1}, black_id, white_id)  # Draw
        self.assertAlmostEqual(pop.players[black_id].elo, 1200, places=2)
        self.assertAlmostEqual(pop.players[white_id].elo, 1200, places=2)
    
    def test_allocate_points_increments_games_played(self):
        """allocatePoints should increment games_played for both players."""
        from core.population import Population
        pop = Population(numberOfPlayers=3, plyDepth=1)
        
        black_id = pop.currentPopulation[0]
        white_id = pop.currentPopulation[1]
        
        # Initially 0 games
        self.assertEqual(pop.players[black_id].games_played, 0)
        self.assertEqual(pop.players[white_id].games_played, 0)
        
        # Play a game
        pop.allocatePoints({"Winner": 0}, black_id, white_id)
        
        # Both should have 1 game
        self.assertEqual(pop.players[black_id].games_played, 1)
        self.assertEqual(pop.players[white_id].games_played, 1)
    
    def test_offspring_games_played_reset(self):
        """Offspring should have games_played reset to 0 in next generation."""
        from core.population import Population
        pop = Population(numberOfPlayers=10, plyDepth=1)
        
        # Set some games_played for parent players
        pop.players[pop.currentPopulation[0]].games_played = 50
        pop.players[pop.currentPopulation[1]].games_played = 30
        
        # Generate next population
        pop.generateNextPopulation()
        
        # All players in new population should have games_played = 0
        # Exclude baseline entity from this check
        for pid in pop.currentPopulation:
            if pid != pop.baselineEntity.id if pop.baselineEntity else True:
                self.assertEqual(pop.players[pid].games_played, 0,
                               f"Player {pid} should have games_played=0")
    
    def test_elites_games_played_reset(self):
        """Elites should have games_played reset when moving to next generation."""
        from core.population import Population
        pop = Population(numberOfPlayers=10, plyDepth=1)
        
        # Set games_played for all players
        for pid in pop.currentPopulation:
            pop.players[pid].games_played = 100
        
        # Generate next population
        pop.generateNextPopulation()
        
        # Elites should have games_played reset
        # Exclude baseline entity (last in population) when checking elites
        elites = pop.currentPopulation[-6:-1] if pop.baselineEntity else pop.currentPopulation[-5:]
        for elite_id in elites:
            self.assertEqual(pop.players[elite_id].games_played, 0)
    
    def test_print_elo_stats(self):
        """printEloStats should return Elo statistics."""
        from core.population import Population
        pop = Population(numberOfPlayers=5, plyDepth=1)
        
        # Set known Elo ratings (excluding baseline)
        pop.players[pop.currentPopulation[0]].elo = 1400
        pop.players[pop.currentPopulation[1]].elo = 1300
        pop.players[pop.currentPopulation[2]].elo = 1200
        pop.players[pop.currentPopulation[3]].elo = 1100
        pop.players[pop.currentPopulation[4]].elo = 1000
        # Baseline is at 500 by default
        
        output = pop.printEloStats()
        
        # Check output contains stats
        self.assertIn("Avg:", output)
        self.assertIn("Best:", output)
        self.assertIn("Worst:", output)
        self.assertIn("P0", output)  # Best player should be P0 (1400)
        
        # Check values - average is (1400+1300+1200+1100+1000+500)/6 = 1083.3
        self.assertIn("1083.3", output)  # Average
        self.assertIn("1400.0", output)  # Best
        self.assertIn("500.0", output)  # Worst (baseline)

if __name__ == '__main__':
    unittest.main(verbosity=2)