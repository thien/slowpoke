import os
from core import population

try:
    from agents import agent, geodude, human, magikarp, slowpoke
except ImportError:
    pass
import random

random.seed(1)

import unittest

# Performance benchmarks comparing NumPy vs MLX implementations
try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

"""
These unit tests cover situations that are hard to cover since they normally take a while to produce.
Normally, the classes have tests within their files.
"""


class PopulationTestCase(unittest.TestCase):
    def setUp(self):
        self.population = population.Population(15, 4)

    def test_champSaveLocations(self):
        # checks champion save location
        k = os.path.realpath(self.population.folderDirectory)
        l = os.path.join("..", "results", "champions")
        l = os.path.realpath(l)
        self.assertEqual(k, l)

    def test_createMutations(self):
        self.assertEqual(1, 1)

    def test_allocate_points(self):
        before = self.population.print_population_by_points()
        for _ in range(0, 100):
            k = random.choice(self.population.current_population)
            j = random.choice(self.population.current_population)
            # some fake result
            res = {"Winner": k}
            self.population.allocate_points(res, k, j)
        after = self.population.print_population_by_points()
        self.assertNotEqual(before, after)

    def test_safe_mutations(self):
        self.population.debug = True
        self.population.safe_mutations = True
        # generate fake moves
        (fakeMoves, coefs) = self.population.generate_fake_moves()
        for player_id in self.population.current_population:
            # overload their neural net with fake coef and moves
            self.population.players[player_id].bot.nn.load_coefficients(coefs)
            self.population.players[player_id].bot.cache = fakeMoves
        for _ in range(0, 100):
            k = random.choice(self.population.current_population)
            j = random.choice(self.population.current_population)
            # some fake result
            res = {"Winner": k}
            self.population.allocate_points(res, k, j)

        self.population.sort_population_by_points()
        print(self.population.print_population_by_points())
        # generate the next population
        self.population.generate_next_population()
        # self.assert


def Testing():
    p = population.Population(15, 1)
    p.safe_mutations = True

    (fakeMoves, coefs) = p.generate_fake_moves()

    for player_id in p.current_population:
        # overload their neural net with fake coef and moves
        p.players[player_id].bot.nn.load_coefficients(coefs)
        p.players[player_id].bot.cache = fakeMoves
    # overload fake tournament results
    for _ in range(0, 100):
        k = random.choice(p.current_population)
        j = random.choice(p.current_population)
        # some fake result
        res = {"Winner": k}
        p.allocate_points(res, k, j)
    p.sort_population_by_points()
    print(p.print_population_by_points())
    p.add_champion()
    # generate the next population
    p.generate_next_population()
    print(p.current_population)


def testCrossover():
    p = population.Population(15, 1)
    p.safe_mutations = True

    (fakeMoves, coefs) = p.generate_fake_moves()

    for player_id in p.current_population:
        # overload their neural net with fake coef and moves
        p.players[player_id].bot.nn.load_coefficients(coefs)
        p.players[player_id].bot.cache = fakeMoves
    # overload fake tournament results
    for _ in range(0, 10):
        k = random.choice(p.current_population)
        j = random.choice(p.current_population)
        # some fake result
        res = {"Winner": k}
        p.allocate_points(res, k, j)
    p.sort_population_by_points()
    # print(p.print_population_by_points())
    p.add_champion()
    # generate the next population
    # p.generate_next_population()
    # print(p.current_population)
    p.heuristic_crossover(1, 2, 3, 4)


if __name__ == "__main__":
    # unittest.main()
    # Testing()
    testCrossover()
