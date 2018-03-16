
import os
from core import checkers, game, mongo, population, tournament
from decision import mcts, splash, minimax
from agents import agent, geodude, human, magikarp, slowpoke

import unittest

# class TestStringMethods(unittest.TestCase):

#     def test_upper(self):
#         self.assertEqual('foo'.upper(), 'FOO')

#     def test_isupper(self):
#         self.assertTrue('FOO'.isupper())
#         self.assertFalse('Foo'.isupper())

#     def test_split(self):
#         s = 'hello world'
#         self.assertEqual(s.split(), ['hello', 'world'])
#         # check that s.split fails when the separator is not a string
#         with self.assertRaises(TypeError):
#             s.split(2)

class PopulationTestCase(unittest.TestCase):
    def setUp(self):
        self.population = population.Population(15,4)

    def test_champSaveLocations(self):
        # checks champion save location
        k = os.path.realpath(self.population.folderDirectory)
        l = os.path.join("..","results","champions")
        l = os.path.realpath(l)
        self.assertEqual(k,l)

if __name__ == '__main__':
    unittest.main()