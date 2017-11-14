"""
Evolutionary algorithms are contained here for the sake 
of tracking (within Git.)
"""

import math
import random
import numpy as np


def generateNewPopulation(players, populationCount):
    """
    Input: list of players.
    Output: a new list of players.
    """
    # we half it and get the ceiling to put a cap on offspring
    player_choice_threshold = math.ceil(populationCount/2)

    offsprings = []

    for i in range(player_choice_threshold):
        mother = players.pop(0)
        father = players.pop(0)
        # perform crossover
        children = crossOver(mother, father)
        # mutate new species and then add it to the offspring
        # list, ready to run on the next generation.
        for i in children:
            offspring = mutate(i)
            offsprings.append(offspring)

    return offsprings[:populationCount]

def crossOver(cpu1, cpu2):
    """
    Basic Crossover Algorithm for the GA.
    """
    mother = cpu1.bot.nn.getWeights()
    father = cpu2.bot.nn.getWeights()

    # generate random indexes to cut off for crossover
    index1 = random.randint(0, len(mother))
    index2 = random.randint(0, len(mother))

    # check the order of the indexes to make sure they make sense.
    if index1 > index2: 
        index1, index2 = index2, index1

    # pythonic crossover
    child1W = np.append(np.append(father[:index1], mother[index1:index2]), father[index2:])
    child2W = np.append(np.append(mother[:index1], father[index1:index2]), mother[index2:])
    
    # create new children with it
    child1 = cpu1
    child1.bot.nn.setWeights(child1W)
    child2 = cpu2
    child2.bot.nn.setWeights(child2W)

    # return the pair of children
    return (child1,child2)  

def mutate(cpu):
    """
    Mutate the weights of the neural network.
    """
    return cpu

