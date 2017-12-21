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
        if len(players) > 1:
            mother = players.pop(0)
            father = players.pop(0)
            # perform crossover
            children = crossOver(mother, father)
            # mutate new species and then add it to the offspring
            # list, ready to run on the next generation.
            for i in children:
                offspring = mutate(i)
                offsprings.append(offspring)
        else:
            parent = roulette(players)
            offspring = mutate(parent)
            offsprings.append(offspring)

    # now we need to generate new ID's for each player.
    for i in offsprings:
        i.genID()

    return offsprings[:populationCount]

def crossOver(cpu1, cpu2):
    """
    Basic Crossover Algorithm for the GA.
    """
    mother = cpu1.bot.nn.getAllCoefficents()
    father = cpu2.bot.nn.getAllCoefficents()

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
    child1.bot.nn.loadCoefficents(child1W)
    child2 = cpu2
    child2.bot.nn.loadCoefficents(child2W)

    # return the pair of children
    return (child1,child2)  

def roulette(players):
    """
    Roulette algorithm for parent selection.
    """
    chosen = None
    overallFitness = 0
    superEqual = False
    # calculate the overall fitness
    for i in players:
        overallFitness += i.points
    # check if overall fitness is zer0
    if overallFitness == 0:
        overallFitness = len(players)
        superEqual = True
    # randomly shuffle the players around.
    random.shuffle(players)
    # initiate a list of probablities
    probabilities = []
    # calculate probabllities for each player.
    for i in players:
        if not superEqual:
            probability = i.points / overallFitness
        else:
            probability = 1 / overallFitness
        if len(probabilities) > 1:
            probability = probability + probabilities[-1]
        probabilities.append(probability)
    # generate a random number
    number = random.random()
    # find a player.
    for i in range(len(probabilities)):
        if number < probabilities[i]:
            # we have chosen a player.
            chosen = players[i]
            break
    return chosen

def mutate(cpu):
    """
    Mutate the weights of the neural network.
    """

    mutationRate = 0.9

    # generate a random number
    chance = random.random()
    if chance >= (1-mutationRate):
        # mutate the weights.
        weights = cpu.bot.nn.getAllCoefficents()
        length = weights.size
        base1 = weights[random.randint(0,length-1)]
        base2 = weights[random.randint(0,length-1)]
        temp = base1
        np.place(weights, weights==base1, temp)
        np.place(weights, weights==base2, base1)
        np.place(weights, weights==temp, base2)
    return cpu

