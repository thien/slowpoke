import agent
import checkers
import random
import slowpoke as sp
import operator
import numpy as np
import math
import json
from random import randint
from multiprocessing import Pool
pool = Pool()

# TODO: CREATE JSON FOR GENERATION
# TODO: CREATE JSON FOR TOURNAMENT
# TODO: Consider Multithreading
# TODO: Store results and player information into db

Black, White, empty = 0, 1, -1


def loadJSONConfigs(file_directory):
    json_data = open(file_directory).read()
    data = json.loads(json_data)
    return data

def storePlayer():
    return False

def storeTournamentGame():
    return False

def botGame(blackCPU, whiteCPU, storeToMongo=False):
    # assign colours
    blackCPU.assignColour(Black)
    whiteCPU.assignColour(White)

    B = checkers.CheckerBoard()
    current_player = B.active

    choice = 0
    # Start the game loop.
    while not B.is_over():
        print("Game is currently on move:", B.turnCount)
        # game loop!
        if  B.turnCount % 2 != choice:
            botMove = blackCPU.make_move(B)
            B.make_move(botMove)
            if B.active == current_player:
                # print ("Jumps must be taken.")
                continue
            else:
                current_player = B.active
        else:
            botMove = whiteCPU.make_move(B)
            B.make_move(botMove)
            if B.active == current_player:
                # print ("Jumps must be taken.")
                continue
            else:
                current_player = B.active
        print(B)
        # store the game to MongoDB.
    # once game is done, return the pgn
    return B.pdn

def sampleGame():
    # initiate agent for Slowpoke (we'll need this so we can make competitions.)
    bot1 = sp.Slowpoke()
    cpu1 = agent.Agent(bot1)

    bot2 = sp.Slowpoke()
    cpu2 = agent.Agent(bot2)

    # make them play a game.
    results = botGame(cpu1, cpu2)
    # print results.
    print(results)

def generatePlayers(tournamentSize=5):
    """
    Generates Players to participate in the tournament.
    This is only called at the beginning of the genetic algorithm.
    """
    participants = []
    for i in range(tournamentSize):
        bot = sp.Slowpoke()
        cpu = agent.Agent(bot)
        # add it to the list.
        participants.append(cpu)
    return participants

def Tournament(participants, gameRounds, storeToMongo=False):
    """
    Tournament; this determines the best players out of them all.
    returns the players in order of how good they are.
    """
    # TODO: CREATE JSON FOR TOURNAMENT
    # make bots play each other.
    for i in range(len(participants)):
        for j in range(gameRounds):

            # choose a random number between 1 and participants.
            rand = randint(0, len(participants)-1)
            while (rand == i):
                rand = randint(0, len(participants)-1)

            # cpu1 is black, cpu2 is white
            cpu1 = participants[i]
            cpu2 = participants[rand]
            # make the bots play a game.
            results = botGame(cpu1, cpu2)
            # allocate points for each player.
            if results["Winner"] == Black:
                participants[i].points += 1
                participants[rand].points -= 2
            elif results["Winner"] == White:
                participants[i].points -= 2
                participants[rand].points += 1
                
    # order the players by how good they are.
    players_ranking = sorted(participants, key=operator.attrgetter('points'))

    # # return r
    # for i in range(len(players_ranking)):
    #     print(players_ranking[i].points)
    return players_ranking

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
    return cpu

def generateNewPopulation(players, populationCount):
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
            children[i] = mutate(children[i])
            offsprings.append(children[i])

    return offsprings[:populationCount]

def GeneticAlgorithm(generations, populationCount=5):
    # generate players
    participants = generatePlayers(populationCount)

    # loop through tournament
    for i in range(generations):
        # TODO: CREATE JSON FOR GENERATION
        print("Playing on Generation:", i)
        # make bots play each other.
        players = Tournament(participants, 5) 
        # get the best players
        participants = generateNewPopulation(players, populationCount)
    return False

# Tournament(6,5)
if __name__ == "__main__":
    GeneticAlgorithm(4, 2)