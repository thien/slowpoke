# import other self packages
import agent
import checkers
import random
import slowpoke as sp

# import libraries
import operator
import numpy as np
import math
import json
from random import randint
from pymongo import MongoClient

# import multiprocessor
from multiprocessing import Pool
pool = Pool()

# TODO: Consider Multithreading
# TODO: Store results and player information into db

Black, White, empty = 0, 1, -1

class Generator:
    """
    Configuration Functions
    """

    def __init__(self, configpath, generationCount=100, population=15):
        # Declare base information
        self.generations = generationCount
        self.population = population
        self.tournamentRounds = 5

        # Initiate other information
        self.config = self.loadJSONConfig(configpath)
        # once we have the config file we can proceed and initiate our
        # MongoDB connection.
        self.initiateMongoConnection()

    def loadJSONConfig(self, filepath):
        """
        Loads config.json
        """
        with open(filepath) as json_file:
            data = json.load(json_file)
        return data

# -------------------------------------------------------------------------
    """
    Helper Functions to initiate and write to mongodb.
    """

    def initiateMongoConnection(self):
        """
        Initiates a connection to the mongo instance.
        This is needed to store our results onto a database.
        Our webviewer will retrieve contents from there in order
        to display analytics related to slowpoke.
        """
        try:
            mongo = MongoClient(self.config['MongoURI'])
            self.db = mongo.zephyr
            print(self.db)
            print("Successfully connected to Mongo.")
        except Exception as e:
            print(e)
            print("Warning: Slowpoke is not currently connected to a mongo instance.")


    # ----------------------------
    # Helper Functions - here to make the code cleaner.
    # ----------------------------
    def mongoWrite(self, collection, entry):
        mongo_id = self.db[collection].insert(entry)
        return mongo_id
    
    def mongoUpdate(self, collection, mid, entry):
        self.db[collection].update_one({'_id':mid}, {"$set": entry}, upsert=False)

# -------------------------------------------------------------------------

    def playGame(self, blackCPU, whiteCPU, gameID="NULL", generationID="NULL"):
        # assign colours
        blackCPU.assignColour(Black)
        whiteCPU.assignColour(White)

        B = checkers.CheckerBoard()
        # set the ID for this game.
        B.setID(gameID)
        B.pdn['Black'] : blackCPU.id
        B.pdn['White'] : whiteCPU.id

        # add the game to mongo.
        mongoGame_id = self.db['generation']['games'].insert(B.pdn)

        # set game settings
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
            self.mongoUpdate('generation.games', mongoGame_id, B.pdn)
        # once game is done, update the pdn with the results and return it.
        self.mongoUpdate('generation.games', mongoGame_id, B.pdn)
        return B.pdn

    def sampleGame(self):

        """
        Generates a sample game for testing purposes.
        """
        # initiate agent for Slowpoke (we'll need this so we can make competitions.)
        bot1 = sp.Slowpoke()
        cpu1 = agent.Agent(bot1)

        bot2 = sp.Slowpoke()
        cpu2 = agent.Agent(bot2)

        # make them play a game.
        results = self.playGame(cpu1, cpu2)
        # print results.
        print(results)

    def generatePlayers(self):
        """
        Generates Players to participate in the tournament.
        This is only called at the beginning of the genetic algorithm.
        """
        population = []
        for i in range(self.population):
            bot = sp.Slowpoke()
            cpu = agent.Agent(bot)
            # add it to the list.
            population.append(cpu)
        return population

    def Tournament(self, population, generationID):
        """
        Tournament; this determines the best players out of them all.
        returns the players in order of how good they are.
        """
        # TODO: CREATE JSON FOR TOURNAMENT
        # make bots play each other.
        for i in range(len(population)):
            for j in range(self.tournamentRounds):
                # choose a random number between 1 and population.
                rand = randint(0, len(population)-1)
                while (rand == i):
                    rand = randint(0, len(population)-1)

                # cpu1 is black, cpu2 is white
                cpu1 = population[i]
                cpu2 = population[rand]
                # generate ID for the game (so we can store it on Mongo)
                IDPadding = str(generationID) +"_"+ str(i) +"_"+ str(j)
                game_id = str(IDPadding) + cpu1.id + cpu2.id

                # make the bots play a game.
                results = self.playGame(cpu1, cpu2, game_id, generationID)
                # allocate points for each player.
                if results["Winner"] == Black:
                    population[i].points += 1
                    population[rand].points -= 2
                elif results["Winner"] == White:
                    population[i].points -= 2
                    population[rand].points += 1
                    
        # order the players by how good they are.
        players_ranking = sorted(population, key=operator.attrgetter('points'))

        # # return r
        # for i in range(len(players_ranking)):
        #     print(players_ranking[i].points)
        return players_ranking
    
    def crossOver(self, cpu1, cpu2):
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

    def mutate(self, cpu):
        """
        Mutate the weights of the neural network.
        """
        return cpu

    def generateNewPopulation(self, players):
        """
        Input: list of players.
        Output: a new list of players.
        """
        # we half it and get the ceiling to put a cap on offspring
        player_choice_threshold = math.ceil(self.population/2)

        offsprings = []

        for i in range(player_choice_threshold):
            mother = players.pop(0)
            father = players.pop(0)
            # perform crossover
            children = self.crossOver(mother, father)
            # mutate new species and then add it to the offspring
            # list, ready to run on the next generation.
            for i in children:
                children[i] = self.mutate(children[i])
                offsprings.append(children[i])

        return offsprings[:self.population]
    
    def writePopulationToDB(self, population):
        """
        Stores the population into Mongo. 
        """
        keys = []
        for i in population:
            keys.append(self.mongoWrite('players', i.getDict()))
        return keys

    def runGenerations(self):
        # generate the initial population.
        population = self.generatePlayers()

        # loop through the generations.
        for i in range(self.generations):
            # store the population into mongo
            population_IDs = self.writePopulationToDB(population)
            # initiate generation dict so that we can store it on mongo.
            generation = {
                'count' : i,
                'population' : population_IDs,
                'games' : []
            }
            # write generation info to mongo.
            generation_id = self.mongoWrite('generation', generation)
            # print debug.
            print("Playing on Generation:", i)
            # make bots play each other.
            players = self.Tournament(population, generation_id)
            # get the best players and generate a new population from them.
            population = self.generateNewPopulation(players)

def run():
    configpath = "config.json"
    ga = Generator(configpath,10, 5)
    ga.runGenerations()

if __name__ == "__main__":
    run()