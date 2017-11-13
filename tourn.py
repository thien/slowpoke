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


class Generator:
# -------------------------------------------------------------------------
    """
        Configuration Functions
    """

    def __init__(self, generationCount=100, population=15, configpath):
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
        json_data = open(filepath).read()
        data = json.loads(filepath)
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
            print("Successfully connected to Mongo.")
        except Exception as e:
            print(e)
            print("Warning: Slowpoke is not currently connected to a mongo instance.")


    # ----------------------------
    # Helper Functions - here to make the code cleaner.
    # ----------------------------
    def mongoWrite(self, collection, entry):
        mongo_id = self.db['generation'].insert(generation)
        return mongo_id
    
    def mongoUpdate(self, collection, mid, entry):
        self.db[collection].update_one({'_id':mid}, {"$set": entry}, upsert=False)

    # # ----------------------------
    # # Generations
    # # ----------------------------
    # def writeGenerationStats(self, generation):
    #     return mongoWrite('generation', generation)
    
    # def updateGenerationStats(self, mongo_id, generation):
    #     return mongoUpdate('generation', mongo_id, generation)

    # # ----------------------------
    # # Tournaments
    # # ----------------------------
    # def writeTournamentStats(self, tournament):
    #     return mongoWrite('tournament', tournament)

    # def updateTournamentStats(self, mongo_id, tournament):
    #     return mongoUpdate('tournament', mongo_id, tournament)

    # # ----------------------------
    # # Players
    # # ----------------------------
    # def writePlayerStats(self, player):
    #     mongo_id = self.db['players'].insert(player)
    #     return mongo_id

    # def updatePlayerStats(self, mongo_id, player):
    #     self.db['players'].update_one({'_id':mongo_id}, {"$set": player}, upsert=False)
# -------------------------------------------------------------------------

    def playGame(self, blackCPU, whiteCPU):
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
        results = botGame(cpu1, cpu2)
        # print results.
        print(results)

    def generatePlayers(self):
        """
        Generates Players to participate in the tournament.
        This is only called at the beginning of the genetic algorithm.
        """
        participants = []
        for i in range(self.population):
            bot = sp.Slowpoke()
            cpu = agent.Agent(bot)
            # add it to the list.
            participants.append(cpu)
        return participants

    def Tournament(self, participants, generationID):
        """
        Tournament; this determines the best players out of them all.
        returns the players in order of how good they are.
        """
        participants = generation['participants']
        # TODO: CREATE JSON FOR TOURNAMENT
        # make bots play each other.
        for i in range(len(participants)):
            for j in range(self.tournamentRounds):

                # choose a random number between 1 and participants.
                rand = randint(0, len(participants)-1)
                while (rand == i):
                    rand = randint(0, len(participants)-1)

                # cpu1 is black, cpu2 is white
                cpu1 = participants[i]
                cpu2 = participants[rand]
                # make the bots play a game.
                results = self.botGame(cpu1, cpu2)
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
            children = crossOver(mother, father)
            # mutate new species and then add it to the offspring
            # list, ready to run on the next generation.
            for i in children:
                children[i] = mutate(children[i])
                offsprings.append(children[i])

        return offsprings[:self.population]
    
    def runGenerations(self):
        # generate the initial population.
        participants = self.generatePlayers()
        # loop through the generations.
        for i in range(self.generations):
            for i in participants:
                participant_id = mongoWrite('players', i)
            # initiate generation dict
            generation = {
                'count' : i,
                'participants' : participants
            }
            # write to mongo.
            generation_id = mongoWrite('generation', generation)
            # print debug.
            print("Playing on Generation:", i)
            # make bots play each other.
            players = self.Tournament(participants, generation_id)
            # get the best players and generate a new population from them.
            participants = self.generateNewPopulation(players)

def run():
    configpath = "config.json"
    ga = Generator(10, 5, configpath)
    ga.runGenerations()

if __name__ == "__main__":
    run()