# import self packages
import agent
import checkers
import random
import slowpoke as sp
import mongo
import game
import genetic as ga

# import libraries
import datetime
import operator
import json
from random import randint

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

    def initiateMongoConnection(self):
        self.db = mongo.Mongo()
        self.db.initiate(self.config['MongoURI'])

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

    def Tournament(self, players, generation):
        """
        Tournament; this determines the best players out of them all.
        returns the players in order of how good they are.
        """
        # make bots play each other.
        for i in range(len(players)):
            for j in range(self.tournamentRounds):
                # choose a random number between 1 and the number of players.
                rand = randint(0, len(players)-1)
                while (rand == i):
                    rand = randint(0, len(players)-1)

                # cpu1 is black, cpu2 is white
                cpu1 = players[i]
                cpu2 = players[rand]
                # generate ID for the game (so we can store it on Mongo)
                IDPadding = generation['_id'] +"_"+ str(i) +"_"+ str(j)
                game_id = IDPadding + cpu1.id + cpu2.id
                generation['games'].append(game_id)
                # update generations entry to include the game.
                self.db.update('generation', generation['_id'], generation)

                # set up debug file.
                debug = {
                    'genCount' : generation['count'],
                    'printDebug' : True,
                    'printBoard' : False,
                    'genID' : generation['_id']
                }

                # make the bots play a game.
                results = game.tournamentMatch(cpu1, cpu2, game_id, self.db, debug)
                # allocate points for each player.
                if results["Winner"] == Black:
                    players[i].points += 1
                    players[rand].points -= 2
                elif results["Winner"] == White:
                    players[i].points -= 2
                    players[rand].points += 1
                    
        # order the players by how good they are.
        players_ranked = sorted(players, key=operator.attrgetter('points'))

        # # return r
        # for i in range(len(players_ranking)):
        #     print(players_ranking[i].points)
        return players_ranked
        
    def writePopulationToDB(self, population):
        """
        Stores the population into Mongo. 
        """
        keys = []
        for i in population:
            keys.append(self.db.write('players', i.getDict()))
        return keys

    def runGenerations(self):
        # generate the initial population.
        population = self.generatePlayers()

        # loop through the generations.
        for i in range(self.generations):
            # store the population into mongo
            population_IDs = self.writePopulationToDB(population)
            # initiate generation dict so that we can store it on mongo.
            timestamp = str(datetime.datetime.utcnow())

            generation = {
                '_id' : timestamp + str(i),
                'timestamp' : timestamp,
                'count' : i,
                'population' : population_IDs,
                'games' : []
            }
            # write generation info to mongo.
            self.db.write('generation', generation)

            # print debug.
            print("Playing on Generation:", i)
            # make bots play each other.
            players = self.Tournament(population, generation)
            # get the best players and generate a new population from them.
            population = ga.generateNewPopulation(players, self.population)

def run():
    configpath = "config.json"
    ga = Generator(configpath,10, 5)
    ga.runGenerations()

if __name__ == "__main__":
    run()