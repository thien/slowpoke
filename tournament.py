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
                # set up debug file.
                debug = {
                    'genCount' : generation['count'],
                    'printDebug' : True,
                    'printBoard' : False,
                    'genID' : generation['_id']
                }
    
                # choose a random number between 1 and the number of players.
                rand = self.randomPlayerRange(i, len(players))

                # cpu1 is black, cpu2 is white
                cpu1 = players[i]
                cpu2 = players[rand]

                # generate ID for the game (so we can store it on Mongo)
                game_id = self.generateGameID(generation['_id'], i, j, cpu1, cpu2)
                generation['games'].append(game_id)
                
                # update generations entry to include the game.
                self.db.update('generation', generation['_id'], generation)

                # make the bots play a game.
                results = game.tournamentMatch(cpu1, cpu2, game_id, self.db, debug)

                # allocate points for each player.
                players[i], players[rand] = self.allocatePoints(results, players[i], players[rand])
                    
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

    @staticmethod
    def randomPlayerRange(val, numberOfPlayers):
        # choose a random number between 1 and the number of players.
        rand = randint(0, numberOfPlayers-1)
        while (rand == val):
            rand = randint(0, numberOfPlayers-1)
        return rand
    
    @staticmethod
    def allocatePoints(results, blackPlayer, whitePlayer):
        # allocates points to players dependent on the game results.
        if results["Winner"] == Black:
            blackPlayer.points += 1
            whitePlayer.points -= 2
        elif results["Winner"] == White:
            blackPlayer.points -= 2
            whitePlayer.points += 1
        return (blackPlayer, whitePlayer)
    
    @staticmethod
    def generateGameID(generationID, i,j, cpu1, cpu2):
        # choose a random number between 1 and the number of players.
        IDPadding = generationID +"_"+ str(i) +"_"+ str(j)
        game_id = IDPadding + cpu1.id + cpu2.id
        return game_id

def run():
    configpath = "config.json"
    ga = Generator(configpath,10, 5)
    ga.runGenerations()

if __name__ == "__main__":
    run()