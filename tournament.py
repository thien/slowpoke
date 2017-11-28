# import self packages
import agent
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


import multiprocessing


# TODO: need to make the champion play previous champions and get scores

Black, White, empty = 0, 1, -1

class Generator:
    """
    Configuration Functions
    """

    def __init__(self, configpath, generationCount=100, population=15):
        # Declare base information
        self.generations = generationCount
        self.population = population
        self.processors = multiprocessing.cpu_count()-1
        self.champions = {}
        # Initiate other information
        self.config = self.loadJSONConfig(configpath)
        # once we have the config file we can proceed and initiate our
        # MongoDB connection.
        self.initiateMongoConnection()

    def loadJSONConfig(self, filepath):
        """
        Loads config.json
        """
        try:
            with open(filepath) as json_file:
                data = json.load(json_file)
            return data
        except:
            data = {'MongoURI' : ""}
            return data

    def initiateMongoConnection(self):
        self.db = mongo.Mongo()
        try:
            self.db.initiate(self.config['MongoURI'])
        except:
            pass

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

    def writePopulationToDB(self, population):
        """
        Stores the population into Mongo. 
        """
        keys = []
        if self.db.connected:
            for i in population:
                if self.db.checkPlayerExists(i.id) == False:
                    entry = self.db.write('players', i.getDict())
                    keys.append(entry)
                else:
                    keys.append(i.id)
        return keys

    def Tournament(self, players, generation):
        """
        Tournament; this determines the best players out of them all.
        returns the players in order of how good they are.
        """
        print(chr(27) + "[2J") # clear screen

        gamePool = []

        # create game count statistics
        gameCount, totalGames = 0, ((len(players) ^ 2) - len(players))
        
        # initiate game results
        for i in range(len(players)):
            for j in range(len(players)):
                if i != j:
                    # increment game count.
                    gameCount+= 1
                    # set up debug file.
                    debug = {
                        'genCount' : generation['count'],
                        'printDebug' : True,
                        'printBoard' : False,
                        'genID' : generation['_id'],
                        'gameCount' : gameCount,
                        'totalGames' : totalGames
                    }
        
                    # choose a random number between 1 and the number of players.
                    rand = self.randomPlayerRange(i, len(players))

                    # cpu1 is black, cpu2 is white
                    cpu1 = players[i]
                    cpu2 = players[rand]

                    # generate ID for the game (so we can store it on Mongo)
                    game_id = self.generateGameID(generation['_id'], i, j, cpu1, cpu2)
                    generation['games'].append(game_id)
                    
                    # update generations entry in mongo to include the game.
                    self.db.update('generation', generation['_id'], generation)

                    # add the game to the list.
                    game = {
                        'game_id' : game_id,
                        'cpu1' : cpu1,
                        'cpu2' : cpu2,
                        'dbURI' : self.config['MongoURI'],
                        'debug' : debug,
                        'i' : i,
                        'rand' : rand
                    }
                    
                    # add it to the list of games that need to be played.
                    gamePool.append(game)
        
        # initiate game pool.
        pool = multiprocessing.Pool(processes=self.processors)

        # run game simulations.
        results = pool.map(self.gameWorker, gamePool)

        # when the pool is done with processing, process the results.
        for game in results:
            # do stuff with job
            blackP = game['black']
            whiteP = game['white']
            # allocate scores at the end of the match
            players[blackP], players[whiteP] = self.allocatePoints(game['game'], players[blackP], players[whiteP])

        # order the players by how good they are.
        players_ranked = sorted(players, key=operator.attrgetter('points'),reverse=True)

        # iterate through the players and add their results to the generation store.
        for i in players_ranked:
            generation['results'][i.id] = i.points
        
        # update results in database.
        self.db.update('generation', generation['_id'], generation)
        
        # print results
        for i in range(len(players_ranked)):
            print(players_ranked[i].id, players_ranked[i].points)

        # add champion to the list.
        champions[generation['count']] = players_ranked[0]
        return players_ranked

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
                'games' : [],
                'results' : {}
            }
            # write generation info to mongo.
            self.db.write('generation', generation)

            # print debug.
            print("Playing on Generation:", i)
            # make bots play each other.
            players = self.Tournament(population, generation)
            # get the best players and generate a new population from them.
            population = ga.generateNewPopulation(players, self.population)
            # compute champion games asyncronously.

    @staticmethod
    def gameWorker(i):
        results = game.tournamentMatch(i['cpu1'], i['cpu2'], i['game_id'], i['dbURI'], i['debug'])
        data = {
            'game' : results,
            'black' : i['i'],
            'white':  i['rand']
        }
        return data

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
    ga = Generator(configpath,20, 3)
    ga.runGenerations()

if __name__ == "__main__":
    run()