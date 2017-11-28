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
import elo

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
        self.champions = {}
        self.progress = []
        self.processors = multiprocessing.cpu_count()-1
        self.pool = multiprocessing.Pool(processes=self.processors)
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

    def ELOShift(self, winner, black, white):
        b_exp = elo.expected(black.elo, white.elo)
        w_exp = elo.expected(white.elo, black.elo)
        # initiate score outcomes
        b_result = 0
        w_result = 0
        if winner == Black:
            # black wins
            b_result = 1
            pass
        elif winner == White:
            # white wins
            w_result = 1
        else:
            # draw
            b_result = 0.5
            w_result = 0.5
        # calculate elo outcomes
        black.elo = elo.elo(black.elo, b_exp, b_result, k=32)
        white.elo = elo.elo(white.elo, w_exp, w_result, k=32)
        return black, white

    def poolChampGame(self, info):

        bwgame = game.tournamentMatch(info['Black'], info['White'])
        results = {
            "Winner" : bwgame['Winner'],
            "i" : info['i'],
            "j" : info['j']
        }
        return results
        # black, white = self.ELOShift(bwgame, black, white)

    def runChampions(self):
        print("RUNNING CHAMPION SIMULATION")
        # load champs.
        champs = self.champions
        # check how many champs there are.
        lenChamps = len(list(champs.keys()))
        # if theres more than 2:
        if lenChamps > 1:
            champGames = []
            i, j = lenChamps-2, lenChamps-1

            info = {
                'Black' : champs[i],
                'White' : champs[j],
                'i' : i,
                'j' : j
            }
            champGames.append(info)
            champGames.append(info)
            champGames.append(info)
            info = {
                'Black' : champs[j],
                'White' : champs[i],
                'i' : j,
                'j' : i
            }
            champGames.append(info)
            champGames.append(info)

            results = self.pool.map(self.poolChampGame, champGames)

            for r in results:
                champs[r['i']], champs[r['j']] = self.ELOShift(r['Winner'], champs[r['i']], champs[r['j']])

            i, j = lenChamps-2, lenChamps-1
            elo_difference = champs[j].elo - champs[i].elo
            print("ELO DIFFERENCE:", elo_difference)
            # store elo difference.
            self.progress.append(champs[j].elo)
            self.db.update('performance', self.performanceID, {"progress":self.progress})
        else:
            # theres only one champion, it defaults at 1200 anyway.
            self.progress.append(champs[0].elo)
            self.performanceID = self.db.write('performance', {"progress":self.progress})
        # update champion stats on mongo.
        return False

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

        # run game simulations.
        results = self.pool.map(self.gameWorker, gamePool)

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
        self.champions[generation['count']] = players_ranked[0]

        # return list of players sorted by ranking.
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
            # compute champion games
            self.runChampions()
            # default new population values.
            for i in population:
                # default to new default elo value.
                i.elo = int(self.progress[-1])
                i.points = 0


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