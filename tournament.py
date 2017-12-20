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
import numpy as np
from random import randint
import elo

import multiprocessing

Black, White, empty = 0, 1, -1

def optionDefaults(options):
    # adds default options if they are absent from options.
    defaultOptions = {
        'mongoConfigPath' : 'config.json',
        'plyDepth' : 4,
        'NumberOfGenerations' : 200,
        'Population' : 15,
        'printStatus' : True,
        'connectMongo' : False
    }
    for i in defaultOptions.keys():
        if i not in options:
            options[i] = defaultOptions[i]
    return options

class Generator:
    def __init__(self, options):
        # initialise default variables when needed.
        options = optionDefaults(options)

        # Declare base information
        self.plyDepth = options['plyDepth']
        self.generations = options['NumberOfGenerations']
        self.population = options['Population'] #number of players

        # time handlers
        self.StartTime = datetime.datetime.now().timestamp()
        self.AverageGameTime = 0
        self.AverageGenrationLength = 0
        self.RemainingTime = 0
        self.EstDateFinished = 0
        self.GenerationTimeLengths = np.array([])
        self.currentGenStartTime = datetime.datetime.now().timestamp()

        # current generation game counts
        self.GamesFinished = 0
        self.GamesQueued = 0
        self.CurrentGeneration = 0

        # champions
        self.AreChampionsPlaying = False
        self.LastChampionScore = 0
        self.cummulativeScore = 0
        self.AverageChampionGrowth = 0
        self.RecentChampionScores = 0

        self.champions = {}
        self.progress = []
        
        # Initiate other information
        self.processors = multiprocessing.cpu_count()-1
        self.config = self.loadJSONConfig(options['mongoConfigPath'])
        self.mongoConnected = options['connectMongo']
        # placeholder values
        self.gameIDCounter = 0
        # once we have the config file we can proceed and initiate our MongoDB connection.
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
            if self.mongoConnected:
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
            bot = sp.Slowpoke(self.plyDepth)
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
        gamePool = []
        
        # create game count statistics
        totalGames = ((len(players) ^ 2) - len(players))

        # initiate game results round robin style (where each player plays as b and w)
        for i in range(len(players)):
            for j in range(len(players)):
                if i != j:
                    # increment game count.
                    self.GamesQueued += 1
                    # set up debug file.
                    debug = {
                        'genCount' : generation['count'],
                        'printDebug' : False,
                        'printBoard' : False,
                        'genID' : generation['_id'],
                        'gameCount' : self.GamesQueued,
                        'totalGames' : totalGames
                    }
                    # choose a random number between 1 and the number of players.
                    rand = self.randomPlayerRange(i, len(players))
                    # cpu1 is black, cpu2 is white
                    cpu1 = players[i]
                    cpu2 = players[rand]
                    # generate ID for the game (so we can store it on Mongo)
                    game_id = self.gameIDCounter
                    self.gameIDCounter += 1
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
                    if self.mongoConnected == False:
                        game['dbURI'] = False
                    # add it to the list of games that need to be played.
                    gamePool.append(game)
                    
        # run game simulations.
        results = []
        # close number of processes when map is done.
        with multiprocessing.Pool(processes=self.processors) as pool:
            results = pool.map(self.gameWorker, gamePool)
        self.displayDebugInfo(self.debugInfo())

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
            # increment generation count
            self.currentGeneration = i
            self.currentGenStartTime = datetime.datetime.now().timestamp()
            # reset game count statistics prior to running
            self.GamesFinished = 0
            self.GamesQueued = 0
            # store the population into mongo
            population_IDs = self.writePopulationToDB(population)
            # initiate generation dict so that we can store it on mongo.
            generation = {
                '_id' : i,
                'timestamp' : str(datetime.datetime.now()),
                'count' : i,
                'population' : population_IDs,
                'games' : [],
                'results' : {}
            }
            # write generation info to mongo.
            self.db.write('generation', generation)
            # initiate timestamp
            startTime = datetime.datetime.now()
            # make bots play each other.
            players = self.Tournament(population, generation)
            # compute champion games
            self.runChampions()
            # get the best players and generate a new population from them.
            population = ga.generateNewPopulation(players, self.population)
            # default new population values.
            for player in population:
                player.elo = int(self.progress[-1])
                player.points = 0
            # initiate end timestamp and add time difference length to list.
            endTime = datetime.datetime.now()
            timeDifference = endTime - startTime
           
            # print(timeDifference)
            timeDifference = timeDifference.total_seconds()
            self.GenerationTimeLengths = np.hstack((self.GenerationTimeLengths, timeDifference))

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
        blackPlayer,whitePlayer = info['Players']
        results = game.tournamentMatch(self.champions[blackPlayer],self.champions[whitePlayer])
        if results['Winner'] == info['champColour']:
            # champion won.
            return 1
        elif results['Winner'] == empty:
            return 0
        else:
            return -1

    def createChampGames(self):
        currentChampID = self.currentGeneration
        previousChampID = self.currentGeneration - 1

        champGames = []
        # set player colours
        info = {
            'Players' : (currentChampID,previousChampID),
            'champColour' : Black
        }
        champGames.append(info)
        champGames.append(info)
        champGames.append(info)
        # reverse players
        info = {
            'Players' : (previousChampID,currentChampID),
            'champColour' : White
        }
        champGames.append(info)
        champGames.append(info)
        champGames.append(info)
        return champGames

    def runChampions(self):
        """
        These champion games are called at the end of every generation
        and are used to determine the progress of the bots.
        """
        self.AreChampionsPlaying = True
        self.displayDebugInfo(self.debugInfo())
        # load champs.
        champs = self.champions
  
        # check how many champs there are.
        lenChamps = len(list(champs.keys()))
        # if theres more than 2:
        if lenChamps > 1:
            # create list of games to play
            champGames = self.createChampGames()
            # compute the games.

            # close number of processes when map is done.
            results = []
            with multiprocessing.Pool(processes=self.processors) as pool:
                results = pool.map(self.poolChampGame, champGames)
            # compute new champ points compared to previous champ
            newChampPoints = sum(results)
            self.cummulativeScore += newChampPoints
            # store points.
            self.progress.append(newChampPoints)
            self.db.update('performance', self.performanceID, {"progress":self.progress})
        else:
            # theres only one champion, it defaults at 1200 anyway.
            self.progress.append(0)
            self.performanceID = self.db.write('performance', {"progress":self.progress})
        self.AreChampionsPlaying = False
        self.displayDebugInfo(self.debugInfo())

    def gameWorker(self,i):
        self.displayDebugInfo(self.debugInfo())
        results = game.tournamentMatch(i['cpu1'], i['cpu2'], i['game_id'], i['dbURI'], i['debug'])
        data = {
            'game' : results,
            'black' : i['i'],
            'white':  i['rand']
        }
        self.GamesFinished += 1
        self.displayDebugInfo(self.debugInfo())
        return data

    def debugInfo(self):
        currentTime = datetime.datetime.now().timestamp()
        recent_scores = self.progress[-10:]
        averageGenTimeLength = np.mean(self.GenerationTimeLengths)

        PercentageEst = 0
        if np.isnan(averageGenTimeLength) == False:
            PercentageEst = (currentTime - self.currentGenStartTime) / averageGenTimeLength

        numGens = np.size(self.progress)
        remainingGenTime = averageGenTimeLength - (currentTime - self.currentGenStartTime)
        RemainingGenCount = self.generations-numGens
        EstRemainingTime = (RemainingGenCount * averageGenTimeLength) + np.sum(self.GenerationTimeLengths)
        EstEndDate = EstRemainingTime + self.StartTime
        currentRunTime = datetime.datetime.now() - datetime.datetime.fromtimestamp(self.StartTime)
        debugList = []
    
        debugList.append(["Generation", str(numGens)+"/"+str(self.generations)])
        debugList.append(["Population", self.population])
        debugList.append(["Ply Depth", self.plyDepth])
        debugList.append(["Connected To Mongo", self.mongoConnected])
        debugList.append(["Cores Utilised", self.processors])
        # start and end dates
        debugList.append([" ", " "])
        debugList.append(["Test Start Date", self.cleanDate(self.StartTime, True)])
        debugList.append(["Current Runtime", currentRunTime])
        debugList.append(["Test End Date*", self.cleanDate(EstEndDate, True)])
        debugList.append(["Remaining Test Time*", self.cleanDate(EstRemainingTime)])
        # Time info
        debugList.append([" ", " "])
        debugList.append(["Mean Game Time", self.cleanDate(averageGenTimeLength)])
        debugList.append(["Gen. Progress*", round(PercentageEst*100,2)])
        debugList.append(["Remaining Gen. Time*", self.cleanDate(remainingGenTime)])
        # current Generation Info
        # debugList.append([" ", " "])
        # debugList.append(["No. of games computed", str(self.GamesFinished)+"/"+str(self.GamesQueued)])
        # champion info
        debugList.append([" ", " "])
        debugList.append(["Champions Currently Playing?", self.AreChampionsPlaying])
        debugList.append(["Previous Score", self.LastChampionScore])
        debugList.append(["Cummulative Score", self.cummulativeScore])
        debugList.append(["Average Growth", np.mean(recent_scores)])
        debugList.append(["Recent Scores", recent_scores])
        return debugList

    def displayDebugInfo(self, debugList):
        # # clear screen
        print(chr(27) + "[2J")
        print("SLOWPOKE")
        print("----------------------")
        for i in debugList:
            print("{0:30} {1}".format(str(i[0]), str(i[1])))
        print("----------------------")

    @staticmethod
    def cleanDate(timestamp, unixDefault=False):
        try:
            if unixDefault == True:
                k = datetime.datetime.fromtimestamp(timestamp)
                return k.strftime('%Y-%m-%d %H:%M:%S')
            else:   
                start = datetime.datetime.fromtimestamp(0)
                k = datetime.datetime.fromtimestamp(timestamp)
                magic = k - start
                return magic
        except Exception as e:
            return 0

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