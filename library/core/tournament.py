# import self packages
import core.population as pop
import core.game as game

import agents.slowpoke as sp
import agents.agent as agent
import core.mongo as mongo

# import libraries
import datetime
import numpy as np
import random
import multiprocessing
import os
import json

# ignore runtime warnings
np.warnings.filterwarnings('ignore')

import statistics

# Piece values on board
Black, White, empty = 0, 1, -1
# Blondie was 1,0,-2
WinPt, DrawPt, LosePt = 2, 0, -1
ChampWinPt, ChampDrawPt, ChampLosePt = 1,0,-1

def optionDefaults(options):
  # adds default options if they are absent from options.
  defaultOptions = {
    'debugMode' : False,
    'mongoConfigPath' : 'config.json',
    'plyDepth' : 4,
    'NumberOfGenerations' : 200,
    'Population' : 15,
    'printStatus' : True,
    'connectMongo' : False,
    'NumberOfGamesPerPlayer' : 5,
    'resultsLocation' : os.path.join("..", "results")
  }
  for i in defaultOptions.keys():
    if i not in options:
      options[i] = defaultOptions[i]
  return options

class Generator:
  def __init__(self, options):
    # initialise default variables when needed.
    options = optionDefaults(options)
    self.isDebugMode = options['debugMode']
    # Declare base information
    self.plyDepth = options['plyDepth']
    self.generations = options['NumberOfGenerations']
    self.populationSize = options['Population'] #number of players
    # generate the initial population.
    self.population = pop.Population(self.populationSize, self.plyDepth, self.isDebugMode)
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
    self.playPreviousChampCount = 5
    self.champGamesRoundsCount = 6 # should always be even and at least 2.
    self.NumberOfGamesPerPlayer = options['NumberOfGamesPerPlayer']
    self.progress = []

    self.previousGenerationRankings = None
    self.previousChampPointList = None
    # Initiate other information
    self.processors = multiprocessing.cpu_count()-1
    self.config = self.loadJSONConfig(options['mongoConfigPath'])
    self.mongoConnected = options['connectMongo']
    self.totalGamesPerGen = ((options['Population'] ^ 2) - options['Population'])

    # placeholder values
    self.gameIDCounter = 0
    # once we have the config file we can proceed and initiate our MongoDB connection.
    self.initiateMongoConnection()
    # we also want to save the stats offline
    self.generationStats = []
    self.folderName = str(self.cleanDate(self.StartTime, True)) +" " + str(self.plyDepth) + "ply"
    self.saveLocation = os.path.join(options['resultsLocation'],self.folderName)
    # self.saveLocation = os.path.join(options['resultsLocation'],self.cleanDate(self.StartTime, True))
    # generate charts as we go?
    self.generateChartsEveryRound = True

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

  def Tournament(self):
    """
    Tournament; this determines the best players out of them all.
    returns the players in order of how good they are.
    """
    gamePool = []
    # initiate game results round robin style (where each player plays as b and w)
    for player_id in self.population.currentPopulation:
      for x in range(self.NumberOfGamesPerPlayer):
        oppoment_id = player_id
        while oppoment_id == player_id:
          oppoment_id = random.choice(self.population.currentPopulation)
        # make sure they're not playing themselves
        if player_id != oppoment_id:
          # generate ID for the game
          game_id = self.gameIDCounter
          self.gameIDCounter += 1
          # increment game count.
          self.GamesQueued += 1
          # create game variables
          game = {
            'game_id' : game_id,
            'black' : self.population.players[player_id],   
            'white' : self.population.players[oppoment_id],
            'dbURI' : False,
            'debugInfo' : False,
          }
          # add it to the list of games that need to be played.
          gamePool.append(game)
    # run game simulations.
    results = []
    # close number of processes when map is done.
    threadCount = self.processors
    if self.processors > len(gamePool):
      threadCount = len(gamePool)
    with multiprocessing.Pool(processes=threadCount) as pool:
      results = pool.map(self.gameWorker, gamePool)
      pool.close()
      pool.join()
  

    self.displayStatusInfo()

    # when the pool is done with processing, process the results.
    for i in range(len(results)):
      self.population.allocatePoints(results[i]['game'], results[i]['black'], results[i]['white'])
      # merge winning players move caches
      if results[i]['game']['Winner'] == Black:
        bCache = self.population.players[results[i]['black']].bot.cache
        self.population.players[results[i]['black']].bot.cache = self.merge_dicts(bCache, results[i]['black_cache'])
      elif results[i]['game']['Winner'] == White:
        wCache = self.population.players[results[i]['white']].bot.cache
        self.population.players[results[i]['white']].bot.cache = self.merge_dicts(wCache, results[i]['white_cache'])
      # nullify the cache since its not needed anymore
      results[i]['black_cache'] = None
      results[i]['white_cache'] = None

    self.population.sortCurrentPopulationByPoints()
    self.population.addChampion()
    return (self.population, results)


  """
  This function is called for every generation.
  """
  def runGenerations(self):
    # loop through the generations.
    for i in range(self.generations):
      print("Initiating generation",i)
      # increment generation count
      self.currentGeneration = i
      self.currentGenStartTime = datetime.datetime.now().timestamp()
      # reset game count statistics prior to running
      self.GamesFinished = 0
      self.GamesQueued = 0
      # initiate timestamp
      startTime = datetime.datetime.now()
      # make bots play each other.
      print("READY")
      self.population, generationResults = self.Tournament()
      self.previousGenerationRankings = self.population.printCurrentPopulationByPoints()
      
      # print(self.population.printCurrentPopulationByPoints())
      # compute champion games (runs independently of others)
      self.runChampions()
      # save champions to file
      self.population.saveChampionsToFile(self.saveLocation)
      # save genomic details
      self.population.savePopulationGenomes(self.saveLocation)
      # get the best players and generate a new population from them.
      self.population.generateNextPopulation()
      self.populationSize = self.population.count
      # initiate end timestamp and add time difference length to list.
      timeDifference = (datetime.datetime.now() - startTime).total_seconds()
      self.GenerationTimeLengths = np.hstack((self.GenerationTimeLengths, timeDifference))
      # need to store the results of this into a json file!
      self.generationStats.append({
        'stats' : [(str(i[0]), str(i[1])) for i in self.statusInfo()],
        'games' : generationResults,
        'durationInSeconds' : str(timeDifference)
      })
      self.saveTrainingStatsToJSON(self.saveLocation, self.generationStats)
      if self.generateChartsEveryRound:
        self.generateStats()

  def nukeCache(self):
    for i in self.population:
      self.population[i].bot.cache = {}    
    
  def generateStats(self):
    # create statistics
    stats = statistics.Statistics(self.folderName)
    stats.loadStatisticsFile()
    stats.saveCharts()
    print("I made some charts!")

  def saveTrainingStatsToJSON(self, saveLocation, stats):
    # check save directory exists prior to saving
    if not os.path.isdir(saveLocation):
      os.makedirs(saveLocation)

    filename = 'statistics.json'
    with open(os.path.join(saveLocation, filename), 'w') as outfile:
      json.dump(stats, outfile)

  def poolChampGame(self, info):
    self.displayStatusInfo()
    blackPlayer = self.population.players[info['Players'][0]]
    whitePlayer = self.population.players[info['Players'][1]]
    results = game.tournamentMatch(blackPlayer,whitePlayer)
    if results['Winner'] == info['champColour']:
      # champion won.
      return ChampWinPt
    elif results['Winner'] == empty:
      return ChampDrawPt
    else:
      return ChampLosePt
    self.displayStatusInfo()

  def createChampGames(self):
    currentChampID = self.population.champions[-1]
    champGames = []
    gameRound = int(self.champGamesRoundsCount/2)
    # playback counter
    playcounter = np.size(self.progress)
    if playcounter > self.playPreviousChampCount:
      playcounter = self.playPreviousChampCount

    for i in range(playcounter):
      previousChampID = self.population.champions[-i+1]
      # set player colours
      info = {
        'Players' : (currentChampID,previousChampID),
        'champColour' : Black
      }
      
      for j in range(gameRound):
        champGames.append(info)
      # reverse players
      info = {
        'Players' : (previousChampID,currentChampID),
        'champColour' : White
      }
      for j in range(gameRound):
        champGames.append(info)
    return champGames

  def runChampions(self):
    """
    These champion games are called at the end of every generation
    and are used to determine the progress of the bots.
    """
    self.AreChampionsPlaying = True
    self.displayStatusInfo()

    # check if theres more than 5 champions.
    if len(self.population.champions) > 2:
      # create list of games to play
      champGames = self.createChampGames()
      # close number of processes when map is done.
      results = []

      numberOfChampgames = len(champGames)
      threadCount = self.processors
      if self.processors > numberOfChampgames:
        threadCount = numberOfChampgames
    
      with multiprocessing.Pool(processes=threadCount) as pool:
        results = pool.map(self.poolChampGame, champGames)
        pool.close()
        pool.join()

      # split results into equal segments
      l = results
      n = self.playPreviousChampCount
      results = [l[i:i + n] for i in range(0, len(l), n)]
      # calculate the gradient of the scores.
    
      medians = []
      for i in results:
        medians.append(np.mean(i))

      self.previousChampPointList = medians

      # compute new champ points compared to previous champ
      newChampPoints = np.mean(medians)
      self.cummulativeScore += newChampPoints
      # store points.
      self.progress.append(newChampPoints)
      self.population.players[self.population.champions[-1]].champScore = newChampPoints
      self.population.players[self.population.champions[-1]].champRange = results
    else:
      # theres only one champion, don't play.
      self.progress.append(0)
    self.AreChampionsPlaying = False
    self.displayStatusInfo()

  def gameWorker(self,i):
    self.displayStatusInfo()
    timeStart = datetime.datetime.now().timestamp()
    results = game.tournamentMatch(i['black'], i['white'], i['game_id'], i['dbURI'], i['debugInfo'])
    bSubset = {}
    wSubset = {}  
    # get a subset of the caches
    if i['black'].bot.enableCache:
      bCache = i['black'].bot.cache
      wCache = i['white'].bot.cache
      for _ in range(100):
        randb = random.choice(list(bCache.keys()))
        bSubset[randb] = bCache[randb]
        randw = random.choice(list(wCache.keys()))
        wSubset[randw] = wCache[randw]
      # nuke cache
      i['black'].bot.cache = {}
      i['white'].bot.cache = {}
    data = {
      'game' : results,
      'black' : i['black'].id,
      'white':  i['white'].id,
      'black_cache' : bSubset,
      'white_cache' : wSubset,
      'duration' : str(self.cleanDate(datetime.datetime.now().timestamp() - timeStart))
    }
    self.displayStatusInfo()
    # print(i['black'].bot.cache)
    return data

  def statusInfo(self):
    currentTime = datetime.datetime.now().timestamp()
    recent_scores = self.progress[-7:]
    
    averageGenTimeLength = np.mean(self.GenerationTimeLengths)

    PercentageEst = 0
    if np.isnan(averageGenTimeLength) == False:
      PercentageEst = (currentTime - self.currentGenStartTime) / averageGenTimeLength

    numGens = np.size(self.progress)
    remainingGenTime = averageGenTimeLength - (currentTime - self.currentGenStartTime)
    RemainingGenCount = self.generations-numGens

    # calculate current run time
    currentRunTime = datetime.datetime.now() - datetime.datetime.fromtimestamp(self.StartTime)
    # calculate remaining time
    EstRemainingTime = (RemainingGenCount * averageGenTimeLength) + np.sum(self.GenerationTimeLengths) - currentRunTime.total_seconds()

    EstEndDate = EstRemainingTime + self.StartTime + currentRunTime.total_seconds()

    messsages = []

    messsages.append(["Generation", str(numGens)+"/"+str(self.generations)])
    messsages.append(["Population", self.populationSize])
    messsages.append(["Ply Depth", self.plyDepth])
    messsages.append(["Connected To Mongo", self.mongoConnected])
    messsages.append(["Cores Utilised", self.processors])
    # start and end dates
    messsages.append([" ", " "])
    messsages.append(["Test Start Date", self.cleanDate(self.StartTime, True)])
    messsages.append(["Current Runtime", currentRunTime])
    messsages.append(["Test End Date*", self.cleanDate(EstEndDate,True)])
    messsages.append(["Remaining Test Time*", self.cleanDate(EstRemainingTime)])
    # Time info
    messsages.append([" ", " "])
    messsages.append(["Mean Game Time", self.cleanDate(averageGenTimeLength)])
    messsages.append(["Gen. Progress*", str(round(PercentageEst*100,2))+"%"])
    messsages.append(["Remaining Gen. Time*", self.cleanDate(remainingGenTime)])
    # champion info
    messsages.append([" ", " "])
    messsages.append(["Champions Currently Playing?", self.AreChampionsPlaying])
    messsages.append(["Previous Score", self.LastChampionScore])
    messsages.append(["Cummulative Score", self.cummulativeScore])

    avgRecentScores = 0
    if len(recent_scores) > 0:
      avgRecentScores = np.mean(recent_scores)
    messsages.append(["Average Growth", avgRecentScores])
    try:
      messsages.append(["Recent Scores", [ "{:0.2f}".format(x) for x in recent_scores ]])
      messsages.append(["Prev. Champ Point Range",[ "{:0.2f}".format(x) for x in self.previousChampPointList ]])
    except:
      pass
    messsages.append([" ", " "])
    messsages.append(["Previous Scoreboard", " "])
    messsages.append([self.previousGenerationRankings, ""])
    messsages.append(["",""])
    messsages.append(["Debug Mode:", self.isDebugMode])
    
    return messsages

  def displayStatusInfo(self):
    # # clear screen
    # print(chr(27) + "[2J")
    print('\033c', end=None)
    print("SLOWPOKE")
    print("----------------------")
    for i in self.statusInfo():
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
    except:
      return 0
  
  @staticmethod
  def generateGameID(generationID, i,j, cpu1, cpu2):
    # choose a random number between 1 and the number of players.
    IDPadding = generationID +"_"+ str(i) +"_"+ str(j)
    game_id = IDPadding + cpu1.id + cpu2.id
    return game_id

  @staticmethod
  def merge_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z