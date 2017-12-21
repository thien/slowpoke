# import self packages
import population as pop
import agent
import random
import slowpoke as sp
import mongo
import game
import genetic as ga

# import libraries
import datetime
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
    self.totalGamesPerGen = ((options['Population'] ^ 2) - options['Population'])

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

  def chooseOpponment(self, list):
    return False

  def Tournament(self, population):
    """
    Tournament; this determines the best players out of them all.
    returns the players in order of how good they are.
    """
    gamePool = []
    # initiate game results round robin style (where each player plays as b and w)
    for player_id in population.currentPopulation:
      for oppoment_id in population.currentPopulation:
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
            'black' : population.players[player_id],   
            'white' : population.players[oppoment_id],
            'dbURI' : False,
            'debug' : False,
          }
          # add it to the list of games that need to be played.
          gamePool.append(game)
    # run game simulations.
    results = []
    # close number of processes when map is done.
    with multiprocessing.Pool(processes=self.processors) as pool:
      results = pool.map(self.gameWorker, gamePool)
    self.displayDebugInfo(self.debugInfo())

    # when the pool is done with processing, process the results.
    for i in results:
      # allocate scores at the end of the match
      population.allocatePoints(i['game'], i['black'], i['white'])

    # order the players by how good they are.
    print("TING DONE FAM")
    population.sortCurrentPopulationByPoints()
    # add champion to the list.
    population.addChampion()
    # return population object
    return population

  def runGenerations(self):
    # generate the initial population.
    population = pop.Population(self.population, self.plyDepth)
    # loop through the generations.
    for i in range(self.generations):
      # increment generation count
      self.currentGeneration = i
      self.currentGenStartTime = datetime.datetime.now().timestamp()
      # reset game count statistics prior to running
      self.GamesFinished = 0
      self.GamesQueued = 0
      # store the population into mongo
      # population_IDs = population.writePopulationToDB()
      # initiate generation dict so that we can store it on mongo.
      # generation_info = {
      #   '_id' : i,
      #   'timestamp' : str(datetime.datetime.now()),
      #   'count' : i,
      #   'population' : population_IDs,
      #   'games' : [],
      #   'results' : {}
      # }
      # # write generation info to mongo.
      # self.db.write('generation', generation)
      # initiate timestamp
      startTime = datetime.datetime.now()
      # make bots play each other.
      population = self.Tournament(population)
      
      population.printCurrentPopulationByPoints()
      # compute champion games (runs independently of others)
      self.runChampions()
      # get the best players and generate a new population from them.
      population.generateNextPopulation()
      # initiate end timestamp and add time difference length to list.
      timeDifference = (datetime.datetime.now() - startTime).total_seconds()
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
    results = game.tournamentMatch(i['black'], i['white'], i['game_id'], i['dbURI'], i['debug'])
    data = {
      'game' : results,
      'black' : i['black'].id,
      'white':  i['white'].id
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
    debugList.append(["Gen. Progress*", str(round(PercentageEst*100,2))+"%"])
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

  # @staticmethod
  # def allocatePoints(results, blackPlayer, whitePlayer):
  #   # allocates points to players dependent on the game results.
  #   if results["Winner"] == Black:
  #     blackPlayer.points += 1
  #     whitePlayer.points -= 2
  #   elif results["Winner"] == White:
  #     blackPlayer.points -= 2
  #     whitePlayer.points += 1
  #   return (blackPlayer, whitePlayer)
  
  @staticmethod
  def generateGameID(generationID, i,j, cpu1, cpu2):
    # choose a random number between 1 and the number of players.
    IDPadding = generationID +"_"+ str(i) +"_"+ str(j)
    game_id = IDPadding + cpu1.id + cpu2.id
    return game_id