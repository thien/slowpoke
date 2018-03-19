"""
This program evaluates the performance of the simulator.
"""

import os
import json
import numpy as np
import multiprocessing
import csv
import datetime

import play as p
import agents.agent as agent

class Evaluate:
  def __init__(self, date, ply, defaultResultsPath=None):
    self.date = date
    self.path = os.path.join("..", "results")
    if defaultResultsPath:
      self.path = defaultResultsPath
    self.directory = os.path.join(self.path, self.date)
    self.champFolderName = "champions"
    # container for the agents
    self.agents = {}
    # container for statistics
    self.statistics = {}
    # used as parameters for the sims.
    self.gameOpts = {
      'show_dialog' : False,
      'show_board' : False,
      'human_white' : False,
      'human_black' : False,
      'preload_moves' : []  
    }
    # cpu information
    self.cores = multiprocessing.cpu_count()
    # simulation specific information
    self.numberOfGames = 4
    if self.cores > 64: self.numberOfGames = 256

    self.ply = ply
    # split to every nth parttioned player.
    self.choiceRange = 5
    # file save information
    self.filename = "gm_stats"


  def loadChampions(self, extensions=True):
    champsPath = os.path.join(self.directory,self.champFolderName)
    print("Loading Agents.. ", end="")
    files = os.listdir(champsPath)
    items = sorted([int(x.split(".json")[0]) for x in files])
    # get the id of the best agent and the worst.
    gmID = items[-1]
    agentCount = len(items)

    tests = [items[0]]
    for i in range(self.choiceRange-1):
      tests.append(int(agentCount*((i+1))/self.choiceRange))

    self.gm_id = gmID
    # load the gold master agent.
    self.agents['gm'] = self.loadAgentFile(self.ply, gmID, champsPath)
    
    # load the opponments
    for i in tests:
      # create agentString
      agent_ID = "generation-" + str(i)
      # load that agent
      self.agents[agent_ID] = self.loadAgentFile(self.ply, i, champsPath)

    if extensions:
      # let's import other agents too
      self.agents['random'] = p.loadPlayerClass('magikarp')
      self.agents['pure_mcts'] = p.loadPlayerClass('geodude')

    # now, we're done!
    print("Done.")
    print("There are", len(self.agents.keys()), "loaded.")

  def createGames(self):
    # we'll make a list of games that the GM will play against.
    games = []
    for agent in self.agents.keys():
      if agent != "gm":
        games.append(["gm", agent])
    return games

  def evaluate(self,games):
    ent = {}
    for x in games:
      # set colour codes
      black, white = 0,1
      ev_ID = x[black] + "_vs_" + x[white]
      print("Calculating",ev_ID)
      # create score container
      ent[ev_ID] = {
        'wins' : 0,
        'losses' : 0,
        'draws' : 0,
        'as_black' : {
          'wins' : 0,
          'losses' : 0,
          'draws' : 0
        },
        'as_white' : {
          'wins' : 0,
          'losses' : 0,
          'draws' : 0
        }
      }
      
      # iterate through games as black and white
      for j in range(0,2):
        scores = []
        # make sure they switch for black and white
        gID = "as_black"
        if j == 1:
          black, white = 1,0
          gID = "as_white"
        # initialise entry for the game
        ent[ev_ID][gID] = {} 

        gamePool = []
        for i in range(0,int(self.numberOfGames/2)):
          # add game to list of games to play
          gamePool.append({
            'black' : self.initAgentClass(x[black], self.agents[x[black]]),
            'white' : self.initAgentClass(x[white], self.agents[x[white]]),
            'gameOpt' : self.gameOpts
          })

        # create game pool.
        with multiprocessing.Pool(processes=self.cores) as pool:
          scores = pool.map(self.gameWorker, gamePool)
        
        # now that's done, we can now tally up the stats
        # count number of wins, losses, draws for given side.
        ent[ev_ID][gID]['wins'] = scores.count(black)
        ent[ev_ID][gID]['losses'] = scores.count(white)
        ent[ev_ID][gID]['draws'] = scores.count(-1)

        # add to overall w/l/d
        ent[ev_ID]['wins'] += ent[ev_ID][gID]['wins']
        ent[ev_ID]['losses'] += ent[ev_ID][gID]['losses']
        ent[ev_ID]['draws'] += ent[ev_ID][gID]['draws']
        # now we need to add this to our results file.
        self.saveResultsToJson(ent)
      print(ev_ID,ent[ev_ID])
    return ent

  """
  easy command to save to json.
  """
  def saveResultsToJson(self, ent):
    filename = self.filename+".json"
    filepath = os.path.join(self.directory,filename)
    with open(filepath, 'w') as outfile:
      json.dump(ent, outfile)

  """
  This gets called by the map (as part of multithread)
  """
  @staticmethod
  def gameWorker(i):
    return p.runGame(i['black'], i['white'], i['gameOpt']).winner

  @staticmethod
  def loadAgentFile(ply,pID,location):
    # returns an numpy array
    filetype = ".json"
    filename = str(pID) + filetype
    filepath = os.path.join(location,filename)
    # print("loading", filepath)
    data = json.load(open(filepath))[str(pID)]['coefficents']
    coefs = np.array(data)
    # generate a class and return that
    return p.genSlowpokeClass(plyCount=ply, weights=coefs)

  @staticmethod
  def initAgentClass(id, bot):
    return (agent.Agent(bot),id)

if __name__ == '__main__':
  date = "2018-03-19 16:42:47"
  ply = 1
  s = Evaluate(date,ply)
  s.loadChampions()
  games = s.createGames()
  s.evaluate(games)