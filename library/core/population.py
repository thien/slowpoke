"""Population management, Elo ratings, and evolution."""

from __future__ import annotations

import datetime
import json
import math
import multiprocessing
import operator
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import agents.agent as agent
import agents.slowbro as sb
import core.mongo as mongo
from agents.evaluator.neural import NeuralNetwork


class EloRating:
    """Elo rating system with anchoring and K-factor calibration.

    K=32 for new players (<10 games), K=24 for intermediate (10-50 games),
    K=10 for established (>50 games).
    """
    def __init__(self, k_factor: int = 32, initial_rating: int = 1200) -> None:
        self.k_factor = k_factor
        self.initial_rating = initial_rating

    def get_k_factor(self, games_played: int) -> int:
        """Get K-factor based on number of games played."""
        if games_played < 10:
            return 32  # New player - high volatility
        elif games_played < 50:
            return 24  # Intermediate - moderate volatility
        else:
            return 10  # Established - low volatility
    
    def expected_score(self, player_rating: float, opponent_rating: float) -> float:
        """Calculate expected score for a player against an opponent."""
        return 1 / (1 + 10 ** ((opponent_rating - player_rating) / 400))

    def update_rating(self, player_rating: float, opponent_rating: float, actual_score: float, games_played: int = 0) -> float:
        expected = self.expected_score(player_rating, opponent_rating)
        # Use dynamic K-factor if games_played > 0, otherwise use fixed K-factor
        if games_played > 0:
            k = self.get_k_factor(games_played)
        else:
            k = self.k_factor
        return player_rating + k * (actual_score - expected)

Black, White, empty = 0, 1, -1

WinPt, DrawPt, LosePt = 2, 0, -1

ONIX_ID = -2  # Permanent heuristic-bot fixture, never champion, never parent

class Population:
  def __init__(self, numberOfPlayers: int, plyDepth: int, isDebug: bool = False, useParallelMCTS: Optional[bool] = None, numParallel: int = 4, includeBaseline: bool = True, baselineElo: float = 500.0, includeOnix: bool = False) -> None:
    self.isDebug = isDebug
    
    self.generation = 0
    self.count = numberOfPlayers
    self.plyDepth = plyDepth
    self.mutationRate = 0.9
    self.players = {}
    self.champions = []
    self.playerCounter = 0
    self.folderDirectory = os.path.join("..", "results", "champions")
    self.elo_system = EloRating(k_factor=32, initial_rating=1200)
    self.baselineElo = baselineElo
    
    if useParallelMCTS is None:
      self.useParallelMCTS = plyDepth > 1
    else:
      self.useParallelMCTS = useParallelMCTS
    self.parallelThreads = numParallel
    
    self.currentPopulation = self.generatePlayers(self.count)
    
    self.baselineEntity = None
    if includeBaseline:
      self.baselineEntity = self.generateBaselinePlayer()
      if self.baselineEntity.id not in self.currentPopulation:
        self.currentPopulation.append(self.baselineEntity.id)

    # Onix: permanent heuristic-bot fixture at ~900 Elo
    self.onixEntity = None
    if includeOnix:
      self.onixEntity = self.generateOnixPlayer()
      if self.onixEntity.id not in self.currentPopulation:
        self.currentPopulation.append(self.onixEntity.id)

    self.numberOfWeights = self.players[0].bot.nn.lenCoefficents
    self.tau = 1 / math.sqrt( 2 * math.sqrt(self.numberOfWeights))

    # if safe mutations are enabled, we use it.
    self.safeMutations = True
    # debug flag
    self.debug = False
    # flag to handle crossover method; if 2 do heuristic method
    self.crossoverMethod = 2

  """
  Generates an individual player. This is only called in the
  generatePlayers() function!
  """
  def generatePlayer(self) -> Agent:
    # Slowbro handles [32,40,10,1] NN natively, with optional parallel TMCTS
    bot = sb.Slowbro(
        plyDepth=self.plyDepth,
        use_mlx=True,
        use_parallel=self.useParallelMCTS,
        num_parallel=self.parallelThreads,
        debug=self.isDebug
    )
    human = agent.Agent(bot, initial_elo=self.baselineElo)
    # generate ID
    human.setID(self.playerCounter)
    self.playerCounter += 1
    return human
  
  """
  Generates a baseline player with uninitialized (random) weights.
  This player has 1200 Elo and serves as a reference point in tournaments.
  """
  def generateBaselinePlayer(self) -> object:
    if self.baselineEntity is not None:
      return self.baselineEntity
    bot = sb.Slowbro(plyDepth=self.plyDepth, debug=self.isDebug, use_mlx=True)
    human = agent.Agent(bot, initial_elo=self.baselineElo)
    human.setID(-1)
    human.isBaseline = True
    human.entity_name = "baseline"
    self.players[human.id] = human
    return human

  def generateOnixPlayer(self) -> object:
    from agents.onix import Onix
    onix_bot = Onix(plyDepth=self.plyDepth, debug=self.isDebug)
    ent = agent.Agent(onix_bot, initial_elo=900.0)
    ent.setID(ONIX_ID)
    ent.entity_name = "Onix"
    ent.origin = [[0, 0, 0]]
    ent.parents = []
    self.players[ent.id] = ent
    return ent

  """
  Generates Players to participate in the tournament.
  This is only called at the beginning of the genetic algorithm.
  """
  def generatePlayers(self, count: int) -> list:
    players = []
    for _ in range(count):
      # generate a new human
      human = self.generatePlayer()
      # add it to the list of players
      self.players[human.id] = human
      # add it to the current population.
      players.append(human.id)
    return players

  """
  Self explanatory, prints the current population in order of
  Elo rating (now the primary ranking metric).
  """
  def printCurrentPopulationByPoints(self) -> str:
    if self.debug:
      print("Current Population:",self.currentPopulation)
    elo_ratings = list(map(lambda x: (x,self.players[x].elo, self.players[x].points), self.currentPopulation))
    elo_ratings = sorted(elo_ratings, key=operator.itemgetter(1), reverse=True)
    output = ""
    for i in elo_ratings:
      label = getattr(self.players[i[0]], 'entity_name', None)
      player_label = f"Player {i[0]} ({label})" if label else f"Player {i[0]}"
      output += f"{player_label}\tElo: {i[1]:.1f}\tPts: {i[2]}\n"
    return output

  def printCurrentPopulationByElo(self) -> str:
    if self.debug:
      print("Current Population:",self.currentPopulation)
    elo_ratings = list(map(lambda x: (x,self.players[x].elo, self.players[x].points), self.currentPopulation))
    elo_ratings = sorted(elo_ratings, key=operator.itemgetter(1), reverse=True)
    output = "Population by Elo Rating:\n"
    for i in elo_ratings:
      label = getattr(self.players[i[0]], 'entity_name', None)
      player_label = f"Player {i[0]} ({label})" if label else f"Player {i[0]}"
      output += f"{player_label}\tElo: {i[1]:.1f}\tPts: {i[2]}\n"
    return output

  """
  Prints the current population in order of Elo rating.
  """
  def printCurrentPopulationByElo(self) -> str:
    if self.debug:
      print("Current Population:",self.currentPopulation)
    elo_ratings = list(map(lambda x: (x,self.players[x].elo, self.players[x].points), self.currentPopulation))
    elo_ratings = sorted(elo_ratings, key=operator.itemgetter(1), reverse=True)
    output = "Population by Elo Rating:\n"
    for i in elo_ratings:
      player_label = f"Player {i[0]}"
      if self.baselineEntity and i[0] == self.baselineEntity.id:
        player_label += " (baseline)"
      output += f"{player_label}\tElo: {i[1]:.1f}\tPts: {i[2]}\n"
    return output

  def printEloStats(self) -> str:
    """Print Elo statistics for the current population."""
    elos = [self.players[pid].elo for pid in self.currentPopulation]
    avg_elo = sum(elos) / len(elos)
    best_elo = max(elos)
    worst_elo = min(elos)
    best_player = max(self.currentPopulation, key=lambda pid: self.players[pid].elo)
    
    output = f"Elo Stats - Avg: {avg_elo:.1f} | Best: {best_elo:.1f} (P{best_player}) | Worst: {worst_elo:.1f}\n"
    return output

  """
  order the players by how good they are.
  Now sorts by Elo rating instead of points.
  """
  def sortCurrentPopulationByPoints(self) -> list:
    # create tuple of players and their elo ratings
    elo_ratings = list(map(lambda x: (x,self.players[x].elo), self.currentPopulation))
    # sort list of tuples by Elo (highest first)
    elo_ratings = sorted(elo_ratings, key=operator.itemgetter(1), reverse=True)
    # assign back the sorted player IDs
    self.currentPopulation = [x[0] for x in elo_ratings]

  """
  Generate new population based on the player performance.
  Input: list of player ID's.
  Output: a new list of players.
  """
  def generateNextPopulation(self) -> None:
    start = datetime.datetime.now()
    self.generation += 1

    # Exclude Onix and baseline from parent/elite selection
    eligible = [pid for pid in self.currentPopulation
                if pid not in (ONIX_ID, -1)]
    elites = eligible[:5]

    for i in elites:
      self.players[i].points = 0
      self.players[i].games_played = 0

    offsprings = []
    for i in range(0, 2):
      parent_a_ID, parent_b_ID = eligible[i], eligible[i + 1]
      children = self.generatePlayers(4)

      # crossover from parents
      children[0], children[1] = self.crossOver(parent_a_ID, parent_b_ID, children[0], children[1])

      # assign evolution blocks to child 0 and child 1.
      self.inheritOrigins(children[0], [parent_a_ID,parent_b_ID])
      self.inheritOrigins(children[1], [parent_b_ID,parent_a_ID])
      # copy the caches of the parent to the child
      self.inheritCache(children[0],parent_a_ID)
      self.inheritCache(children[1],parent_b_ID)
      
      self.addOrigins(children[0], [1,1,0])
      self.addOrigins(children[1], [1,1,0])

      # for the last 2 offsprings, they obtain the same weights as their parents.
      self.setWeights(children[2],self.getWeights(parent_a_ID))
      self.setWeights(children[3],self.getWeights(parent_b_ID))
      self.addOrigins(children[2], [0,1,0])
      self.addOrigins(children[3], [0,1,0])
      self.inheritOrigins(children[2], [parent_a_ID])
      self.inheritOrigins(children[3], [parent_b_ID])
      # copy the caches of the parent to the child
      self.inheritCache(children[2],parent_a_ID)
      self.inheritCache(children[3],parent_b_ID)
      
      # mutate all offsprings
      # for offspring in children:
      #   self.mutate(offspring)
      # now we add children to the list of offsprings
      offsprings = offsprings + children

    # the last two children are mutations of 4th and 5th place bots.
    remainders = self.generatePlayers(2)
    self.setWeights(remainders[0], self.getWeights(eligible[3]))
    self.setWeights(remainders[1], self.getWeights(eligible[4]))
    self.addOrigins(remainders[0], [0,1,0])
    self.addOrigins(remainders[1], [0,1,0])
    self.inheritOrigins(remainders[0], [self.currentPopulation[3]])
    self.inheritOrigins(remainders[1], [self.currentPopulation[4]])
    # copy the caches of the parent to the child
    self.inheritCache(remainders[0], self.currentPopulation[3])
    self.inheritCache(remainders[1], self.currentPopulation[4])

    # add remainders to list of offsprings
    offsprings = offsprings + remainders
    if self.debug:
      print("offsprings:",offsprings)
   
    # mutate the offsprings. we should parallelise this.
    mutations = []
    print("Computing Mutations..")
    threadCount = multiprocessing.cpu_count()
    if len(offsprings) < threadCount:
      threadCount = len(offsprings)
    with multiprocessing.Pool(processes=threadCount) as pool:
      mutations = pool.map(self.mutate, offsprings)
      pool.close()
      pool.join()
    
    # for i in offsprings:
    #   mutations.append(self.mutate(i))
    print("Finished computing mutations.")
    
    # now that we have the mutations, load them to each agent.
    for mutation in mutations:
      self.players[mutation[0]].bot.nn.loadCoefficents(mutation[1])
    
    for i in range(0, 2):
      parent_a_ID, parent_b_ID = eligible[i], eligible[i + 1]
      parent_a_elo = self.players[parent_a_ID].elo
      parent_b_elo = self.players[parent_b_ID].elo
      mean_elo = (parent_a_elo + parent_b_elo) / 2
      # Children from crossover get mean of parents
      self.players[offsprings[i*4]].elo = mean_elo
      self.players[offsprings[i*4 + 1]].elo = mean_elo
      # Children from copy get parent's Elo
      self.players[offsprings[i*4 + 2]].elo = parent_a_elo
      self.players[offsprings[i*4 + 3]].elo = parent_b_elo
    
    # Set Elo for remainder offspring (copies of 4th and 5th place)
    self.players[offsprings[-2]].elo = self.players[eligible[3]].elo
    self.players[offsprings[-1]].elo = self.players[eligible[4]].elo
    
    # Reset games_played for all offspring (they start fresh)
    for offspring_id in offsprings:
      self.players[offspring_id].games_played = 0
      self.players[offspring_id].points = 0  # Also reset points for new generation
    
    newPopulation = offsprings + elites
    # Preserve Onix across generations (keep its Elo, never reset)
    if self.onixEntity is not None:
      newPopulation.append(self.onixEntity.id)
      self.players[ONIX_ID].points = 0
    # Preserve baseline entity across generations
    if self.baselineEntity is not None:
      newPopulation.append(self.baselineEntity.id)
      self.players[self.baselineEntity.id].elo = self.baselineElo
      self.players[self.baselineEntity.id].points = 0
    self.currentPopulation = newPopulation
    self.count = len(self.currentPopulation)
    end = datetime.datetime.now() - start
    if self.debug:
      print("DONE, that took", end)

    self.killCaches()

    print("Successfully computed offsprings for the next generation.")


  def heuristicCrossover(self, cpu1, cpu2, child1, child2) -> None:
    print("Processing Crossover")
    mother = self.players[cpu1].bot.nn.weights
    father = self.players[cpu2].bot.nn.weights

    randomlayer = random.randint(0,len(mother)-1)
    lenWeightsRandlayer = len(father[randomlayer])
    maxLim = int(0.4*lenWeightsRandlayer)
    randWeightIndexes = list(set([random.randint(0,lenWeightsRandlayer-1) for i in range(maxLim)]))

    newWeightSetA = []
    newWeightSetB = []

    for i in range(lenWeightsRandlayer):
      if i in randWeightIndexes:
        newWeightSetA.append(father[randomlayer][i].tolist()[0])
        newWeightSetB.append(mother[randomlayer][i].tolist()[0])
      else:
        newWeightSetA.append(mother[randomlayer][i].tolist()[0])
        newWeightSetB.append(father[randomlayer][i].tolist()[0])

    # turn back into matrix
    newWeightSetA = np.matrix(newWeightSetA)
    newWeightSetB = np.matrix(newWeightSetB)

    # for i in newWeightSetA:
    #   print(i)

    # now to load them to the offspring
    self.setWeights(child1, self.getWeights(cpu1))
    self.setWeights(child2, self.getWeights(cpu2))

    self.players[child2].bot.nn.weights[randomlayer] = newWeightSetA
    self.players[child2].bot.nn.weights[randomlayer] = newWeightSetB

    print("Crossover Successful.")
    # return the pair of children
    return (child1,child2)
    
  """
  Crossover mechanism for creating offspring children
  Input: two parents, two children, two indexes to swap from
  """
  def crossOver(self, cpu1, cpu2, child1, child2) -> None:
    """
    Basic Crossover Algorithm for the GA.
    """
    if self.debug:
      print("Implementing Crossover for IDs "+ str(child1) +"," +str(child2), end=".. ")
    mother = self.getWeights(cpu1)
    father = self.getWeights(cpu2)
    
    
    if self.crossoverMethod == 0:
      for _ in range(10):
        # generate a random index and swap genes
        index = random.randint(0, self.numberOfWeights)
        genome_m,genome_f = mother[index],father[index]
        mother[index] = genome_m
        father[index] = genome_f
      self.setWeights(child1, mother)
      self.setWeights(child2, father)
    elif self.crossOver == 2:
      self.heuristicCrossover(cpu1,cpu2,child1,child2)
    else:
      # generate random cutoff positions, 
      index1 = random.randint(0, self.numberOfWeights)
      index2 = random.randint(0, self.numberOfWeights)
      # check the order of the indexes to make sure they make sense.
      if index1 > index2: 
        index1, index2 = index2, index1
      # pythonic crossover
      child1W = np.append(np.append(father[:index1], mother[index1:index2]), father[index2:])
      child2W = np.append(np.append(mother[:index1], father[index1:index2]), mother[index2:])
      
      # create new children with it
      self.setWeights(child1, child1W)
      self.setWeights(child2, child2W)
    
    print("Crossover Successful.")
    # return the pair of children
    return (child1,child2)  



  """
  Mutate the weights of the neural network.
  """
  def mutate(self, cpu) -> None:
    """
    Mutate the weights of the neural network.
    """
    if self.debug:
      print("Generating mutations for player " + str(cpu))

    weights = self.getWeights(cpu)
    # weights = self.players[cpu].bot.nn.weights
    # check whether their moves are cached.
    moveBase = self.getMoveCache(cpu)

    if self.safeMutations and len(moveBase) > 100:
      weights = self.safeMutation(cpu)
      # nuke the cache
      self.players[cpu].bot.cache = {}
    else:
      # random mutation multipliers
      multipliers = np.random.random_sample([self.numberOfWeights])
      multipliers = self.tau * multipliers
      multipliers = np.exp(multipliers)
      weights=weights*multipliers
      weights=np.clip(weights, -1, 1)
      self.setWeights(cpu, weights)
    if self.debug:
      print("Finished mutations for player " + str(cpu))
    return (cpu, weights)

  """
  Static function to create safe mutations
  """
  def safeMutation(self, cpu, static: bool = False) -> None:
    print("Computing Safe Mutations..")
    cache = self.getMoveCache(cpu)
    curreneWeight1D = self.players[cpu].bot.nn.getAllCoefficents()

    # get a subset of those cached moves.
    subset = {}
    subsetSize = int(len(cache.keys())/10)
    if subsetSize < 1000:
      subsetSize = len(cache.keys())
    for _ in range(subsetSize):
      rand = random.choice(list(cache.keys()))
      subset[rand] = np.array(rand)

    # now we find the safest mutation
    bestWeight = curreneWeight1D
    bestScore = 0

    for su in range(100):
      weights = None
      if static:
        # create a new mutation
        multipliers = np.random.random_sample([self.numberOfWeights])
        multipliers = self.tau * multipliers
        multipliers = np.exp(multipliers)
        weights = multipliers * curreneWeight1D
        weights = np.clip(weights, -1, 1)
        self.players[cpu].bot.nn.loadCoefficents(weights)
      else:
        for w in range(len(self.players[cpu].bot.nn.weights)):
          weights = self.players[cpu].bot.nn.weights[w]
          multipliers = np.random.random_sample(weights.shape)
          multipliers = self.tau * multipliers
          multipliers = np.exp(multipliers)
          # print(weights.shape, multipliers.shape)
          self.players[cpu].bot.nn.weights[w] = np.add(weights,multipliers)
          self.players[cpu].bot.nn.weights[w] = np.clip(self.players[cpu].bot.nn.weights[w],-1,1)

          biases = self.players[cpu].bot.nn.biases[w]
          multipliers = np.random.random_sample(biases.shape)
          multipliers = self.tau * multipliers
          multipliers = np.exp(multipliers)
          # print(weights.shape, multipliers.shape)
          self.players[cpu].bot.nn.biases[w] = np.add(biases,multipliers)
          self.players[cpu].bot.nn.biases[w] = np.clip(self.players[cpu].bot.nn.biases[w],-1,1)
        weights = self.players[cpu].bot.nn.getAllCoefficents()

      # calculate to see whether these weights are better
      qa = 0
      for i in subset:
        # evaluate this cached move
        eval_a = self.players[cpu].bot.nn.compute(subset[i])
        # compare it to the current value
        if eval_a/cache[i] >= 0.95:
          qa += 1
      if qa > bestScore:
        bestWeight = weights
        bestScore = qa
        # print("New best:", qa, su)
      percentile = round(bestScore*100/len(subset),2)
      print(str(cpu)+" - Count:"+str(su)+" Best:"+str(bestScore)+"/"+str(len(subset))+ " - " +str(percentile)+"%\r",end="")
    # print("\nDONE")
    return bestWeight


  """
  Saves champions to a file.
  """
  def saveChampionsToFile(self, folderDirectory: str) -> None:
    folderDirectory = os.path.join(folderDirectory, "champions")
      # check save directory exists prior to saving
    if not os.path.isdir(folderDirectory):
      os.makedirs(folderDirectory)
        
    championJson = {}
    i = self.generation
    championJson[i] = {}

    championID = self.champions[-1]
    # store player and its weights.
    championJson[i]['pid'] = self.players[championID].id
    championJson[i]['coefficents'] = self.players[championID].bot.nn.getAllCoefficents().tolist()
    championJson[i]['champRange'] = self.players[championID].champRange
    championJson[i]['champScore'] = self.players[championID].champScore
  
    filename = str(i) + ".json"
    with open(os.path.join(folderDirectory, filename), 'w') as outfile:
      json.dump(championJson, outfile)
    
      # append to file.
    print("saved champs to ",filename)

  """
  Saves genomic properties to a file. Each champion's properties
  gets saved in this file, such as whether they were made from
  mutation, their parents IDs, whether crossovers were used 
  and so on.
  """
  def savePopulationGenomes(self, folderDirectory: str) -> None:
    # check save directory exists prior to saving
    if not os.path.isdir(folderDirectory):
      os.makedirs(folderDirectory)

    agent = {}
    for player_id in self.currentPopulation:
      # store player and its weights.
      agent[player_id] = {}
      agent[player_id]['score'] = self.players[player_id].points
      agent[player_id]['origin'] = self.players[player_id].origin
      agent[player_id]['parents'] = self.players[player_id].parents
  
    filename = "genomes.json"
    with open(os.path.join(folderDirectory, filename), 'w') as outfile:
      json.dump(agent, outfile)
    
      # append to file.
    print("saved players genomic info.")     

  """
  NOT USED
  """
  def savePopulationToDB(self, db) -> None:
    population = self.currentPopulation
    """
    Stores the population into Mongo. 
    """
    keys = []
    if db.connected:
      for i in population:
        if db.checkPlayerExists(i.id) == False:
          entry = db.write('players', i.getDict())
          keys.append(entry)
        else:
          keys.append(i.id)
    return keys

  """
  Allocates points to players based on the game outcomes
  Also updates Elo ratings for both players.
  """
  def allocatePoints(self, results, black, white) -> None:
    black_rating = self.players[black].elo
    white_rating = self.players[white].elo
    black_games = self.players[black].games_played
    white_games = self.players[white].games_played

    self.players[black].games_played += 1
    self.players[white].games_played += 1

    # Determine the result scores for Elo calculation
    if results["Winner"] == Black:
      black_score, white_score = 1.0, 0.0
      self.players[black].points += WinPt
      self.players[white].points += LosePt
    elif results["Winner"] == White:
      black_score, white_score = 0.0, 1.0
      self.players[black].points += LosePt
      self.players[white].points += WinPt
    else:
      black_score, white_score = 0.5, 0.5

    # Onix is a fixed anchor at 900, never moves
    if getattr(self.players[black], 'entity_name', None) != 'Onix':
      self.players[black].elo = self.elo_system.update_rating(black_rating, white_rating, black_score, black_games)
    if getattr(self.players[white], 'entity_name', None) != 'Onix':
      self.players[white].elo = self.elo_system.update_rating(white_rating, black_rating, white_score, white_games)

  def addChampion(self) -> None:
    for pid in self.currentPopulation:
      if pid not in (ONIX_ID, -1):
        self.champions.append(pid)
        return

  """
  Assign weights to a bot's neural net.
  """
  def setWeights(self, botID, weights) -> None:
    self.players[botID].bot.nn.loadCoefficents(weights)

  # Done
  def getWeights(self, botID) -> object:
    return self.players[botID].bot.nn.getAllCoefficents()

  """
  Helper function to retrieve cache if it exists,
  otherwise return empty dict.
  """
  def getMoveCache(self, botID) -> dict:
    if self.players[botID].bot.enableCache:
      return self.players[botID].bot.cache
    else:
      return {}  # Return empty dict instead of False

  """
  Kill the caches when we're done with mutations or whatever.
  This is really important!
  """
  def killCaches(self) -> None:
    for i in range(self.playerCounter):
      self.players[i].bot.cache = {}
    print("Killed all caches.")

  # Done
  def addOrigins(self, botID, values) -> None:
    self.players[botID].origin = [values]

  # Done
  def inheritOrigins(self, botID, parentIDs) -> None:
    for i in parentIDs:
       self.players[botID].parents.append(i)

  def inheritCache(self, botID, parentID) -> None:
    self.players[botID].bot.cache = self.players[parentID].bot.cache

  @staticmethod
  def generateRandomWeights(weights=None):
    multipliers = np.random.random_sample([self.numberOfWeights])
    multipliers = self.tau * multipliers
    multipliers = np.exp(multipliers)
    if np.any(weights):
      weights=weights*multipliers
      weights=np.clip(weights, -1, 1)
      return weights
    else:
      return np.clip(multipliers,-1, 1)

  @staticmethod
  def generateFakeMoves():
    print("Generating fake moves", end=".. ")
    # we create a dictionary of fake moves. 
    nn = NeuralNetwork(layer_list=[32,40,10,1])
    fakeMoves = {}
    for _ in range(10000):
      # generate a random list of nn inputs
      state = np.random.random_sample([32])
      stateID = tuple(state)
      evals = nn.compute(state)
      fakeMoves[stateID] = evals
      # evaluate it
    # get a copy of the current nn coefs.
    coefs = nn.getAllCoefficents()
    # return this.
    print("Generated Fake moves.")
    return (fakeMoves, coefs)
    

if __name__ == '__main__':
  # we can use this to test out the population program
  pass
  # x = Population(15,1)