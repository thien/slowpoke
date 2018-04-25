import agents.slowpoke as sp
import agents.agent as agent
import core.mongo as mongo
from agents.evaluator.neural import NeuralNetwork

import math
import random
import numpy as np
import json
import operator
import os
import datetime
import multiprocessing

Black, White, empty = 0, 1, -1

WinPt, DrawPt, LosePt = 2, 0, -1

class Population:
  def __init__(self, numberOfPlayers, plyDepth, isDebug=False):
    # used for testing purposes.
    self.isDebug = isDebug
    
    self.generation = 0
    self.count = numberOfPlayers
    self.plyDepth = plyDepth  # plydepth
    self.mutationRate = 0.9   # rate for player mutation
    self.players = {}       # this is a list of players (all players)
    self.champions = []     # here we list the champions
    self.playerCounter = 0    # used to create playerID's.
    # generate an initial population
    self.currentPopulation = self.generatePlayers(self.count)

    self.numberOfWeights = self.players[0].bot.nn.lenCoefficents
    self.tau = 1 / math.sqrt( 2 * math.sqrt(self.numberOfWeights))

    # if safe mutations are enabled, we use it.
    self.safeMutations = True
    # create reference neural network
    self.nn = NeuralNetwork(self.players[0].bot.layers)
    # debug flag
    self.debug = False
    # flag to handle crossover method; if 2 do heuristic method
    self.crossoverMethod = 2

  """
  Generates an individual player. This is only called in the
  generatePlayers() function!
  """
  def generatePlayer(self):
    bot = sp.Slowpoke(self.plyDepth,debug=self.isDebug)
    human = agent.Agent(bot)
    # generate ID
    human.setID(self.playerCounter)
    self.playerCounter += 1
    return human

  """
  Generates Players to participate in the tournament.
  This is only called at the beginning of the genetic algorithm.
  """
  def generatePlayers(self, count):
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
  how good they are (in terms of points)
  """
  def printCurrentPopulationByPoints(self):
    if self.debug:
      print("Current Population:",self.currentPopulation)
    points = list(map(lambda x: (x,self.players[x].points), self.currentPopulation))
    # sort list of tuples
    output = ""
    for i in points:
      # i[0] is the player ID, i[1] is the player's score.
      output += "Player "+str(i[0])+ "\t" + str(i[1])+"\n"
    return output

  """
  order the players by how good they are.
  """
  def sortCurrentPopulationByPoints(self):
    # create tuple of players and their points
    points = list(map(lambda x: (x,self.players[x].points), self.currentPopulation))
    # sort list of tuples
    points = sorted(points, key=operator.itemgetter(1), reverse=True)
    # assign back the first half of the tuples to the list of players.
    self.currentPopulation = [x[0] for x in points]

  """
  Generate new population based on the player performance.
  Input: list of player ID's.
  Output: a new list of players.
  """
  def generateNextPopulation(self):
    start = datetime.datetime.now()
    # increment generation count
    self.generation += 1
    # start with the top 5 players from the previous generation
    elites = self.currentPopulation[:5]
    # reset scores for these elites
    for i in elites: self.players[i].points = 0
    # create list of offsprings
    offsprings = []
    # create magic crossover from new parents
    for i in range(0,2):
      # # get ID's of parents
      parent_a_ID, parent_b_ID = self.currentPopulation[i], self.currentPopulation[i+1]
      # here we create 4 new children
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
    self.setWeights(remainders[0], self.getWeights(self.currentPopulation[3]))
    self.setWeights(remainders[1], self.getWeights(self.currentPopulation[4]))
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

    newPopulation = offsprings + elites
    # assign this set of offsprings as the new population.
    self.currentPopulation = newPopulation
    self.count = len(self.currentPopulation)
    end = datetime.datetime.now() - start
    if self.debug:
      print("DONE, that took", end)

    self.killCaches()

    print("Successfully computed offsprings for the next generation.")


  def heuristicCrossover(self,cpu1,cpu2,child1,child2):
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
  def crossOver(self, cpu1, cpu2, child1, child2):
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
  def mutate(self,cpu):
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
  def safeMutation(self, cpu, static=False):
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
  def saveChampionsToFile(self, folderDirectory):
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
  def savePopulationGenomes(self, folderDirectory):
    # check save directory exists prior to saving
    if not os.path.isdir(folderDirectory):
      os.makedirs(folderDirectory)

    agent = {}
    for i in range(len(self.players)):
      # store player and its weights.
      agent[i] = {}
      agent[i]['score'] = self.players[i].points
      agent[i]['origin'] = self.players[i].origin
      agent[i]['parents'] = self.players[i].parents
  
    filename = "genomes.json"
    with open(os.path.join(folderDirectory, filename), 'w') as outfile:
      json.dump(agent, outfile)
    
      # append to file.
    print("saved players genomic info.")     

  """
  NOT USED
  """
  def savePopulationToDB(self, db):
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
  """
  def allocatePoints(self, results, black, white):
    if results["Winner"] == Black:
      self.players[black].points += WinPt
      self.players[white].points += LosePt
    elif results["Winner"] == White:
      self.players[black].points += LosePt
      self.players[white].points += WinPt

  """
  Adds the champion to the list of champions
  """
  def addChampion(self):
    self.champions.append(self.currentPopulation[0])

  """
  Assign weights to a bot's neural net.
  """
  def setWeights(self,botID, weights):
    self.players[botID].bot.nn.loadCoefficents(weights)

  # Done
  def getWeights(self,botID):
    return self.players[botID].bot.nn.getAllCoefficents()

  """
  Helper function to retrieve cache if it exists,
  otherwise return false.
  """
  def getMoveCache(self,botID):
    if self.players[botID].bot.enableCache:
      return self.players[botID].bot.cache
    else:
      return False

  """
  Kill the caches when we're done with mutations or whatever.
  This is really important!
  """
  def killCaches(self):
    for i in range(self.playerCounter):
      self.players[i].bot.cache = {}
    print("Killed all caches.")

  # Done
  def addOrigins(self,botID, values):
    self.players[botID].origin = [values]

  # Done
  def inheritOrigins(self,botID, parentIDs):
    for i in parentIDs:
       self.players[botID].parents.append(i)

  def inheritCache(self,botID,parentID):
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
    nn = NeuralNetwork(layer_list=[91,40,10,1])
    fakeMoves = {}
    for _ in range(10000):
      # generate a random list of nn inputs
      state = np.random.random_sample([91])
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