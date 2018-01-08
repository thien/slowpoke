import slowpoke as sp
import agent
import math
import random
import numpy as np
import mongo
import json
import operator
import os

Black, White, empty = 0, 1, -1

WinPt, DrawPt, LosePt = 2, 0, -1

class Population:
  def __init__(self, numberOfPlayers, plyDepth):
    self.generation = 0
    self.count = numberOfPlayers
    self.plyDepth = plyDepth  # plydepth
    self.mutationRate = 0.9   # rate for player mutation
    self.players = {}       # this is a list of players (all players)
    self.champions = []     # here we list the champions
    self.playerCounter = 0    # used to create playerID's.
    self.ting = "234234"
    # generate an initial population
    self.currentPopulation = self.generatePlayers(self.count)

    self.numberOfWeights = self.players[0].bot.nn.lenCoefficents
    self.tau = 1 / math.sqrt( 2 * math.sqrt(self.numberOfWeights))

    self.folderDirectory = os.getcwd() + "/champions/"
    # check save directory exists prior to saving
    if not os.path.isdir(self.folderDirectory):
      os.makedirs(self.folderDirectory)

  # Done
  def generatePlayer(self):
    bot = sp.Slowpoke(self.plyDepth)
    human = agent.Agent(bot)
    # generate ID
    human.setID(self.playerCounter)
    self.playerCounter += 1
    return human

  # Done
  def generatePlayers(self, count):
    """
    Generates Players to participate in the tournament.
    This is only called at the beginning of the genetic algorithm.
    """
    players = []
    for i in range(count):
      # generate a new human
      human = self.generatePlayer()
      # add it to the list of players
      self.players[human.id] = human
      # add it to the current population.
      players.append(human.id)
    return players

  # Done
  def printCurrentPopulationByPoints(self):
    print("Current Population:",self.currentPopulation)
    points = list(map(lambda x: (x,self.players[x].points), self.currentPopulation))
    # sort list of tuples
    output = ""
    for i in points:
      player_id = i[0]
      output += "Player "+str(i[0])+ "\t" + str(i[1])+"\n"
    return output

  # Done
  def sortCurrentPopulationByPoints(self):
    """
    order the players by how good they are.
    """
    # create tuple of players and their points
    points = list(map(lambda x: (x,self.players[x].points), self.currentPopulation))
    # sort list of tuples
    points = sorted(points, key=operator.itemgetter(1), reverse=True)
    # assign back the first half of the tuples to the list of players.
    self.currentPopulation = [x[0] for x in points]

  # Done
  def generateNextPopulation(self):
    """
    Generate new population based on the player performance.
    Input: list of player ID's.
    Output: a new list of players.
    """
    # increment generation count
    self.generation += 1
    # start with the top 5 players from the previous generation
    offsprings = self.currentPopulation[:5]
    # reset scores for these current offsprings
    for i in offsprings:
      self.players[i].points = 0
    # create magic crossover from new parents
    for i in range(0,2):
      # # get ID's of parents
      parent_a_ID, parent_b_ID = self.currentPopulation[i], self.currentPopulation[i+1]
      # # here we create 4 new children
      children = self.generatePlayers(4)
      # generate random cutoff positions
      index1 = random.randint(0, self.numberOfWeights)
      index2 = random.randint(0, self.numberOfWeights)
      # check the order of the indexes to make sure they make sense.
      if index1 > index2: 
        index1, index2 = index2, index1
      # crossover from parents
      children[0], children[1] = self.crossOver(parent_a_ID, parent_b_ID, children[0], children[1], index1, index2)

      # assign evolution blocks to child 0 and child 1.
      self.inheritOrigins(children[0], [parent_a_ID,parent_b_ID])
      self.inheritOrigins(children[1], [parent_b_ID,parent_a_ID])
      self.addOrigins(children[0], [1,1,0])
      self.addOrigins(children[1], [1,1,0])

      # for the last 2 offsprings, they obtain the same weights as their parents.
      self.setWeights(children[2],self.getWeights(i))
      self.setWeights(children[3],self.getWeights(i+1))
      self.addOrigins(children[2], [0,1,0])
      self.addOrigins(children[3], [0,1,0])
      self.inheritOrigins(children[2], [self.currentPopulation[i]])
      self.inheritOrigins(children[3], [self.currentPopulation[i+1]])

      # mutate all offsprings
      for offspring in children:
        self.mutate(offspring)
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

    for offspring in remainders:
      self.mutate(offspring)
    # add remainders to list of offsprings
    offsprings = offsprings + remainders
    # assign this set of offsprings as the new population.
    self.currentPopulation = offsprings
    self.count = len(offsprings)
  
  # Done
  def crossOver(self, cpu1, cpu2, child1, child2, index1, index2):
    """
    Basic Crossover Algorithm for the GA.
    """
    mother = self.getWeights(cpu1)
    father = self.getWeights(cpu2)

    # pythonic crossover
    child1W = np.append(np.append(father[:index1], mother[index1:index2]), father[index2:])
    child2W = np.append(np.append(mother[:index1], father[index1:index2]), mother[index2:])
    
    # create new children with it
    self.setWeights(child1, child1W)
    self.setWeights(child2, child2W)

    # return the pair of children
    return (child1,child2)  

  # Done
  def mutate(self,cpu):
    """
    Mutate the weights of the neural network.
    """
    # # generate a random number
    # chance = random.random()
    # if chance >= (1-self.mutationRate):
    #   # mutate the weights.
    #   weights = self.getWeights(cpu)
    #   length = weights.size
    #   base1 = weights[random.randint(0,length-1)]
    #   base2 = weights[random.randint(0,length-1)]
    #   temp = base1
    #   np.place(weights, weights==base1, temp)
    #   np.place(weights, weights==base2, base1)
    #   np.place(weights, weights==temp, base2)
    #   self.setWeights(cpu, weights)
    weights = self.getWeights(cpu)
    # random mutation multipliers
    multipliers = np.random.random_sample([self.numberOfWeights])
    multipliers = self.tau * multipliers
    multipliers = np.exp(multipliers)
    weights=weights*multipliers
    weights=np.clip(weights, -1, 1)
    self.setWeights(cpu, weights)

  # Done
  def storeCurrentChampionStats(self,stats):
    self.generation
    championJson[i]

  # Done
  def saveChampionsToFile(self, datepath="today"):
    folderDirectory = os.getcwd() + "/champions/" + datepath + "/"
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
  
    with open(folderDirectory + datepath + " - " + str(i) + ".json", 'w') as outfile:
      json.dump(championJson, outfile)
    
      # append to file.
    print("saved champs to ",( folderDirectory+ str(i) + ".json"))

  def savePopulationGenomes(self, datepath):
    folderDirectory = os.getcwd() + "/champions/" + datepath + "/"
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
  
    with open(folderDirectory + datepath + " - genomes.json", 'w') as outfile:
      json.dump(agent, outfile)
    
      # append to file.
    print("saved players genomic info.")     

  def savePopulation(self):
    """
    Saves population to file, and also to database if needed.
    """
    return False

  def savePopulationToDB(self):
    population = self.currentPopulation
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

  # Done
  def allocatePoints(self, results, black, white):
    # allocates points to players dependent on the game results.
    if results["Winner"] == Black:
      self.players[black].points += WinPt
      self.players[white].points += LosePt
    elif results["Winner"] == White:
      self.players[black].points += LosePt
      self.players[white].points += WinPt

  # Done
  def addChampion(self):
    self.champions.append(self.currentPopulation[0])

  # Done
  def setWeights(self,botID, weights):
    self.players[botID].bot.nn.loadCoefficents(weights)

  # Done
  def getWeights(self,botID):
    return self.players[botID].bot.nn.getAllCoefficents()

  # Done
  def addOrigins(self,botID, values):
    self.players[botID].origin = [values]

  # Done
  def inheritOrigins(self,botID, parentIDs):
    for i in parentIDs:
       self.players[botID].parents.append(i)

  # Done
  @staticmethod
  def randomPlayerID(val):
    # choose a random number between 1 and the number of players. checks that it isn't the same number as val.
    rand = randint(0, self.count-1)
    while (rand == val):
      rand = randint(0, self.count-1)
    return rand


  # def roulette(self, players):
  #   """
  #   Roulette algorithm for parent selection.
  #   """
  #   chosen = None
  #   overallFitness = 0
  #   superEqual = False
  #   # calculate the overall fitness
  #   for i in players:
  #     overallFitness += i.points
  #   # check if overall fitness is zer0
  #   if overallFitness == 0:
  #     overallFitness = len(players)
  #     superEqual = True
  #   # randomly shuffle the players around.
  #   random.shuffle(players)
  #   # initiate a list of probablities
  #   probabilities = []
  #   # calculate probabllities for each player.
  #   for i in players:
  #     if not superEqual:
  #       probability = i.points / overallFitness
  #     else:
  #       probability = 1 / overallFitness
  #     if len(probabilities) > 1:
  #       probability = probability + probabilities[-1]
  #     probabilities.append(probability)
  #   # generate a random number
  #   number = random.random()
  #   # find a player.
  #   for i in range(len(probabilities)):
  #     if number < probabilities[i]:
  #       # we have chosen a player.
  #       chosen = players[i]
  #       break
  #   return chosen
