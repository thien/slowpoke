import os
import termcolor
# import ijson
import json
import re


class agentLoader:
  def __init__(self):
    self.basepath = os.path.join("..", "results")
    self.championFiletype = ".json"
    self.championsFoldername = "champions"
    self.statisticsFilename = "statistics.json"
    self.systems = []
    self.cacheFilename = "menuCache.json"
    self.checkDirectoryChange()

  def checkDirectoryChange(self):
    cached = False
    # check if our cache file is there
    if self.cacheFilename in os.listdir(self.basepath):
      # load the file 
      cache = self.loadCache()
      # count number of items
      if cache is not False:
        if "count" in cache.keys():
          if self.countDirectoryitems() == cache["count"]:
            if "systems" in cache.keys():
              self.systems = cache['systems']
              cached = True
    if not cached:
      self.rebuildCacheList()
      # when done, save to cache.
      self.saveCache()

  def loadCache(self):
    filepath = os.path.join(self.basepath, self.cacheFilename)
    cache = {}
    try:
      f = open(filepath, 'r')
      cache = json.load(f)
      f.close()
      return cache
    except:
      return False

  def saveCache(self):
    filepath = os.path.join(self.basepath, self.cacheFilename)
    cache = {
      "count" : self.countDirectoryitems(),
      "systems" : self.systems
    }
    if os.path.isfile(filepath):
      os.remove(filepath)
    with open(filepath, 'w') as outfile:
      json.dump(cache, outfile)
    return True

  def countDirectoryitems(self):
    count = 0
    for system in os.listdir(self.basepath):
      sysdir = os.path.join(self.basepath, system)
      if os.path.isdir(sysdir):
        if self.championsFoldername in os.listdir(sysdir):
          champPath = os.path.join(sysdir, self.championsFoldername)
          for entry in os.listdir(champPath):
            count += 1
    return count

  def loadCoefficents(self,filepath):
    return json.load(filepath)

  def detectAgentFiles(self,path):
    if os.path.isdir(path):
      contents = os.listdir(path)
      if self.statisticsFilename in contents:
        if self.championsFoldername in contents:
          p = os.listdir(os.path.join(path, self.championsFoldername))
          if len(p) > 0:
            return True
    return False

  def getLatestAgent(self,agentPath):
    """
    returns path to the latest agent.
    """
    champPath = os.path.join(agentPath, "champions")
    directory = os.listdir(champPath)
    directory = sorted([int(i.replace(self.championFiletype,"")) for i in directory])
    latestAgent = str(directory[-1]) + self.championFiletype
    return os.path.join(champPath, latestAgent)

  def getNumberOfChampions(self,path):
    return len(os.listdir(os.path.join(path, self.championsFoldername)))

  def scrapeStats(self, directory):
    statFilepath = os.path.join(directory, self.statisticsFilename)
    statistics = None

    try:
      f = open(statFilepath, 'r')
      statistics = json.load(f)
      f.close()
    except:
      return False

    plyDepth = str(statistics[0]['stats'][2][1])
    
    # finds the best agent.
    bestScore = 0
    bestGeneration = 0
    endScore = 0
    for j in range(len(statistics)):
      k = statistics[j]['stats']
      generation = 0
      for i in k:
        if i[0] == "Generation":
          generation = str(i[1].replace("/200",""))
        if i[0] == "Cummulative Score":
          if float(i[1]) > bestScore:
            bestGeneration = generation
            bestScore = round(float(i[1]),2)
          else:
            endScore = round(float(i[1]),2)

    stats = {
      "bestScore" : bestScore,
      "bestGeneration" : bestGeneration,
      "plyDepth" : plyDepth,
      "OldestGen": self.getNumberOfChampions(directory),
      "endScore" : endScore
    }
    return stats

  def rebuildCacheList(self):
    # lets find all the items in the directory
    counter = 0
    print()
    for i in os.listdir(self.basepath):
      counter += 1
      print("Loading Files in Directory: "+str(counter)+"/"+str(len(self.basepath))+"\r", end="")
      agentPath = os.path.join(self.basepath, i)
      containsAgent = self.detectAgentFiles(agentPath)
      if containsAgent:
        stats = self.scrapeStats(agentPath)
        stats["Name"] = i
        stats["latestAgentFile"] = self.getLatestAgent(agentPath)
        # only keep systems with long enough tings
        if stats['OldestGen'] >1:
          self.systems.append(stats)
        stats['baseDir'] = os.path.join(self.basepath, stats['Name'])
        stats['ChampDir'] = os.path.join(stats['baseDir'], self.championsFoldername)
    print("",end="")
    # sort files by score
    self.systems = sorted(self.systems, key=lambda x: x["endScore"])[::-1]


  def loadAgentUI(self):
    print("Select the system to load, by the index:")
    chosen = False
    while not chosen:
      for j in range(len(self.systems)):
        i = self.systems[j]
        print("Index:"+str(j), "\t\t", end="")
        print("Ply:"+ i['plyDepth']+"\t", end="")
        print("#Champs:",str(i['OldestGen'])+"\t", end="")
        print("Hi-Score:", str(i['bestScore'])+"\t", end="")
        print("End-Score:", str(i['endScore'])+"\t", end="")
        print("Foldername:",i['Name'])
      index = input("Select the Index: ")
      if int(index) not in [i for i in range(len(self.systems))]:
        print("This is an invalid input, please try again.")
      else:
        chosen = True
        system = self.systems[int(index)]
        return system

  @staticmethod
  def loadSpecificAgent(system):
    chosenAnswer = False
    option = False
    agentID = system['OldestGen']-1
    while not chosenAnswer:
      print("Would you like to choose a specific Agent?")
      print("By Default, it would choose the latest agent.")
      answer = input("Y/N: ")
      if answer in "ynYN":
        chosenAnswer = True
        if answer in "yY":
          option = True
      else:
        print("You chose an invalid option, please try again.")
        print("---")

    if option:
      chosenAgent = False
      while not chosenAgent:
        print("Please choose a generation from 1 to "+str(system['OldestGen']+1))
        genID = input("Generation: ")
        try:
          genID = int(genID)
          if (genID < agentID+1) and (genID > 0):
            agentID = genID
            chosenAgent = True
        except:
          print("That's an invalid response. Please try again..")
    return agentID

  def findTheBest(self, ply=1):
    bestScore = 0
    bestSystem = None
    for i in self.systems:
      # print(i)
      if int(i["plyDepth"]) == ply:
        # now we iterate through the best score.
        if i['bestScore'] >= bestScore:
          bestSystem = i
          bestScore = i['bestScore']
    return bestSystem
  
  def loadStatisticsFile(self,system):
    filepath = os.path.join(system['baseDir'], self.statisticsFilename)
    stats = {}
    try:
      f = open(filepath, 'r')
      stats = json.load(f)
      f.close()
      return stats
    except:
      return False

  def loadAgentWeights(self,system, pid):
    champfile = str(pid) + ".json"
    champpath = os.path.join(system['ChampDir'], champfile)
    agent = {}
    try:
      f = open(champpath, 'r')
      agent = json.load(f)
      f.close()
      return agent[str(pid)]['coefficents']
    except:
      print("I can't load the file for some reason.")
      return False

if __name__ == "__main__":
  al = agentLoader()
  # al.rebuildCacheList()
  # system = al.loadAgentUI()
  # theid = al.loadSpecificAgent(system)
  # print(system)
  # print("chosen", theid)
