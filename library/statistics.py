import json
import os

class Statistics:
  def __init__(self, date, defaultResultsPath=None):
    self.date = date
    self.path = os.path.join("..", "results")
    if defaultResultsPath:
      self.path = defaultResultsPath
    self.statistics = {}
    leaderboards = []
  
  def loadStatisticsFile(self, filename="statistics.json"):
    directory = os.path.join(self.path, date)
    filepath = os.path.join(directory, filename)
    print("Loading Statistics from file:")
    print("\t", filepath)
    f = open(filepath, 'r')
    self.statistics = json.load(f)
    f.close()
    print("Loaded Stats File!")

  def parseLeaderboards(self):
    for gen in self.statistics:
      pass

  """
  Gets tournament timing information
  """
  def timeStatsPerGeneration(self):
    x = []
    for i in range(len(self.statistics)):
      times = {
        'gameDurations' : []
      }
      # get generation tournament length
      times["duration"] = self.statistics[i]['durationInSeconds']
      # get individual game lengths
      for j in self.statistics[i]['games']:
        times['gameDurations'].append(j['duration'])
      # sort the duration times
      times['gameDurations'] = sorted(times['gameDurations'])
      x.append(times)
    return x

  """
  Calculates the average number of moves per generation
  """
  def averageNumMovesPerGeneration(self):
    print("Calculating Means..")
    x = []
    for i in range(len(self.statistics)):
      moveCounts = []
      for j in self.statistics[i]['games']:
        moveCounts.append(len(j['Moves']))
    mean = sum(x) / float(len(x))
    print("Mean:", mean)
    x.append(mean)
    return x

if __name__ == '__main__':
  date = "2018-03-19 16:42:47"
  s = Statistics(date)
  s.loadStatisticsFile()
  print(s.averageNumMovesPerGeneration)