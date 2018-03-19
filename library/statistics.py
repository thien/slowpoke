import json
import os
import numpy as np
import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import ast

class Statistics:
  def __init__(self, date, defaultResultsPath=None):
    self.date = date
    self.path = os.path.join("..", "results")
    if defaultResultsPath:
      self.path = defaultResultsPath
    self.statistics = {}
    self.directory = os.path.join(self.path, self.date)
    leaderboards = []
  
  def loadStatisticsFile(self, filename="statistics.json"):
    filepath = os.path.join(self.directory, filename)
    print("Loading Statistics from file:")
    print("\t", filepath)
    f = open(filepath, 'r')
    self.statistics = json.load(f)
    f.close()
    print("Loaded Stats File!")

  def parseLeaderboards(self):
    for gen in self.statistics:
      pass

  def saveChartToFile(self, plt, title):
    savefig('foo.png', bbox_inches='tight')

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

    # generate a graph for this
    return x

  """
  Calculates the average number of moves per generation.
  """
  def averageNumMovesPerGeneration(self):
    print("Calculating Move Counts (Per Generation)")
    means = []
    medians = []
    graphData = {
      "x" : [],
      "y" : []
    }
    counter = 0
    for i in range(len(self.statistics)):
      moveCounts = []
      for j in self.statistics[i]['games']:
        graphData['x'].append(i)
        graphData['y'].append(len(j['game']['Moves']))
        moveCounts.append(len(j['game']['Moves']))
        counter += 1
      moveCounts = sorted(moveCounts)
      mean = sum(moveCounts) / float(len(moveCounts))
      # print(i, "Mean:", round(mean,2), "S:", moveCounts[0], "L:", moveCounts[-1])
      means.append(mean)
      medians.append(max(set(moveCounts), key=moveCounts.count))

    mean_colour = "red"
    median_colour = "yellow"
    # add legend
    mean_patch = mpatches.Patch(color=mean_colour, label='Mean')
    median_patch = mpatches.Patch(color=median_colour, label='Median')
    plt.legend(handles=[mean_patch,median_patch])

    # plot 2d histogram
    plt.hist2d(graphData['x'], graphData['y'], bins=50)
    # plot line graph of means
    plt.plot(means, '--', linewidth=2, color=mean_colour)
    plt.plot(medians, '--', linewidth=2, color=median_colour)
    # needs title
    plt.ylabel('Number of Moves')
    plt.xlabel('Generation')
    plt.colorbar()

    plt.show()

    return means

  def getLearningRate(self):
    scores = []
    cummulative = []
    champRange = []
    print("Getting learning Rate..")
    for i in range(len(self.statistics)):
      for j in self.statistics[i]['stats']:
        if j[0] == "Recent Scores":
          scores.append(float(ast.literal_eval(j[1])[-1]))
        if j[0] == "Cummulative Score":
          cummulative.append(float(j[1]))
        if j[0] == "Prev. Champ Point Range":
          champRange.append(ast.literal_eval(j[1]))

    
    # verify if we can compute the same cummulative score
    # base = 0
    # for i in range(len(cummulative)):
    #   base += round(scores[i],2)
    #   print(base, cummulative[i])
    # convert champion range into floats
    for i in range(len(champRange)):
      li = champRange[i]
      for j in range(len(li)):
        li[j] = float(li[j])
      champRange[i] = li
    
    # for i in champRange:
    #   print(i)

    # plot cummulative graph
    plt.plot(cummulative, '--', linewidth=2, color='blue')
    # needs title
    plt.ylabel('Learning Rate')
    plt.xlabel('Generation')
    plt.show()

if __name__ == '__main__':
  date = "2018-03-19 16:42:47"
  s = Statistics(date)
  s.loadStatisticsFile()
  # s.averageNumMovesPerGeneration()
  s.getLearningRate()