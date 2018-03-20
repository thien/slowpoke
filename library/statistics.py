import json
import os
import numpy as np
import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import ast
import time, datetime

class Statistics:
  def __init__(self, date, defaultResultsPath=None):
    self.date = date
    self.path = os.path.join("..", "results")
    if defaultResultsPath:
      self.path = defaultResultsPath
    self.statistics = {}
    self.directory = os.path.join(self.path, self.date)
    leaderboards = []
    self.saveChartsToImages = True
    self.gmFilename = "gm_stats.json"
    self.enableTitles = True
  
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

  def saveChartToFile(self, title, chart, filetype="eps"):
    directory = os.path.join(self.directory,"charts")
    if not os.path.isdir(directory):
      os.makedirs(directory)
    filename = title + "." + filetype
    filepath = os.path.join(directory, filename)
    chart.savefig(filepath, bbox_inches='tight')
    print("saved",filename)

  """
  Gets tournament timing information
  """
  def timeStatsPerGeneration(self):
    sx = []
    plotx = []
    ploty = []
    means = []
    medians = []
    for i in range(len(self.statistics)):
      times = {
        'gameDurations' : []
      }
      # get generation tournament length
      times["duration"] = self.statistics[i]['durationInSeconds']
      # get individual game lengths
      for j in self.statistics[i]['games']:
        timestamp = j['duration']
        x = time.strptime(timestamp,'%H:%M:%S.%f')
        su = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
        # print(timestamp,su)
        times['gameDurations'].append(su)
        plotx.append(i)
        ploty.append(su)
      # sort the duration times
      times['gameDurations'] = sorted(times['gameDurations'])
      mean = sum(times['gameDurations']) / float(len(times['gameDurations']))
      # print(i, "Mean:", round(mean,2), "S:", moveCounts[0], "L:", moveCounts[-1])
      means.append(mean)
      medians.append(max(set(times['gameDurations']), key=times['gameDurations'].count))
      sx.append(times)

    # generate a graph for this
    # for i in sx:
    #   print(i['duration'])

    simRuntimes = [float(i['duration']) for i in sx]
    # for i in range(len(sx)):
    plotx.append(1)
    ploty.append(max(simRuntimes)+10)
    # print(simRuntimes)
    mean_colour = "red"
    net_colour = "pink"
    # add legend
    mean_patch = mpatches.Patch(color=mean_colour, label='Mean')
    gen_patch = mpatches.Patch(color=net_colour, label='CPU Runtime')
    plt.legend(handles=[mean_patch, gen_patch])

    # plot 2d histogram
    plt.hist2d(plotx, ploty, bins=50)
    # plot line graph of means
    plt.plot(means, '--', linewidth=2, color=mean_colour)
    # plot overall sim time
    plt.plot(simRuntimes, '--', linewidth=2, color=net_colour)
    # plt.plot(medians, '--', linewidth=2, color=median_colour)
    # needs title
    plt.ylabel('Seconds')
    plt.xlabel('Generation')
    if self.enableTitles:
      plt.suptitle('Game Run Time Over Generations')
    # plt.colorbar()
    # set axis
    x1,x2,y1,y2 = plt.axis() # get current axis
    plt.axis((x1,x2,y1,max(simRuntimes)))

    # plt.show()
    if self.saveChartsToImages:
      self.saveChartToFile("simulation_timings",plt)
    plt.close()

    return sx

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
    if self.enableTitles:
      plt.suptitle('Move Count Distribution Over Generations')
    plt.colorbar()

    # plt.show()
    if self.saveChartsToImages:
      self.saveChartToFile("moves",plt)
    plt.close()
    return means

  """
  Calculates the scores and learning rates
  """
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

    # plot cummulative graph
    plt.axhline(0, color='grey')
    plt.plot(cummulative, '--', linewidth=2, color='blue')
    # needs title
    if self.enableTitles:
      plt.suptitle('Cummulative Learning Rate Over Generations')
    plt.ylabel('Learning Rate (Cummulative)')
    plt.xlabel('Generation')
    # plt.show()
    if self.saveChartsToImages:
      self.saveChartToFile("cummulative_growth", plt)
    plt.close()

    # plot standard point range
    plt.axhline(0, color='grey')
    plt.plot(scores, '--', linewidth=2, color='blue')
    # needs title
    if self.enableTitles:
      plt.suptitle('Champion Scores Over Generations')
    plt.ylabel('Scores')
    plt.xlabel('Generation')
    # plt.show()
    if self.saveChartsToImages:
      self.saveChartToFile("champ_scores", plt)
    plt.close()


  """
  Load GM file
  """
  def loadGMFile(self):
    filepath = os.path.join(self.directory, self.gmFilename)
    print("Loading Statistics from file:")
    print("\t", filepath)
    f = open(filepath, 'r')
    self.gm_stats = json.load(f)
    f.close()
    print("Loaded GM Stats File!")

  """
  Parse the GM performance
  """
  def analyseGM(self):
    opp_names = []
    w,l,d = [], [], []
    aw_w, aw_l, aw_d = [],[],[]
    ab_w, ab_l, ab_d = [],[],[]

    # load the gm_stats file.load
    for i in self.gm_stats.keys():
      intel = self.gm_stats[i]
      # print(intel)
      opp_name = intel['opp'].upper().replace("ERATION", "")
      opp_names.append(opp_name)
      w.append(intel['wins'])
      l.append(intel['losses'])
      d.append(intel['draws'])
      aw_w.append(intel['as_white']['wins'])
      aw_l.append(intel['as_white']['losses'])
      aw_d.append(intel['as_white']['draws'])
      ab_w.append(intel['as_black']['wins'])
      ab_l.append(intel['as_black']['losses'])
      ab_d.append(intel['as_black']['draws'])
    
    # for i in range(len(opp_names)):
    #   print(ab_w[i], aw_w[i], ab_d[i], aw_d[i], ab_l[i], aw_l[i])

    # create overal wld chart
    t = "Overall Game Statistics Against All Opponments"
    data = [aw_w, aw_d, aw_l, ab_w, ab_d, ab_l]
    self.createWDLChart(opp_names,data,t)
    # plt.show()
    if self.saveChartsToImages:
      self.saveChartToFile("gm_net_stats",plt)
    plt.close()
    
    return False

  def createWDLChart(self, opp_names,data,title=None):
    ww,dw,lw,wb,db,lb = data
    #  normalise values to 100%
    totals = [i+j+k+l+m+n for i,j,k,l,m,n in zip(ww,dw,lw,wb,db,lb)]
    ww = [i / j * 100 for i,j in zip(ww, totals)]
    dw = [i / j * 100 for i,j in zip(dw, totals)]
    lw = [i / j * 100 for i,j in zip(lw, totals)]
    wb = [i / j * 100 for i,j in zip(wb, totals)]
    db = [i / j * 100 for i,j in zip(db, totals)]
    lb = [i / j * 100 for i,j in zip(lb, totals)]

    r = [x for x in range(len(opp_names))]

    # plot
    names = opp_names
    # Create green Bars
    plt.bar(r, wb, color='#27375a', edgecolor='white',  label="Win (Black)")
    plt.bar(r, ww, bottom=wb, color='#377bae', edgecolor='white',  label="Win (White)")
    # Create orange Bars
    plt.bar(r, db, bottom=[i+j for i,j in zip(wb,ww)], color='#c340a9', edgecolor='white',  label="Draw (Black)")
    plt.bar(r, dw, bottom=[i+j+k for i,j,k in zip(wb,ww,db)], color='#9310ab', edgecolor='white',  label="Draw (White)")
    # Create blue Bars
    plt.bar(r, lb, bottom=[i+j+k+l for i,j,k,l in zip(wb,ww,db,dw)], color='#d3d460', edgecolor='white',  label="Loss (Black)")
    plt.bar(r, lw, bottom=[i+j+k+l+m for i,j,k,l,m in zip(wb,ww,db,dw,lb)], color='#eeac0e', edgecolor='white',  label="Loss (White)")
    
    # Custom x axis
    plt.xticks(r, names)
    plt.xlabel("Opponments")
    plt.ylabel("Percentage")

    plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    if self.enableTitles:
      if title:
        plt.suptitle(title)

  def saveCharts(self):
    self.averageNumMovesPerGeneration()
    self.getLearningRate()
    self.timeStatsPerGeneration()

if __name__ == '__main__':
  date = "2018-03-19 16:42:47"
  s = Statistics(date)
  # s.loadStatisticsFile()
  # s.saveCharts()
  s.loadGMFile()
  s.analyseGM()
  # s.averageNumMovesPerGeneration()
  # s.getLearningRate()
  # s.timeStatsPerGeneration()