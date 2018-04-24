import matplotlib
import multiprocessing
# dirty mira check
if multiprocessing.cpu_count() > 10: matplotlib.use('Agg')

import json
import os
import numpy as np
import sys

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import ast
import time, datetime
import hashlib
import random

class Statistics:
  def __init__(self, date, defaultResultsPath=None):
    self.date = date
    self.path = os.path.join("..", "results")
    if defaultResultsPath:
      self.path = defaultResultsPath
    self.statistics = {}
    self.directory = os.path.join(self.path, self.date)
    self.leaderboards = []
    self.saveChartsToImages = True
    self.gmFilename = "gm_stats.json"
    self.enableTitles = True
    self.debug = True
  
  def loadStatisticsFile(self, filename="statistics.json"):
    filepath = os.path.join(self.directory, filename)
    if self.debug:
      print("Loading Statistics from file:")
      print("\t", filepath)
    try:
      f = open(filepath, 'r')
      self.statistics = json.load(f)
      f.close()
      if self.debug:
        print("Loaded Stats File!")
        return True
    except:
      return False

  def parseLeaderboards(self): 
    leaderboards = []
    for i in range(len(self.statistics)):
      lbEntry = {}
      lb = None
      for k in range(len(self.statistics[i]['stats'])):
        j = self.statistics[i]['stats'][k]
        if j[0] == "Previous Scoreboard":
          lb = self.statistics[i]['stats'][k+1][0]
      lb = lb.split("\n")
      # empty item in the list; remove that
      lb.pop()
      for p in range(len(lb)):
        lb[p] = lb[p].split("\t")
        lb[p][0]=lb[p][0].replace("Player ", "")
      # add results into the dict.
      lbEntry['champion'] = lb[0][0]
      lbEntry['scores'] = lb
      lbEntry['players'] = [lb[i][0] for i in range(len(lb))]
      # print(lb)
      leaderboards.append(lbEntry)
    self.leaderboards = leaderboards.copy()
    # lets do some processing of the leaderboards.
    stats = {
      "persistent" : [],
      "elitism" : [],
      "mutation" : [],
      "crossover" : []
    }
    # copy the keys to make scoreStats
    scoreStats = {}
    for key in stats.keys():
      scoreStats[key] = []

    for i in range(1,len(leaderboards)):
      lb = leaderboards[i-1]
      # print(i)
      next_lb = leaderboards[i]
      # print(i+1,sorted(int(next_lb['players'])))
      sorted_scores = sorted([int(i) for i in next_lb['players']])
      # print(i+1,sorted_scores[-10:])

      if lb['champion'] == next_lb['champion']:
        # if the following generation's champion remains the same
        stats["persistent"].append(i+1)
      elif next_lb['champion'] in lb['players']:
        # if the following generation is from the previous leaderboard
        stats['elitism'].append(i+1)
      elif sorted_scores[-10:].index(int(next_lb['champion'])) < 5:
        # if its in the first 5 agents of the offspring then its crossed over
        stats['crossover'].append(i+1)
      else:
        stats['mutation'].append(i+1)
    # print(stats)
  
    # add score stats
    for i in range(0,len(self.scores['scores'])):
      score = self.scores['scores'][i]
      for key in stats:
        if i in stats[key]:
          # print(i, "is in", key, "and its score is",score)
          scoreStats[key].append(score)
    
    
    # iterate through the 
    for playerType in stats:
      # print(scoreStats[playerType])
      # print(playerType, len(stats[playerType]))
      sums = sum(scoreStats[playerType])
      size = len(scoreStats[playerType])
      # calculate the score mean
      mean = round(sums/size, 5)
      # print(mean)

    syu = [scoreStats[s] for s in scoreStats]
    xlabels = [x.title() for x in scoreStats]
    # print(syu)
    # plotScoreStats = np.concatenate((spread, center, flier_high, flier_low), 0)
    plt.figure()
    plt.boxplot(syu,showfliers=False)
    plt.xticks([i+1 for i in range(len(scoreStats))],xlabels)
    # add zeroline
    plt.axhline(0, color='grey')
    # add jitter to boxplot
    jitterx = []
    jittery = []

    for i in range(len(syu)):
      values = syu[i]
      x = np.random.normal(i+1, 0.07, size=len(values))
      for v in range(len(values)):
        jitterx.append(x[v])
        jittery.append(values[v])
        
    plt.scatter(jitterx,jittery,c='r', zorder=10, alpha=0.25)

    t = "Distribution of Champion Learning Rates by Agent Type"
    if self.enableTitles:
      plt.suptitle(t)
    if self.saveChartsToImages:
      self.saveChartToFile("champ_score_distribution", plt)
    # plt.show()
    plt.close()

    # now we create a chart of champion distributions.
    labels = xlabels
    sizes = [sum(stats[x]) for x in [y for y in scoreStats]]
    explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    cols = {
      'Persistent' : "#FF4E50", 
      'Elitism' : "#FC913A", 
      'Mutation' : "#B9D7D9",
      'Crossover' : "#99B898"
    }

      #  colors = [self.string2HexColor(x) for x in xlabels]
    colors = [cols[x] for x in xlabels]
    # print(colors, xlabels)
    # input()
    # plot subplots
    _, ax1 = plt.subplots()
    # generate pie chart
    _, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=False, startangle=90, pctdistance=0.5, labeldistance=0.8)
    fontsize = 13
    texts = [ _.set_fontsize(fontsize) for _ in texts]
    autotexts = [ _.set_fontsize(fontsize) for _ in autotexts]
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    t = "Generation Champion Distribution"

    # save chart
    if self.enableTitles:
      plt.suptitle(t)
    # plt.show()
    if self.saveChartsToImages:
      self.saveChartToFile("champ_gen_dist", plt)
    plt.close()

  @staticmethod
  def string2HexColor(string):
    hex = hashlib.md5(string.encode("utf-8")).hexdigest()
    # randomint = random.randint(1,len(hex)-6)
    randomint = 5
    return "#" + hex[randomint:randomint+6]

  def saveChartToFile(self, title, chart, filetype="pdf"):
    directory = os.path.join(self.directory,"charts")
    if not os.path.isdir(directory):
      os.makedirs(directory)
    filename = title + "." + filetype
    filepath = os.path.join(directory, filename)
    chart.savefig(filepath, bbox_inches='tight')
    if self.debug:
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

    self.timings = {
      "sx" : sx,
      "plotx" : plotx,
      "ploty" : ploty
    }
    return sx

  """
  Calculates the average number of moves per generation.
  """
  def averageNumMovesPerGeneration(self):
    if self.debug:
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
    # median_patch = mpatches.Patch(color=median_colour, label='Median')
    plt.figure(num=None, figsize=(5,6), facecolor='w', edgecolor='k')
    plt.legend(handles=[mean_patch])

    # plot 2d histogram
    plt.hist2d(graphData['x'], graphData['y'], bins=50)
    # plot line graph of means
    plt.plot(means, '--', linewidth=2, color=mean_colour)
    # plt.plot(medians, '--', linewidth=2, color=median_colour)
    # needs title
    plt.ylabel('Number of Moves')
    plt.xlabel('Generation')
    if self.enableTitles:
      plt.suptitle('Move Count Distribution Over Generations')
    plt.colorbar(orientation="horizontal", shrink=0.8, pad=0.1)

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
    if self.debug:
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

    self.scores = {
      "scores":scores,
      "cummulative":cummulative,
      "champRange": champRange
    }

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
    if self.debug:
      print("Loading Statistics from file:")
      print("\t", filepath)
    f = open(filepath, 'r')
    self.gm_stats = json.load(f)
    f.close()
    if self.debug:
      print("Loaded GM Stats File!")

  """
  Parse the GM performance
  """
  def analyseGM(self):
    opp_names = []
    w,l,d = [], [], []
    aw_w, aw_l, aw_d = [],[],[]
    ab_w, ab_l, ab_d = [],[],[]

    gameKeys = list(self.gm_stats.keys())
    ints = []
    elses = []
    for i in range(len(gameKeys)):
      if("mcts" not in gameKeys[i]) and ("random" not in gameKeys[i]):
        inty = ''.join(x for x in gameKeys[i] if x.isdigit())
        ints.append(int(inty))
      else:
        elses.append(gameKeys[i])
    ints = sorted(ints)
    for i in range(len(ints)):
      ints[i] = "gm_vs_gen-"+str(ints[i])
    ints = ints + elses
    
    # load the gm_stats file.load
    for i in ints:
      if "mcts" not in i:
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

    # create overall wld chart
    t = "Overall Game Statistics Against All Opponents"
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
    colours = {
      "lose_white" : "#B80021",
      "lose_black" : "#540000",
      "draw_white" : "#FBDF1A",
      "draw_black" : "#DE9400",
      "win_white" : "#99B800",
      "win_black" : "#3A6100"
    }
    names = opp_names
    # Create green Bars
    plt.bar(r, wb, color=colours['win_black'], edgecolor='white',  label="Win (Black)")
    plt.bar(r, ww, bottom=wb, color=colours['win_white'], edgecolor='white',  label="Win (White)")
    # Create orange Bars
    plt.bar(r, db, bottom=[i+j for i,j in zip(wb,ww)], color=colours['draw_black'], edgecolor='white',  label="Draw (Black)")
    plt.bar(r, dw, bottom=[i+j+k for i,j,k in zip(wb,ww,db)], color=colours['draw_white'], edgecolor='white',  label="Draw (White)")
    # Create blue Bars
    plt.bar(r, lb, bottom=[i+j+k+l for i,j,k,l in zip(wb,ww,db,dw)], color=colours['lose_black'], edgecolor='white',  label="Loss (Black)")
    plt.bar(r, lw, bottom=[i+j+k+l+m for i,j,k,l,m in zip(wb,ww,db,dw,lb)], color=colours['lose_white'], edgecolor='white',  label="Loss (White)")
    
    # Custom x axis
    plt.xticks(r, names)
    plt.xlabel("Opponments")
    plt.ylabel("Percentage")

    # box = plt.get_position()
    # plt.set_position([box.x0, box.y0 + box.height * 0.1,
    #                 box.width, box.height * 0.9])


    # plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    if self.enableTitles:
      if title:
        plt.suptitle(title)

    # now we attempt to create a trendline
    overall_wins = [i+j for i,j in zip(ww,wb)]
    overall_draws = [i+j for i,j in zip(dw,db)]
    overall_losses = [i+j for i,j in zip(lw,lb)]

    # trendlines (3rd degree polynonial i.e cubic trendline)
    z = np.polyfit(r, overall_wins, 3)
    p1 = np.poly1d(z)
    z = np.polyfit(r, overall_draws, 3)
    p2 = np.poly1d(z)
    z = np.polyfit(r, overall_losses, 3)
    p3 = np.poly1d(z)
    overall_wins = p1(r)
    overall_draws = p2(r)
    overall_losses = p3(r)

    # print(overall_wins)
    r = r[0:-1]
    su = plt.plot(r,overall_wins[0:-1], "--", linewidth=2, label="Win Trend", color=self.hexMedian(colours['win_black'], colours['win_white']))
    su = plt.plot(r,overall_draws[0:-1], "-.", linewidth=2, label="Draw Trend", color=self.hexMedian(colours['draw_black'], colours['draw_white']))
    su = plt.plot(r,overall_losses[0:-1], ":", linewidth=2, label="Loss Trend", color=self.hexMedian(colours['lose_black'], colours['lose_white']))
    # print(self.hexMedian(colours['win_black'], colours['win_white']))

    # Put a legend below current axis
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=3)


  @staticmethod
  def hexMedian(a,b):
    a = a[1:]
    b = b[1:]
    colour = ""
    # colour wise split
    for i in range(0,6,2):
      na = int("0x" + a[i:i+2], 16)
      nb = int("0x" + b[i:i+2], 16)
      # find the median colour between the two.
      c = int((na + nb) / 2)
      # darken the colour a bit
      c = int(c*0.7)
      col = hex(c)[2:]
      if len(col) == 1:
        col = "0" + col
      colour = colour + col
    return "#" + colour

  def saveCharts(self):
    self.averageNumMovesPerGeneration()
    self.getLearningRate()
    self.timeStatsPerGeneration()
    self.parseLeaderboards()

def batchRun():
  """
  Loads all the files in the repo and creates charts for all files if
  it can.
  """
  path = os.path.join("..", "results")
  folders = os.listdir(path)
  stats = []
  times = []
  # s = Statistics()
  for folder in folders:
    newPath = os.path.join(path, folder)
    verify = os.path.isdir(newPath)
    if verify:
      # check if theres a statistics.json.
      if "statistics.json" in os.listdir(newPath):
        print("Running stats for", folder)
        s = Statistics(folder)
        s.saveChartsToImages = True
        s.enableTitles = False
        s.debug = False
        s.loadStatisticsFile()
        s.saveCharts()
        if "gm_stats.json" in os.listdir(newPath):
          s.loadGMFile()
          s.analyseGM()
        stats.append({
          "folder":folder,
          "ob":s
        })
  runBatchCummulativeChart(stats)
  # measureBatchTimings(stats)

def measureBatchTimings(stats):
  timings ={}
  for ob in stats:
    folderName = ob['folder'][20:].replace(" generations)","")
    folderName = folderName.split("ply")
    folderName[0] = int(folderName[0][1])
    folderName[1] = int(folderName[1])
    timings[tuple(folderName)] = ob['ob'].timings

  # init timings graph
  s = Statistics("General")
  s.enableTitles = False
  plt.close()

  colours = []
  labels = []
  # iterate through the items
  for i in sorted(list(timings)):
    print("NEW")
    batchx = []
    batchy = []
    count = 1
    for round in timings[i]['sx']:
      cpuDuration = float(round['duration'])
      games = round['gameDurations']
      batchy.append(cpuDuration)
      batchx.append(count)
      count += 1
    # print(batchy)
    hexColour = s.string2HexColor(str(i))
    colours.append(hexColour)
    keyString = str(i[0]) + " Ply"

    for i in range(1,len(batchy)):
      batchy[i] = batchy[i]+batchy[i-1]
    print(keyString, batchy[-1])
    print("runtime",str(datetime.timedelta(seconds=batchy[-1])))
    print("mean",str(datetime.timedelta(seconds=batchy[-1]/len(batchx))))

    su = plt.plot(batchx, batchy, "-", linewidth=2, color=hexColour, label=keyString)
  # plt.legend(handles=su, labels = labels)

  # if s.enableTitles:
  #   plt.suptitle('Cummulative Learning Rate Over Generations')

  # plt.ylabel('Learning Rate (Cummulative)')
  # plt.xlabel('Generation')
  plt.show()

def runBatchCummulativeChart(stats):
  print("-----------")
  # parse the stats objects to get the cummulative items
  cummulatives = {}
  for ob in stats:
    folderName = ob['folder'][20:].replace(" generations)","")
    folderName = folderName.split("ply")
    folderName[0] = int(folderName[0][1])
    folderName[1] = int(folderName[1])
    cummulatives[tuple(folderName)] = ob['ob'].scores['cummulative']
    

  s = Statistics("General")
  s.enableTitles = False
  # iterate through the keys
  colours = []
  charts = []
  labels = []

  plt.close()
  plt.figure(num=None, figsize=(10,4), facecolor='w', edgecolor='k')
  plt.axhline(0, color='grey')
  
  for cum in sorted(list(cummulatives)):
    print(cum)
    hexColour = s.string2HexColor(str(cum))
    colours.append(hexColour)
    keyString = str(cum[0]) + " Ply"
    su = plt.plot(cummulatives[cum], "-", linewidth=2, color=hexColour, label=keyString)
    charts.append(su)
    # calculate the trend line
    y = cummulatives[cum]
    x = [x for x in range(len(y))]
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(p(x), "r-.",color=hexColour, label=keyString + " Trend")
    

  plt.legend(handles=su, labels = labels)
  

  if s.enableTitles:
    plt.suptitle('Cummulative Learning Rate Over Generations')

  plt.ylabel('Learning Rate (Cummulative)')
  plt.xlabel('Generation')
  # plt.show()
  # if self.saveChartsToImages:
  #   self.saveChartToFile("cummulative_growth", plt)
  title = "combined_cummulative"
  s.saveChartToFile(title, plt)
  plt.close()
  
def handleArguments():
  for i in range(1,len(sys.argv)):
    entry = sys.argv[i]
    maxSize = len(sys.argv)-1
    # look at flags
    if (entry == "-folder") or (entry == "-f"):
      if i != maxSize:
        foldername = sys.argv[i+1]
        # check if this foldername is actually a directory.
        s = Statistics(foldername)
        s.saveChartsToImages = False
        check = s.loadStatisticsFile()
        if check:
          s.saveCharts()
          s.loadGMFile()
          s.analyseGM()
        else:
          print("The folder either does not exist or a statistics.json is not found.")
      else:
        print("You didn't add a folder name. Please try again.")
    elif (entry == "-batch") or (entry == "-b"):
      batchRun()
    else:
      print("You need some arguments:")
      print("-b : Batch run all the folders with statistics inside")
      print('-f "foldername" : Generate statistics in one folder')
  

if __name__ == '__main__':
  # batchRun()
  handleArguments()
  # foldername = "2018-03-19 16:42:47 (1ply 100 generations)"
  # s = Statistics(foldername)
  # s.saveChartsToImages = False
  # s.loadStatisticsFile()
  # s.getLearningRate()
  # s.parseLeaderboards()
  # s.saveCharts()
  # s.loadGMFile()
  # s.analyseGM()
  # s.averageNumMovesPerGeneration()
  # s.getLearningRate()
  # s.timeStatsPerGeneration()