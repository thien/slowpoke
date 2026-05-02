#!/usr/bin/python
import core.tournament as tournament
import sys
import os
import datetime
import evaluator
import statistics
# print ('Number of arguments:', len(sys.argv), 'arguments.')
# print ('Argument List:', str(sys.argv))

def train():
    options = {
        'mongoConfigPath':'config2.json',
        'Population' : 15,
        'debugMode' : False, # if enabled, makes the system play randomly for testing purposes
        'printStatus' : True,
        'connectMongo' : False,
        'resultsLocation' : os.path.join("..", "results")
    }
    readyBool = False
    verifiedBool = False
    # Check for arguments
    if len(sys.argv) > 1:
        
        # check arguments
        if "light" in sys.argv:
            print("You are about to load a light simulation.")
            options['plyDepth'] = 1
            options['NumberOfGenerations'] = 200
            verifiedBool = True

        elif "medium" in sys.argv:
            print("You are about to load a medium simulation.")
            options['plyDepth'] = 3
            options['NumberOfGenerations'] = 200
            verifiedBool = True

        elif "heavy" in sys.argv:
            print("You are about to load a heavy simulation.")
            options['plyDepth'] = 6
            options['NumberOfGenerations'] = 200
            verifiedBool = True

        elif "ohno" in sys.argv:
            print("You are about to load a really heavy simulation.")
            options['plyDepth'] = 8
            options['NumberOfGenerations'] = 1500
            verifiedBool = True
        
        elif "vheavy" in sys.argv:
            print("You are about to load a VERY HEAVY simulation (12 ply).")
            print("This will be extremely computationally intensive!")
            options['plyDepth'] = 12
            options['NumberOfGenerations'] = 500
            verifiedBool = True

        if "debug" in sys.argv:
            print("You are about to load a debug simulation.")
            options['plyDepth'] = 1
            options['debugMode'] = True
            options['NumberOfGenerations'] = 200
            verifiedBool = True
        
        # parallel threads option
        if "--parallel" in sys.argv:
            idx = sys.argv.index("--parallel")
            if idx + 1 < len(sys.argv):
                try:
                    options['numParallel'] = int(sys.argv[idx + 1])
                    print(f"Using {options['numParallel']} parallel threads.")
                except ValueError:
                    print("Invalid parallel count, using default (4).")
                    options['numParallel'] = 4
            else:
                options['numParallel'] = 4
        
        # check for user input
        if verifiedBool:
            print("Are you ready to run? Y/N")
            k = input()
            
            if k.upper() == "Y":
                print("dank")
                readyBool = True
        else:
            print("You didn't use an available option.")
            print("options: light, medium, heavy, vheavy, debug, ohno")
    else:
        # no arguments loaded; ask user for load type.
        print("You'll need to load some argument into this file. for instance:")
        print("     python3 simulate.py light")
        print("     python3 simulate.py vheavy --parallel 8")
    # run tournament
    if readyBool:
        t = tournament.Generator(options)
        t.runGenerations()
        # create statistics
        stats = statistics.Statistics(t.folderName)
        stats.loadStatisticsFile()
        stats.saveCharts()
        # stats.averageNumMovesPerGeneration()
        # stats.getLearningRate()
        # stats.timeStatsPerGeneration()

        # evaluate performance
        su = evaluator.Evaluate(t.folderName,options['plyDepth'])
        su.loadChampions()
        games = su.createGames()
        su.evaluate(games)

        # create statistics of Gold Master
        stats.loadGMFile()
        stats.analyseGM()
        # print that we're done.
        print("DONE!")
    else:
        print("Terminating.")

if __name__ == "__main__":
  train()