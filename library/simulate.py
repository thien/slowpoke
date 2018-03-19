#!/usr/bin/python
import core.tournament as tournament
import sys
import os
import datetime
# print ('Number of arguments:', len(sys.argv), 'arguments.')
# print ('Argument List:', str(sys.argv))

def train():
    options = {
        'mongoConfigPath':'config2.json',
        'Population' : 15,
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
            options['NumberOfGenerations'] = 100
            verifiedBool = True

        elif "heavy" in sys.argv:
            print("You are about to load a heavy simulation.")
            options['plyDepth'] = 6
            options['NumberOfGenerations'] = 200
            verifiedBool = True
        
        # check for user input
        if verifiedBool:
            print("Are you ready to run? Y/N")
            k = input()
            
            if k.upper() == "Y":
                print("dank")
                readyBool = True
        else:
            print("You didn't use an available option.")
            print("There are two options: light, heavy")
    else:
        # no arguments loaded; ask user for load type.
        print("You'll need to load some argument into this file. for instance:")
        print("     python3 simulate.py light")
    # run tournament
    if readyBool:
        t = tournament.Generator(options)
        t.runGenerations()
    else:
        print("Terminating.")

if __name__ == "__main__":
  train()