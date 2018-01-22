#!/usr/bin/python
import tournament
import sys

# print ('Number of arguments:', len(sys.argv), 'arguments.')
# print ('Argument List:', str(sys.argv))

def train():
    options = {}
    readyBool = False
    # Check for arguments
    if len(sys.argv) > 1:
        # check arguments
        if "light" in sys.argv:
            print("You are about to load a light simulation.")
            options = {
                'mongoConfigPath':'config2.json',
                'plyDepth' : 1,
                'NumberOfGenerations' : 100,
                'Population' : 15,
                'printStatus' : True,
                'connectMongo' : False
            }

        elif "heavy" in sys.argv:
            print("You are about to load a heavy simulation.")
            options = {
                'mongoConfigPath':'config2.json',
                'plyDepth' : 6,
                'NumberOfGenerations' : 200,
                'Population' : 15,
                'printStatus' : True,
                'connectMongo' : False
            }
        
        # check for user input
        print("Are you ready to run? Y/N")
        k = input()
        
        if k.upper() == "Y":
            print("dank")
            readyBool = True
    else:
        # no arguments loaded; ask user for load type.
        input()
    # run tournament
    if readyBool:
        t = tournament.Generator(options)
        t.runGenerations()
    else:
        print("Terminating.")

if __name__ == "__main__":
  train()