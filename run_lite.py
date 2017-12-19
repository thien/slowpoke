import tournament

def run():
  #config variables
  options = {
    'mongoConfigPath':'config2.json',
    'plyDepth' : 1,
    'NumberOfGenerations' : 200,
    'Population' : 3,
    'printStatus' : True,
    'connectMongo' : False
  }

  # run tournament
  t = tournament.Generator(options)
  t.runGenerations()

if __name__ == "__main__":
  run()