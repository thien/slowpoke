import tournament

def run():
  #config variables
  options = {
    'mongoConfigPath':'confige.json',
    'plyDepth' : 1,
    'NumberOfGenerations' : 20,
    'Population' : 2,
    'loadRemoteMongo' : False
  }

  # run tournament
  t = tournament.Generator(options)
  t.runGenerations()

if __name__ == "__main__":
  run()