import tournament

def run():
  #config variables
  options = {
    'mongoConfigPath':'config.json',
    'plyDepth' : 4,
    'NumberOfGenerations' : 200,
    'Population' : 15,
    'loadRemoteMongo' : False
  }

  # run tournament
  t = tournament.Generator(options)
  t.runGenerations()

if __name__ == "__main__":
  run()