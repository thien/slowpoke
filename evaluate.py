import play as p
import json
import multiprocessing

# player classes we want to play with.
players = {
  'slowpoke' : p.loadPlayerClass('slowpoke'),
  'random' : p.loadPlayerClass('magikarp'),
  'slowpoke_gen100' : p.loadPlayerClass('slowpokeish'),
  'slowpoke_no_subsquares' : p.loadPlayerClass('slowpoke_nosb'),
  'slowpoke_minimax' : p.loadPlayerClass('slowpoke_minimax'),
  'slowpoke0' : p.loadPlayerClass('slowpoke0')
}

# configure game options so we don't print the game status
gameOpt = {
  'show_dialog' : False,
  'show_board' : False,
  'human_white' : False,
  'human_black' : False    
}

# load the games we want to see.
games = [
  ['slowpoke', 'random'],
  ['slowpoke', 'slowpoke_gen100'],
  ['slowpoke', 'slowpoke_no_subsquares'],
  ['slowpoke', 'slowpoke0'],
  ['slowpoke', 'slowpoke_minimax']
]

# gm vs random
# gm(200) vs gm100
# gm vs minimax
# gm(0) vs minimax(0)
# gm vs mcts
# gm(0) vs mcts


# --------------------------
# Multithreading Operations

cores = multiprocessing.cpu_count()-1

def gameWorker(i):
  score = p.runGame(i['black'], i['white'], i['gameOpt']).winner
  return score

# --------------------------

def verifyClasses(games):
  try:
    for x in games:
      k = {
          'black' : players[x[0]],
          'white' : players[x[1]],
          'gameOpt' : gameOpt
        }
    return True
  except KeyError as e:
    # print(KeyError)
    print("Error:",e, "isn't a recognised player type. Here's a traceback:")
    return False

def create_csv(entry):
  csv_ent = []
  # add entries for headings
  for key, value in entry.items():
    headings = []
    headings.append("opponment")
    for k, v in value.items():
      if isinstance(v, dict):
        for l, w in v.items():
          headings.append(k + "_" + l)
      else:
        headings.append(k)
    csv_ent.append(headings)
    break

  # add entries for values
  for key, value in entry.items():
    ent = []
    ent.append(value)
    for k, v in value.items():
      if isinstance(v, dict):
        for l, w in v.items():
          headings.append(w)
      else:
        headings.append(v)
    csv_ent.append(ent)
  
  return csv_ent

# iterate through the games.
def evaluate(games, numberOfGames=10, filename='evaluations'):
  if verifyClasses(games):
    entry = {}
    for x in games:
      black, white = 0, 1

      evaluate_id = x[black] + "_vs_" + x[white]

      entry[evaluate_id] = {
        'wins' : 0,
        'losses' : 0,
        'draws' : 0
      }
      print(evaluate_id)

      for j in range(0,2):
        scores = []
        # make sure they switch for black and white
        ent_string = "as_black"
        if j == 1:
          black, white = 1,0
          ent_string = "as_white"

        entry[evaluate_id][ent_string] = {} 

        gamePool = []
        for i in range(0,int(numberOfGames/2)):
          # play game and add result
          gamePool.append({
            'black' : players[x[black]],
            'white' : players[x[white]],
            'gameOpt' : gameOpt
          })

        # create game pool.
        with multiprocessing.Pool(processes=cores) as pool:
          scores = pool.map(gameWorker, gamePool)
        
        # count number of wins, losses, draws for given side.
        entry[evaluate_id][ent_string]['wins'] = scores.count(black)
        entry[evaluate_id][ent_string]['losses'] = scores.count(white)
        entry[evaluate_id][ent_string]['draws'] = scores.count(-1)

        # add to overall w/l/d
        entry[evaluate_id]['wins'] += entry[evaluate_id][ent_string]['wins']
        entry[evaluate_id]['losses'] += entry[evaluate_id][ent_string]['losses']
        entry[evaluate_id]['draws'] += entry[evaluate_id][ent_string]['draws']
      
      entry[evaluate_id]['win_ratio'] = entry[evaluate_id]['wins']/numberOfGames * 100
      entry[evaluate_id]['lose_ratio'] = entry[evaluate_id]['losses']/numberOfGames * 100
      entry[evaluate_id]['draw_ratio'] = entry[evaluate_id]['draws']/numberOfGames * 100
      entry[evaluate_id]['success_ratio'] = 100 - entry[evaluate_id]['lose_ratio']

      entry[evaluate_id]['first_mover_advantage'] = {
        'win_ratio' = entry[evaluate_id]["as_black"]['wins'] / entry[evaluate_id]["as_white"]['wins']
        'lose_ratio' = entry[evaluate_id]["as_black"]['losses'] / entry[evaluate_id]["as_white"]['losses']
        'draw_ratio' = entry[evaluate_id]["as_black"]['draws'] / entry[evaluate_id]["as_white"]['draws']
      }
      print(entry)
      # now write it to json.
      with open(filename + ".json", 'w') as outfile:
        json.dump(entry, outfile)
    # now we need to make a csv.
    csv_ent = create_csv(entry)
    for i in csv_ent:
      print(i)
    print("evaluations done!")
  else:
    print("Evaluations cancelled.")

if __name__ == "__main__":
  evaluate(games, 2, 'evaluations.json')