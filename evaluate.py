import play as p
import json
import multiprocessing
import csv
import datetime

# player classes we want to play with.
players = {
  'slowpoke_gm' : p.loadPlayerClass('slowpoke'),
  'random' : p.loadPlayerClass('magikarp'),
  'slowpoke_99' : p.loadPlayerClass('slowpoke_99'),
  'slowpoke_149' : p.loadPlayerClass('slowpoke_149'),
  'slowpoke_no_subsquares' : p.loadPlayerClass('slowpoke_nosb'),
  'slowpoke_minimax' : p.loadPlayerClass('slowpoke_minimax'),
  'slowpoke0' : p.loadPlayerClass('slowpoke_0'),
  'slowpoke_r1' : p.loadPlayerClass('slowpoke_rand'),
  'slowpoke_r2' : p.loadPlayerClass('slowpoke_rand'),
  'slowpoke_r3' : p.loadPlayerClass('slowpoke_rand'),
  'slowpoke_r4' : p.loadPlayerClass('slowpoke_rand'),
  'slowpoke_r5' : p.loadPlayerClass('slowpoke_rand'),
  'mcts' : p.loadPlayerClass('geodude')
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
  ['random', 'random'],
  ['slowpoke_gm', 'random'],
  ['slowpoke_gm', 'slowpoke0'],
  ['slowpoke_gm', 'slowpoke_99'],
  ['slowpoke_gm', 'slowpoke_149'],
  ['slowpoke_gm', 'slowpoke_no_subsquares'],
  ['slowpoke_gm', 'slowpoke_minimax'],
  ['slowpoke_gm', 'slowpoke_r1'],
  ['slowpoke_gm', 'slowpoke_r2'],
  ['slowpoke_gm', 'slowpoke_r3'],
  ['slowpoke_gm', 'slowpoke_r4'],
  ['slowpoke_gm', 'slowpoke_r5'],
  ['slowpoke_gm', 'slowpoke_gm'],
  ['slowpoke_gm', 'mcts'],
  ['slowpoke_r1', 'mcts']
]

# --------------------------
# Multithreading Operations

cores = multiprocessing.cpu_count()-1

def gameWorker(i):
  score = p.runGame(i['black'], i['white'], i['gameOpt']).winner
  print(score)
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
  headings = []
  headings.append("player")
  headings.append("opponment")
  i = entry[list(entry.keys())[0]]
  for key in i:
    if isinstance(i[key], dict):
      for l, w in i[key].items():
        headings.append(key+ "_" + l)
    else:
      headings.append(key)
  csv_ent.append(headings)

  # add entries for values
  for key, value in entry.items():
    ent = []
    # ent.append(key)
    title = key.split("_vs_")
    for i in title:
      ent.append(i)
    for k, v in value.items():
      if isinstance(v, dict):
        for l, w in v.items():
          ent.append(w)
      else:
        ent.append(v)
    csv_ent.append(ent)
  return csv_ent

def printStatus(entry, startTime=None):
  print('\033c', end=None)
  for evaluate_id, _ in entry.items():
    scoreboard = str(entry[evaluate_id]['wins']) + ":" + str(entry[evaluate_id]['losses'])+":"+str(entry[evaluate_id]['draws'])
    print(evaluate_id, scoreboard)
    print("Wins:", entry[evaluate_id]['wins'], "("+str(entry[evaluate_id]['win_ratio']) + "%", "-", entry[evaluate_id]['as_black']['wins'], "as black,", entry[evaluate_id]['as_white']['wins'] , "as white; FMA Ratio: ",entry[evaluate_id]['first_mover_advantage']['win_ratio'], ")" )
    print("Losses:", entry[evaluate_id]['losses'], "("+str(entry[evaluate_id]['lose_ratio']) + "%", "-", entry[evaluate_id]['as_black']['losses'], "as black,", entry[evaluate_id]['as_white']['losses'] , "as white; FMA Ratio: ",entry[evaluate_id]['first_mover_advantage']['lose_ratio'], ")" )
    print("Draws:", entry[evaluate_id]['draws'], "("+str(entry[evaluate_id]['draw_ratio']) + "%", "-", entry[evaluate_id]['as_black']['draws'], "as black,", entry[evaluate_id]['as_white']['draws'] , "as white; FMA Ratio: ",entry[evaluate_id]['first_mover_advantage']['draw_ratio'], ")" )
    print("KD Ratio:",entry[evaluate_id]['kd_ratio'], "- Success Ratio:", entry[evaluate_id]['success_ratio'])
    print()

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
        'draws' : 0,
        'as_black' : {
          'wins' : 0,
          'losses' : 0,
          'draws' : 0
        },
        'as_white' : {
          'wins' : 0,
          'losses' : 0,
          'draws' : 0
        }
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
            'gameOpt' : gameOpt,
            'entry' : entry
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
      try:
        entry[evaluate_id]['kd_ratio'] =  entry[evaluate_id]['win_ratio']/entry[evaluate_id]['lose_ratio']
      except:
        entry[evaluate_id]['kd_ratio'] = '+inf'
      entry[evaluate_id]['success_ratio'] = 100 - entry[evaluate_id]['lose_ratio']

      bw_winratio, bw_loseratio, bw_drawratio = "-", "-", "-"
      try:
        bw_winratio = entry[evaluate_id]['as_black']['wins'] / entry[evaluate_id]['as_white']['wins']
      except:
        pass
      try:
        bw_loseratio = entry[evaluate_id]['as_black']['losses'] / entry[evaluate_id]['as_white']['losses']
      except:
        pass
      try:
        bw_drawratio = entry[evaluate_id]['as_black']['draws'] / entry[evaluate_id]['as_white']['draws']
      except:
        pass

      entry[evaluate_id]['first_mover_advantage'] = {
        'win_ratio' : bw_winratio,
        'lose_ratio' : bw_loseratio,
        'draw_ratio' : bw_drawratio
      }
      printStatus(entry)
      
      # now write it to json.
      with open(filename + ".json", 'w') as outfile:
        json.dump(entry, outfile)
      # now we need to make a csv.
      csv_ent = create_csv(entry)
      # print(csv_ent)

      with open(filename + '.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(csv_ent)):
          writer.writerow(csv_ent[i])
    print("evaluations done; saved results to file.")
  else:
    print("Evaluations cancelled.")

if __name__ == "__main__":
  if cores > 64:
    evaluate(games, 256, 'evaluations')
  else:
    evaluate(games, 2, 'evaluations')
