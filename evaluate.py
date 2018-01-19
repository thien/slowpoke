import play as p
import json
import multiprocessing

# player classes we want to play with.
players = {
  'slowpoke' : p.loadPlayerClass('slowpoke'),
  'random' : p.loadPlayerClass('magikarp'),
  'slowpoke_gen100' : p.loadPlayerClass('slowpokeish'),
  'slowpoke_no_subsquares' : p.loadPlayerClass('slowerpoke'),
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

# iterate through the games.
def evaluate(games, numberOfGames=10, filename='evaluations.json'):
  for x in games:
    entry = {}

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
      if j == 1:
        black, white = 1,0
    
      entry[evaluate_id][j] = {} 

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
      entry[evaluate_id][j]['wins'] = scores.count(black)
      entry[evaluate_id][j]['losses'] = scores.count(white)
      entry[evaluate_id][j]['draws'] = scores.count(-1)

      # add to overall w/l/d
      entry[evaluate_id]['wins'] += entry[evaluate_id][j]['wins']
      entry[evaluate_id]['losses'] += entry[evaluate_id][j]['losses']
      entry[evaluate_id]['draws'] += entry[evaluate_id][j]['draws']
    
    entry[evaluate_id]['win_ratio'] = entry[evaluate_id]['wins']/numberOfGames * 100
    entry[evaluate_id]['lose_ratio'] = entry[evaluate_id]['losses']/numberOfGames * 100
    entry[evaluate_id]['draw_ratio'] = entry[evaluate_id]['draws']/numberOfGames * 100
    entry[evaluate_id]['success_ratio'] = 100 - entry[evaluate_id]['lose_ratio']
    print(entry)
    # now write it to json.
    with open(filename, 'w') as outfile:
      json.dump(entry, outfile)

if __name__ == "__main__":
  evaluate(games, 256, 'evaluations.json')