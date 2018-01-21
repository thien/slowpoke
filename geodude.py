"""
  Geodude
                                                  _,.---.
                                              _,-'       `.
                                          _,'  ,          \
                                        ,'  _,'   .        `.
                                        /  ,'     ,'          `.
              __                       .,'    _,'              `.
          _,..'  `-....___              :    ,'     '             \
        ,'   /            :             /`.,'      /               `
      /    /  ._         |         __..|  `.    .'       ,         `.
      |   |   ,'"--._    |      ,-'    `-._`.,-'       ,:            .
      .'\   \     _,'.    `'___.'           `"`.     _,' /            |
      |  \   \---'       ,"'  .-""'"----.       `.  '  ,'             |
      `. `-.'          /    /                    `-..^._             '
        |._|    _.    /    /                            `._           .
        `...:--'--+..'   ,'                              /            |
            '._  `|   ,-'       _..._                   j     \       |
              |` |   /       ,-'     `-.__              |      L      |
              |  |  /      ,'                           |      |      |
              |_,'        /         _,-                  .     |      |
              ,'  ,   |  ,'        ,|            ,..._     \    |      '
            ,     \ j  '       _." |           /     `-.__'    '    ,'
              +._   '|       ,'|    |          /        ,'    .'    /
              |  `._  `-' .:|  |    '.       -'        '           j
              '    |`    ' |'  |     |                             |
              `.  |       |--'     _|        .                    |
                \ |       '----'"'"'           \      __,....-+----'
                | '                            `---""      .' 
                `. `.                                     ,
                  `" \_...-"''"'--..         _+          ,'
                        '            -.'  `'  `.  ."-..'
                        `-..'._            _____,.'
                              `-'-'.....,-"' mh
"""

import numpy as np
import random
import math
import datetime
from math import log, sqrt

"""
Piece Weights
"""
pieceWeights = {
  "Black" : 0,
  "White" : 1,
  "empty" : -1,
  "blackKing" : -2,
  "whiteKing" : -3
}

# We arbitrarily defined the value of a winning board as +1.0 and a losing board as −1.0. All other boards would receive values between −1.0 and +1.0, with a neural network favoring boards with higher values.

minimax_win = 1
minimax_lose = -minimax_win
minimax_draw = 0
minimax_empty = -1

# -----------------------------------------------------------

class Geodude:
  def __init__(self, plyDepth=4):
    """
    Initialise Agent

    Note that we keep the weights since it is
    essential for the bot to evaluate the board.
    """
    self.ply = plyDepth


  def move_function(self, board, colour):
    return self.mcts_code(board,self.ply, colour)


  def mcts_code(self, B, ply, colour):
    moves = B.get_moves()

    # if theres only one move to make theres no point
    # evaluating future moves.
    if not moves:
      return
    if len(moves) == 1:
      return moves[0]

    self.mcts_plays, self.mcts_chances = {}, {}

    seconds = 0.5*self.ply

    self.calculation_time = datetime.timedelta(seconds=seconds)

    self.c = 1.4

    # if the user adds some dud plycount default to 1 random round.
    max_rounds = 1
    if self.ply > 0:
      max_rounds = 50*self.ply
    
    # begin mcts
    begin = datetime.datetime.utcnow()
    number_of_sims = 0
    ply = self.ply
    while datetime.datetime.utcnow() - begin < self.calculation_time:
      movebases = self.mcts_simulate(B,ply,colour,max_rounds)
      number_of_sims += 1
  
    move_states = []
    for i in moves:
      bruh = B.copy()
      bruh = bruh.make_move(i)
      move_states.append((i, hash(bruh.pdn['FEN'])))

    # Pick the move with the highest percentage of winning chances divided by the number of games.
    percent_winchance, best_move = max(
      (self.mcts_chances.get((colour, S), 0) /self.mcts_plays.get((colour, S), 1),p)
      for p, S in move_states
    )

    if colour == 0:
      percent_winchance, best_move = min(
      (self.mcts_chances.get((colour, S), 0) /self.mcts_plays.get((colour, S), 1),p)
      for p, S in move_states
    )

    

    # # Display the stats for each possible play.
    # goods = sorted(
    #   ((100 * self.mcts_chances.get((colour, S), 0) /
    #     self.mcts_plays.get((colour, S), 1),
    #     self.mcts_chances.get((colour, S), 0),
    #     self.mcts_plays.get((colour, S), 0), p)
    #    for p, S in move_states),
    #   reverse=True
    # )

    # for i in goods:
    #   print(i[3], "Moves:",i[2], "Good Moves",i[1], str(i[0]) + "%")

    # print ("Maximum depth searched:", ply)
    # print(percent_winchance)

    return best_move

  def mcts_simulate(self,B,ply,colour,rounds):
    """
    For a given hypothetical move, this function simulates how good the game is.
    It returns a percentage of how good it is. If it defaults to a winning game,
    then it'll return with certainty.
    """
    moves = B.get_moves()

    visited_states = set()
    state_stack = [B.copy()] # iterate a tree of moves stack style!
    state = state_stack[-1]
    player = colour

    expand = True

    # loop through all the moves
    for t in range(1, rounds+1):
      legal_moves = state.get_moves()

      # generate move, hypothetical states pair
      move_states = [(i, state.copy().make_move(i)) for i in legal_moves]

      if all(self.mcts_plays.get((player, S)) for p, S in move_states):
        # if we have the statistics on all of the legal moves here, use them!
  
        all_move_states = [mcts_plays[(player, S)] for p, S in move_states]
        if len(all_move_states) > 0:
          log_total = log(sum(all_move_states))

          value, move, state = max(
            (
              (
                self.mcts_chances[(player, S)] / self.mcts_plays[(player, S)]
              ) +
              (self.c * sqrt(log_total / self.mcts_plays[(player, S)])), p, S
            )
            for p, S in move_states
          )
          # print(value, move, state)
      else:
        choice = random.choice(move_states)
        move, state = choice
    
      # add current state to list of states
      state_stack.append(state)
      # `player` here and below refers to the player
      # who moved into that particular state.
    
      su = hash(state.pdn['FEN'])
      if expand and (player, su) not in self.mcts_plays:
        expand = False
        self.mcts_plays[(player, su)] = 0
        self.mcts_chances[(player, su)] = 0
        if t > ply:
          ply = t

      visited_states.add((player, su))
      player = state.current_player()
      winner = -1
      if state.is_over():
        winner = state.winner

      for player, x in visited_states:
        if (player, x) not in self.mcts_plays:
          continue
        # increment this position
      
        self.mcts_plays[(player, x)] += 1
        if player == winner:
          self.mcts_chances[(player, x)] += 1