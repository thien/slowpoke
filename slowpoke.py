"""
  Slowpoke
                                      _.---"'----"'`--.._
                                _,.-'                   `-._
                            _,."                            -.
                        .-""   ___...---------.._             `.
                        `---'""                  `-.            `.
                                                    `.            \
                                                      `.           \
                                                        \           \
                                                          .           \
                                                          |            .
                                                          |            |
                                    _________             |            |
                              _,.-'"         `"'-.._      :            |
                          _,-'                      `-._.'             |
                      _.'                              `.             '
            _.-.    _,+......__                           `.          .
          .'    `-"'           `"-.,-""--._                 \        /
        /    ,'                  |    __  \                 \      /
        `   ..                       +"  )  \                 \    /
        `.'  \          ,-"`-..    |       |                  \  /
          / " |        .'       \   '.    _.'                   .'
        |,.."--"-"--..|    "    |    `""`.                     |
      ,"               `-._     |        |                     |
    .'                     `-._+         |                     |
    /                           `.                        /     |
    |    `     '                  |                      /      |
    `-.....--.__                  |              |      /       |
      `./ "| / `-.........--.-   '              |    ,'        '
        /| ||        `.'  ,'   .'               |_,-+         /
        / ' '.`.        _,'   ,'     `.          |   '   _,.. /
      /   `.  `"'"'""'"   _,^--------"`.        |    `.'_  _/
      /... _.`:.________,.'              `._,.-..|        "'
    `.__.'                                 `._  /
                                              "' 

  Slowpoke is a draughts AI based on a convolutional neural network.
  It's been trained using genetic algorithms :)
"""

import numpy as np
import random
import math
import datetime
from neural import NeuralNetwork
from math import log, sqrt
import json

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

class Slowpoke:
  
  def __init__(self, plyDepth=4, kingWeight=1.5, weights=[], layers=[91,40,10,1], minimax=False):
    """
    Initialise Agent

    Note that we keep the weights since it is
    essential for the bot to evaluate the board.
    """
    self.chooseMinimax = minimax
    self.nn = False
    self.ply = plyDepth
    self.layers = layers
    self.pieceWeights = {
      "Black" : -1,
      "White" : 1,
      "empty" : 0,
      "blackKing" : -kingWeight,
      "whiteKing" : kingWeight
    }

    # Once we have everything we are ready to initiate
    # the board.
    self.initiateNeuralNetwork(layers, weights)
    self.movesConsidered = []

  def initiateNeuralNetwork(self, layers, weights=[]):
    """
    This function initiates the neural network and adds it
    to the AI class.
    """
    # Now we can initialise the neural network.
    self.nn = NeuralNetwork(layers)
    if weights:
      self.loadWeights(weights)
  
  def loadWeights(self, weights):
    # loads weights to the neural network
    self.nn.loadCoefficents(weights)

  """
  Move Functions (to be called by game.py)
  """

  def move_function(self, board, colour):
    # we'll need to get the current pieces of the board, manipulated 
    # in a way so that it can be shoved into the NN.
    # stage = self.checkGameStage(boardStatus)

    # Now we choose appropiately which outcome we want. From here, we add
    # that current board choice and the probability of it doing damage.
    # chance = chooseGameStage(chances, stage)
    if self.chooseMinimax:
      return self.minimax(board, self.ply, colour)
    else:
      # import mcts
      # k = mcts.MonteCarlo(board)
      # return k.get_play()
      return self.mcts_code(board,self.ply, colour)
      # return self.random_ts(board, self.ply, colour)

  def minimax(self, B, ply, colour):    
    self.counter = 0

    # start with flip being min
    def alphabeta(B,ply,alpha,beta,colour,flip=True):
      self.counter += 1
      if B.is_over():
        if B.winner != minimax_empty:
          if flip:
            return minimax_win
          else:
            return minimax_lose
        else:
          return minimax_draw
      # get moves
      moves = B.get_moves()
      # iterate through moves
      for move in moves:
        HB = B.copy()
        HB.make_move(move)

        if ply == 0:
          score = self.evaluate_board(HB, colour)
        else:
          if flip:
            score = alphabeta(HB, ply-1, alpha, beta, colour, False)
          else:
            score = alphabeta(HB, ply-1, alpha, beta, colour, True)
        if flip:
          if score < beta:
            beta = score
          if beta <= alpha:
            return beta
        else:
          if score > alpha:
            alpha = score
          if alpha >= beta:
            return alpha
      if flip:
        return beta
      else:
        return alpha
        
    # ---------------------------------------------
    moves = B.get_moves()
    best_move = moves[0]
    best_score = float('-inf')

    alpha = float('-inf')
    beta = float('inf')

    # iterate through the current possible moves.
    lol = B.get_move_strings()

    # if theres only one move to make theres no point evaluating future moves.
    if len(moves) == 1:
      # print("only one move")
      # return the only move you can make.
      return moves[0]
    else:
      for i in range(len(moves)):
        HB = B.copy()
        HB.make_move(moves[i])
        if ply == 0:
          score = self.evaluate_board(HB)
        else:
          score = alphabeta(HB,ply-1,alpha,beta,colour,True)
        if score > best_score:
          best_move = moves[i]
          best_score = score
        #   print(lol[i], ":\t\t", score, "!")
        # else:
        #   print(lol[i], ":\t\t", score, )
      # print("moves considered:",self.counter)
      # print(best_move)
      self.movesConsidered.append(self.counter)
      # input("")
      # print(best_move)
      return best_move

  def evaluate_board(self,board,colour):
    """
    We throw in the board into the neural network here, and
    then the neural network evaluates the position of the
    board.

    Make the bot think it's always playing from blacks perspective.
    """

    if board.is_over():
      if board.winner != minimax_empty:
        if board.winner == colour:
          return minimax_win
        else:
          return minimax_lose
      else:
        return minimax_draw
    else:
      # Get the current status of the board.
      boardStatus = board.getBoardPosWeighted(colour, self.pieceWeights)
      if self.layers[0] == 91:
        boardStatus = self.nn.subsquares(boardStatus)
      # Evaluate the board array using our CNN.
      return self.nn.compute(boardStatus)

  def random_ts(self, B, ply, colour):
    moves = B.get_moves()
    best_move = moves[0]
    best_score = float('-inf')

    # if theres only one move to make theres no point evaluating future moves.
    if len(moves) == 1:
      return moves[0]
    else:
      # if the user adds some dud plycount default to 1 random round.
      random_rounds = 1
      # iterate some random amount of times.
      if self.ply > 0:
        random_rounds = 300*self.ply
      
      for i in range(random_rounds):
        random_move = random.choice(moves)
        HB = B.copy()
        HB.make_move(random_move)
        # start mcts
        score = self.treesearch(HB,ply-1,colour)
        # get best score.
        if score > best_score:
          best_score = score
          if best_move != random_move:
            best_move = random_move
      return best_move

  def treesearch(self,B,ply,colour):
    if B.is_over():
      if B.winner != minimax_empty:
        if B.winner == colour:
          return minimax_win
        else:
          return minimax_lose
      else:
        return minimax_draw
    # get moves
    moves = B.get_moves()
    # iterate through a random move
    move = random.choice(moves)
  
    HB = B.copy()
    HB.make_move(move)
    if ply < 1:
      score = self.evaluate_board(HB, colour)
    else:
      score = self.treesearch(HB, ply-1, colour)
    return score

  def mcts_code(self, B, ply, colour):
    moves = B.get_moves()

    # if theres only one move to make theres no point
    # evaluating future moves.
    if not moves:
      return
    if len(moves) == 1:
      return moves[0]

    self.mcts_plays, self.mcts_chances = {}, {}

    seconds = 1
    self.calculation_time = datetime.timedelta(seconds=seconds)

    self.c = 1.4

    # if the user adds some dud plycount default to 1 random round.
    max_rounds = 1
    if self.ply > 0:
      max_rounds = 30*self.ply
    
    # begin mcts
    begin = datetime.datetime.utcnow()
    number_of_sims = 0
    while datetime.datetime.utcnow() - begin < self.calculation_time:
      movebases = self.mcts_simulate(B,ply,colour,max_rounds)
      number_of_sims += 1
  
    print (number_of_sims, datetime.datetime.utcnow() - begin)

    move_states = []
    for i in moves:
      bruh = B.copy()
      bruh = bruh.make_move(i)
      move_states.append((i, hash(bruh.pdn['FEN'])))

    player = B.current_player()

    # Display the stats for each possible play.
    # for x in sorted(
    #   ((100 * self.mcts_chances.get((player, S), 0) /
    #     self.mcts_plays.get((player, S), 1),
    #     self.mcts_chances.get((player, S), 0),
    #     self.mcts_plays.get((player, S), 0), p)
    #    for p, S in move_states),
    #   reverse=True
    # ):
    #   # print ("{3}: {0:.2f}% ({1} / {2})".format(*x))
    #   # print("{3}: ({1} / {2})".format(*x))

    print ("Maximum depth searched:", ply)

    # Pick the move with the highest percentage of winning chances divided by the number of games.
    percent_winchance, best_move = max(
      (self.mcts_chances.get((player, S), 0) /self.mcts_plays.get((player, S), 1),p)
      for p, S in move_states
    )
    print(percent_winchance)

    return best_move

  def mcts_simulate(self,B,ply,colour,rounds):
    """
    For a given hypothetical move, this function simulates how good the game is.
    It returns a percentage of how good it is. If it defaults to a winning game,
    then it'll return with certainty.
    """
    moves = B.get_moves()

    visited_states = set()
    states_copy = [B.copy()]
    state = states_copy[-1]
    player = state.current_player()

    expand = True

    # loop through all the moves
    for t in range(1, rounds+1):
      # print("doing ", t)
      legal_moves = state.get_moves()
      # print(legal_moves)
      # print(state.make_move(8704))
      # print(state.make_move(17408))
      # move_states = [(p, state.make_move(p)) for p in legal_moves]

      move_states = []
      for i in legal_moves:
        bruh = state.copy()
        bruh = bruh.make_move(i)
        move_states.append((i, bruh))

      if all(self.mcts_plays.get((player, S)) for p, S in move_states):
        # if we have the statistics on all of the legal moves here, use them!

        all_move_states = [mcts_plays[(player, S)] for p, S in move_states]
        # print(len(all_move_states))
        if len(all_move_states) > 0:
          log_total = log(sum(all_move_states))
          print(log_total)
          value, move, state = max(
            (
              (
                self.mcts_chances[(player, S)] / self.mcts_plays[(player, S)]
              ) +
              (self.c * sqrt(log_total / self.mcts_plays[(player, S)])), p, S
            )
            for p, S in move_states
          )
        # else:
        #   move = random.choice(legal_moves)
        #   state = states_copy[-1].make_move(move)
      else:
        move = random.choice(legal_moves)
        state = states_copy[-1].make_move(move)
    
      # add current state to list of states
      states_copy.append(state)

      # `player` here and below refers to the player
      # who moved into that particular state.
    
      # print(state.pdn['FEN'])
      su = hash(state.pdn['FEN'])
      if expand and (player, su) not in self.mcts_plays:
        expand = False
        self.mcts_plays[(player, su)] = 0
        self.mcts_chances[(player, su)] = 0
        if t > ply:
          ply = t

      visited_states.add((player, su))
      player = state.current_player()
      if state.is_over():
        winner = state.winner
        break
      else:
        winner = -1

  
      # print("calculating visited states")
      for player, x in visited_states:

        if (player, x) not in self.mcts_plays:
          continue
        # print("countin")

        self.mcts_plays[(player, x)] += 1
        self.mcts_chances[(player, x)] += self.evaluate_board(state, player)



    # # loop through rounds.
    # for i in range(rounds):
    #   # iterate through a random move
    #   move = random.choice(moves)
    #   HB = B.copy()
    #   HB.make_move(move)
    #   if ply < 1:
    #     score = self.evaluate_board(HB, colour)
    #   else:
    #     score = self.mcts_expand(HB, ply-1, colour)
    # return score
  
  # def mcts_expand(self,B, remaining_ply, colour):
  #   if B.is_over():
  #     if B.winner != minimax_empty:
  #       if B.winner == colour:
  #         return minimax_win
  #       else:
  #         return minimax_lose
  #     else:
  #       return minimax_draw
  #   # get moves
  #   moves = B.get_moves()
  #   # iterate through a random move
  #   move = random.choice(moves)
  
  #   HB = B.copy()
  #   HB.make_move(move)
  #   if remaining_ply < 1:
  #     score = self.evaluate_board(HB, colour)
  #   else:
  #     score = self.mcts_simulate(HB, remaining_ply-1, colour)
  #   return score

  # """
  # Checks the current stage of the board.
  # """
  # @staticmethod
  # def checkGameStage(board):
  #   """
  #   There are three stages of Checkers:
  #     Beginning: Both players have at least three pieces on the board. No kings.
  #     Kings: Both players have at least three pieces on the board. At least one king.
  #     Ending: one player has less than three pieces on the board.

  #   Returns:
  #     A value that is either 0,1, or 2.
  #     0: beginning
  #     1: kings
  #     2: ending
  #   """
  #   b = 0
  #   w = 0
  #   kb = 0
  #   kw = 0
  #   for i in range(len(board)):
  #     if pieceWeights['empty'] != board[i]:
  #       if board[i] == pieceWeights['Black']:
  #         b += 1
  #       elif board[i] == pieceWeights['White']:
  #         w += 1
  #       elif board[i] == pieceWeights['blackKing']:
  #         kb += 1
  #       elif board[i] == pieceWeights['whiteKing']:
  #         kw += 1
  #   if b >= 3 and w >= 3:
  #     # we're either at Kings or Beginning.
  #     if kb > 1 or kw > 1:
  #       # we're at intermediate.
  #       return 1
  #     else:
  #       # we're at beginning.
  #       return 0
  #   else:
  #     # we've reached the ending.
  #     return 2

  # @staticmethod
  # def chooseGameStage(nn_results, stage):
  #   if stage == 0:
  #     return nn_results[0]
  #   elif stage == 1:
  #     return nn_results[1]
  #   else:
  #     return nn_results[2]

