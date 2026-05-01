import datetime
import random
import math

class MCTS:

  def __init__(self, ply, evaluator=None, debug=False):
    self.ply = ply
    self.evaluator = evaluator
    self.c = 1.4
    self.debug = debug
    
  def Decide(self, B, colour):
    return self.mcts_code(B, self.ply, colour)

  def mcts_code(self, B, ply, colour):
    if self.debug:
      print("I AM:", colour)
    # get the current set of moves
    moves = B.get_moves()
    # if theres only one move to make theres no point
    # evaluating future moves.
    if not moves:
      return
    if len(moves) == 1:
      return moves[0]

    self.mcts_plays, self.mcts_chances = {}, {}
    seconds = self.ply
    self.calculation_time = datetime.timedelta(seconds=seconds)

    # if the user adds some dud plycount default to 1 random round.
    max_rounds = 1
    if self.ply > 0:
      max_rounds = 200*self.ply

    # begin mcts
    begin = datetime.datetime.utcnow()
    number_of_sims = 0
    ply = 100*self.ply
    while datetime.datetime.utcnow() - begin < self.calculation_time:
      self.mcts_simulate(B, ply, colour, max_rounds)
      number_of_sims += 1
      if self.debug:
        print("sims:", number_of_sims, "\r", end="")
    if self.debug:
      print("sims:", number_of_sims)

    # Use push/pop pattern for move evaluation
    move_states = []
    for i in moves:
      B.push_move(i)
      FEN_hash = hash(B.pdn['FEN'])
      move_states.append((i, FEN_hash))
      B.pop_move()

    # Pick the move with the highest percentage of winning chances divided by the number of games.
    percent_winchance, best_move = max(
      (self.mcts_chances.get((colour, S), 0) / self.mcts_plays.get((colour, S), 1), p)
      for p, S in move_states
    )

    if colour == 1:
      percent_winchance, best_move = min(
      (self.mcts_chances.get((colour, S), 0) / self.mcts_plays.get((colour, S), 1),p)
      for p, S in move_states
    )

    if self.debug:
      # Display the stats for each possible play.
      goods = sorted(
        ((100 * self.mcts_chances.get((colour, S), 0) /
          self.mcts_plays.get((colour, S), 1),
          self.mcts_chances.get((colour, S), 0),
          self.mcts_plays.get((colour, S), 0), p)
         for p, S in move_states),
        reverse=True
      )

      for i in goods:
        print(i[3], "Moves:",i[2], "Good Moves",i[1], str(i[0]) + "%")

      print ("Maximum depth searched:", ply)
      print(percent_winchance)

    return best_move

  def mcts_simulate(self, B, ply, colour, rounds):
    """For a given hypothetical move, this function simulates how good the game is.
    It returns a percentage of how good it is. If it defaults to a winning game,
    then it'll return with certainty.
    Uses push/pop pattern - simulates on original B and restores state afterward.
    """
    visited_states = set()
    player = colour
    move_stack = []

    expand = True
    winner = -1
    current_ply = ply

    # loop through all the moves
    for t in range(1, rounds+1):
      legal_moves = B.get_moves()

      if not legal_moves:
        break

      # Generate move states using push/pop instead of copy
      move_states = []
      for i in legal_moves:
        B.push_move(i)
        FEN_hash = hash(B.pdn['FEN'])
        move_states.append((i, FEN_hash))
        B.pop_move()

      # Check if all moves have been explored for UCB1 selection
      if all(self.mcts_plays.get((player, S)) for p, S in move_states):
        all_move_states = [self.mcts_plays[(player, S)] for p, S in move_states]
        log_total = math.log(sum(all_move_states))
        value, move, FEN_hash = max(
          (
            self.mcts_chances[(player, S)] / self.mcts_plays[(player, S)]
            + self.c * math.sqrt(log_total / self.mcts_plays[(player, S)]), p, S
          )
          for p, S in move_states
        )
        B.push_move(move)
        move_stack.append(move)
      else:
        # Random selection for unexplored moves
        choice = random.choice(move_states)
        move, FEN_hash = choice
        B.push_move(move)
        move_stack.append(move)
  
      if B.is_over():
        winner = B.winner
        break

      # Track this state
      su = FEN_hash
      if expand and (player, su) not in self.mcts_plays:
        expand = False
        self.mcts_plays[(player, su)] = 0
        self.mcts_chances[(player, su)] = 0
        if t > current_ply:
          current_ply = t

      visited_states.add((player, su))
      player = B.current_player()
      if B.is_over():
        winner = B.winner

    # Record stats for all visited states
    for p, x in visited_states:
      if (p, x) not in self.mcts_plays:
        continue
      self.mcts_plays[(p, x)] += 1
      if p == winner:
        self.mcts_chances[(p, x)] += 1

    # Undo all moves to restore original state
    for _ in range(len(move_stack)):
      B.pop_move()
    
    return