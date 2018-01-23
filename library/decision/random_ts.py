class RandomTS:

    def __init__(self, ply, evaluator):
        self.ply = ply
        self.evaluator = evaluator

    def Decide(self, B, colour):
        return self.random_ts(B, self.ply, colour)

# -------------------------------------------------------

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