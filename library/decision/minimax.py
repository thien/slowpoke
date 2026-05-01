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

class MiniMax:

  def __init__(self, ply, evaluator):
    self.ply = ply
    self.evaluator = evaluator

  def Decide(self, B, colour):
    return self.minimax(B, colour)

# -------------------------------------------------------

  def alphabeta(self, B, ply, alpha, beta, colour, maximizing=True):
    self.counter += 1
    if B.is_over():
      if B.winner != minimax_empty:
        if maximizing:
          return minimax_lose  # We lost (opponent's win)
        else:
          return minimax_win   # We won (opponent's loss)
      else:
        return minimax_draw
    # get moves
    moves = B.get_moves()
    # iterate through moves using push/pop instead of copy
    for move in moves:
      B.push_move(move)

      if ply == 0:
        score = self.evaluator(B, colour)
      else:
        # Toggle maximizing between moves
        score = self.alphabeta(B, ply-1, alpha, beta, B.current_player(), not maximizing)
      B.pop_move()
      
      if maximizing:
        if score > alpha:
          alpha = score
        if alpha >= beta:
          return alpha
      else:
        if score < beta:
          beta = score
        if beta <= alpha:
          return beta
    return alpha if maximizing else beta
      
  def minimax(self, B, colour):    
    self.counter = 0
    self.movesConsidered = []
    
    moves = B.get_moves()
    best_move = moves[0]
    best_score = float('-inf')

    alpha = float('-inf')
    beta = float('inf')

    # if theres only one move to make theres no point evaluating future moves.
    if len(moves) == 1:
      return moves[0]
    else:
      for move in moves:
        B.push_move(move)
        if self.ply == 0:
          score = self.evaluator(B, colour)
        else:
          score = self.alphabeta(B, self.ply-1, alpha, beta, B.current_player(), False)
        B.pop_move()
        if score > best_score:
          best_move = move
          best_score = score
      self.movesConsidered.append(self.counter)
      return best_move