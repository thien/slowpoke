
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

    def alphabeta(self, B,ply,alpha,beta,colour,flip=True):
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
          score = self.evaluator(HB, colour)
        else:
          if flip:
            score = self.alphabeta(HB, ply-1, alpha, beta, colour, False)
          else:
            score = self.alphabeta(HB, ply-1, alpha, beta, colour, True)
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
        
    def minimax(self, B, colour):    
      self.counter = 0
      self.movesConsidered = []
      # start with flip being min
      
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
          if self.ply == 0:
            score = self.evaluator(HB, colour)
          else:
            score = self.alphabeta(HB,self.ply-1,alpha,beta,colour,True)
          if score > best_score:
            best_move = moves[i]
            best_score = score
          #   print(lol[i], ":\t\t", score, "!")
          # else:
          #   print(lol[i], ":\t\t", score, )
        # print("moves considered:",self.counter)
        # print(best_move)
        self.movesConsidered.append(self.counter)
        return best_move