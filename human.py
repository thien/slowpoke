
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

# -----------------------------------------------------------

class Human:
  
  """
  Initialisation Functions
  """

  def __init__(self):
    """
    Initialise Agent

    Note that we keep the weights since it is
    essential for the bot to evaluate the board.
    """

    self.null = 0

  def printStatus(self, B):
    print('\033c', end=None)
    print("--------")
    print (B)
    print(B.pdn)
    print(B.AIBoardPos)
    # print(B.moves())
    print("--------")

  def move_function(self, B, colour=None):
    legal_moves = B.get_moves()
    if B.jump:
      print ("Make jump.")
      print ("")
    else:
      print ("Turn %i" % B.turnCount)
      print ("")
    for (i, move) in enumerate(B.get_move_strings()):
      print ("Move " + str(i) + ": " + move)
    while True:
      move_idx = input("Enter your move number: ")
      try:
        move_idx = int(move_idx)
      except ValueError:
        print ("Please input a valid move number.")
        continue
      if move_idx in range(len(legal_moves)):
        break
      else:
        print ("Please input a valid move number.")
        continue

    return legal_moves[move_idx]
