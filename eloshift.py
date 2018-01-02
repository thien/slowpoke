def ELOShift(self, winner, black, white):
    b_exp = elo.expected(black.elo, white.elo)
    w_exp = elo.expected(white.elo, black.elo)
    # initiate score outcomes
    b_result = 0
    w_result = 0
    if winner == Black:
      # black wins
      b_result = 1
      pass
    elif winner == White:
      # white wins
      w_result = 1
    else:
      # draw
      b_result = 0.5
      w_result = 0.5
    # calculate elo outcomes
    black.elo = elo.elo(black.elo, b_exp, b_result, k=32)
    white.elo = elo.elo(white.elo, w_exp, w_result, k=32)
    return black, white