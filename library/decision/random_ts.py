import random
from decision.minimax import minimax_win, minimax_lose, minimax_draw, minimax_empty

class RandomTS:

    def __init__(self, ply, evaluator):
        self.ply = ply
        self.evaluator = evaluator

    def Decide(self, B, colour):
        return self._random_ts(B, self.ply, colour)

# -------------------------------------------------------

    def _random_ts(self, B, ply, colour):
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
                B.push_move(random_move)
                # start mcts
                score = self._treesearch(B, ply-1, colour)
                B.pop_move()
                # get best score.
                if score > best_score:
                    best_score = score
                    if best_move != random_move:
                        best_move = random_move
            return best_move

    def _treesearch(self, B, ply, colour):
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
        
        B.push_move(move)
        if ply < 1:
            score = self.evaluator.evaluate_board(B, colour)
        else:
            score = self._treesearch(B, ply-1, colour)
        B.pop_move()
        return score