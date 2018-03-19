

# We arbitrarily defined the value of a winning board as +1.0 and a losing board as −1.0. All other boards would receive values between −1.0 and +1.0, with a neural network favoring boards with higher values.

minimax_win = 1
minimax_lose = -minimax_win
minimax_draw = 0
minimax_empty = -1

import random


class TMCTS:

    def __init__(self, ply, evaluator):
        self.ply = ply
        self.evaluator = evaluator

    def Decide(self, B, colour):
        self.movesets = {}
        return self.random_ts(B, self.ply, colour)

    # -------------------------------------------------------

    def random_ts(self, B, ply, colour, debug=False):
        # print("YOU ARE", colour)
        # colour = 1 if colour == 0 else 0
        moves = B.get_moves()

        # if theres only one move to make theres no point evaluating future moves.
        if len(moves) == 1:
            return moves[0]
        else:
            # if the user adds some dud plycount default to 1 random round.
            random_rounds = 1
            # iterate some random amount of times.
            if self.ply > 0:
                random_rounds = 300*self.ply
            
            # set up moves
            for move in moves:
                self.movesets[move] = {
                    'plays' : 0,
                    'chances' : 0
                }
            
            # iterate through the random number of rounds
            for i in range(random_rounds):
                random_move = random.choice(moves)
                HB = B.copy()
                HB.make_move(random_move)
                # start mcts
                self.movesets[random_move]['chances'] += self.treesearch(HB,ply,colour)
                self.movesets[random_move]['plays'] += 1

            bestChance = -1000
            bestMove = moves[0]
            for m in self.movesets:
                moveIndex = B.get_moves().index(m)
                moveString =  B.get_move_strings()[moveIndex]
                # calculate the chance of it winning
                chance = self.movesets[m]['chances'] / self.movesets[m]['plays']
                if chance > bestChance:
                    bestChance = chance
                    bestMove = m
                    if debug:
                        print(moveString, chance,  self.movesets[m]['chances'] ,self.movesets[m]['plays'],  "*")
                else:
                    if debug:
                        print(moveString, chance)
            
            return bestMove

    def treesearch(self,B,ply,colour):
        # enemyColour = 1 if colour == 0 else 0
        isOver = self.isOver(B, colour)
        if isOver[0]:
            return isOver[1]
        else:
            if ply < 1:
                return self.evaluator(B, colour)
            else:
                # get moves
                moves = B.get_moves()
                # choose random enemy move
                move = random.choice(moves)
                HB = B.copy()
                HB.make_move(move)
                # check if that move ended the game
                isOver = self.isOver(HB, colour)
                if isOver[0]:
                    return isOver[1]
                else:
                    # get moves
                    moves = HB.get_moves()
                    if len(moves) > 0:
                        # choose random player move
                        move = random.choice(moves)
                        HB.make_move(move)
                        # traverse, moving down the player ply
                        return self.treesearch(HB, ply-1, colour)
                    else:
                        return 0

    def isOver(self,B, colour):
        if B.is_over():
            if B.winner != minimax_empty:
                if B.winner == colour:
                    return (True,minimax_win)
                else:
                    return (True,minimax_lose)
            else:
                return (True,minimax_draw)
        else:
            return (False,-1)
