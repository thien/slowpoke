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

# import decision files
import sys

sys.path.insert(0, "..")
import decision.mcts as mcts
from agents.bot import Bot


class Geodude(Bot):
    def __init__(self, plyDepth=4):
        """
        Initialise Agent

        Note that we keep the weights since it is
        essential for the bot to evaluate the board.
        """
        self.ply = plyDepth
        self.decisionFunction = mcts.MCTS(self.ply)

    def move_function(self, board, colour):
        # return self.mcts_code(board,self.ply, colour)
        return self.decisionFunction.Decide(board, colour)
