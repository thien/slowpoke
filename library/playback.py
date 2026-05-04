from __future__ import annotations

import math
import os
import time
import core.checkers as checkers
from termcolor import colored


class Playback:
    def __init__(self, defaultResultsPath: str = None) -> None:
        self.games = {}
        self.debug = True
        self.load_gm = True
        # display options
        self.time_delay = 0.5
        self.load_how_many_games = 8

    def load_champ_games(self, genObj: dict, champ: dict) -> list:
        games = genObj["games"]
        if self.debug:
            print("Finding games played by Player", champ)

        gameMoveDistribution = {}
        champGames = []
        for i in games:
            champFound = False
            if i["game"]["Black"] not in gameMoveDistribution:
                gameMoveDistribution[i["game"]["Black"]] = 1
            else:
                gameMoveDistribution[i["game"]["Black"]] += 1
            if i["game"]["White"] not in gameMoveDistribution:
                gameMoveDistribution[i["game"]["White"]] = 1
            else:
                gameMoveDistribution[i["game"]["White"]] += 1
            if int(i["game"]["Black"]) == int(champ):
                champFound = True
            if int(i["game"]["White"]) == int(champ):
                champFound = True
            if champFound:
                champGames.append(i["game"])

        if self.debug:
            print("Retrieved", len(champGames), "games.")
            print("Here's a game move distribution.")
            print(gameMoveDistribution)
        return champGames

    def load_generation(self, generation: int = 0) -> dict:
        if self.debug:
            print("Loaded Generation", generation)
        return self.statistics[generation]

    def process_leaderboard(self, lb: list) -> list:
        l = None
        # we know the leaderboard is the ones containing Player
        for i in lb:
            if "\nPlayer" in i[0]:
                l = i[0]
        # remove cruft
        l = l.replace("\n", "")
        l = l.split("Player ")
        l.pop(0)
        for i in range(len(l)):
            l[i] = l[i].split("\t")
        if self.debug:
            print("Processed leaderboard:")
            for i in l:
                print("Player", i[0], "\t", i[1])
            print("The Champion is: Player", self.get_champion(l))
        return l

    def run_replay_programme(self, gameMoves: list, champ=None, gen=None) -> None:
        checkers.CheckerBoard()
        # get some of the games.
        games = gameMoves[: self.load_how_many_games]
        # keep track of whether all games have finished
        max_game_length = 0
        # add checkerboards into each object
        for i in range(len(games)):
            games[i]["Board"] = checkers.CheckerBoard()
            # find the max game length
            gameLength = len(games[i]["replay"])
            if max_game_length < gameLength:
                max_game_length = gameLength

        title = "GENERATION " + str(gen + 1)
        lineSeparator = " "
        # this is for a specific game.
        # someGame = gameMoves[0]
        # # now we need to create the gameboards that we play
        # # the games on
        # replays = someGame['replay']
        # for i in replays:
        # 	# get the index of the move we're going to use
        # 	moveIndex = B.get_move_strings().index(i[1])
        # 	move = B.get_moves()[moveIndex]
        # 	# make move
        # 	B.make_move(move)
        # 	print(B)
        # 	time.sleep(self.time_delay)
        # 	#

        # this is for a specific game.
        for i in range(max_game_length):
            print("\033c", end="")
            # create a list of boards
            board_display = []
            # iterate through each game and make a move.
            for game in range(len(games)):
                # check if the game has finished already
                replayLen = len(games[game]["replay"])
                if replayLen > i:
                    replayMove = games[game]["replay"][i][1]
                    # get the index of the move we're going to use
                    moveIndex = (
                        games[game]["Board"].get_move_strings().index(replayMove)
                    )
                    move = games[game]["Board"].get_moves()[moveIndex]
                    # make move
                    games[game]["Board"].make_move(move)
                # add the board to the list of boards
                gameBoard = games[game]["Board"].generate_ascii_board()
                gameBoard.insert(0, self.add_static_board_info(games[game], i, champ))
                board_display.append(gameBoard)
            # compile the boards together.
            boardPrints = self.process_board_displays(board_display)
            # Print!
            print("\n" + title, "\n" + lineSeparator, "\n" + boardPrints)
            time.sleep(self.time_delay)
        return True

    def process_board_displays(self, board_display) -> None:
        # calculate terminal row, column
        _, TColumns = os.popen("stty size", "r").read().split()
        # print("CLI Rows:", TRows, "CLI Columns:",TColumns)

        # calculate size of individual board
        someBoard = board_display[0]

        BRows = len(someBoard)
        BColumns = len("".join(map(lambda x: "".join(x), someBoard[1])))
        # print("Board Rows:", BRows, "Board Columns:",BColumns)

        # calculate number of horizontal boards
        horizontal_board_count = math.floor(int(TColumns) / BColumns)
        # split the board into chunks
        chunks = [
            board_display[x : x + horizontal_board_count]
            for x in range(0, len(board_display), horizontal_board_count)
        ]

        # concatenate boards together.
        master = []
        for i in chunks:
            # mega_chunk = []
            for row in range(BRows):
                mega_row = []
                for board in i:
                    board_row = board[row]
                    # check if it's the last board
                    # if its not, remove the \n at the end.
                    if i.index(board) + 1 != len(i):
                        board_row.pop()
                        board_row.append("   ")
                    mega_row = mega_row + board_row
                master.append(mega_row)
            master.append(["\n"])
        return self.print_board(master)

    @staticmethod
    def add_static_board_info(
        gameInfo: dict, replayLen: int, champID: str = None
    ) -> dict:
        black_player = "%04d" % (int("000") + gameInfo["Black"])
        black_player = "P" + black_player
        if gameInfo["Black"] == int(champID):
            black_player = colored(black_player, "yellow")
        black_player = colored("B: ", "red") + black_player
        White_player = "%04d" % (int("000") + gameInfo["White"])
        White_player = "P" + White_player
        if gameInfo["White"] == int(champID):
            White_player = colored(White_player, "yellow")
        White_player = colored("W: ", "cyan") + White_player
        gameCount = replayLen + 1
        if replayLen > len(gameInfo["replay"]):
            gameCount = len(gameInfo["replay"])
        gameLength = "Move: " + "%03d" % (int("000") + gameCount)

        gameStatus = "D"
        # print(gameInfo['Winner'])
        if gameInfo["Winner"] == 0:
            if gameInfo["Black"] == int(champID):
                gameStatus = "W"
            else:
                gameStatus = "L"
        elif gameInfo["Winner"] == 1:
            if gameInfo["White"] == int(champID):
                gameStatus = "W"
            else:
                gameStatus = "L"

        if gameStatus == "W":
            gameStatus = colored(gameStatus, "green")
        elif gameStatus == "L":
            gameStatus = colored(gameStatus, "magenta")

        # inp = "abcdefghijklmnopqrstuvwxyzabcdefghijk"
        inp = (
            black_player + "  " + White_player + "  " + gameStatus + "   " + gameLength
        )
        # inp = str(inp)[:boardCol]
        inp = list(inp)

        inp.append("\n")
        return inp

    @staticmethod
    def print_board(board) -> None:
        return "".join(map(lambda x: "".join(x), board))

    @staticmethod
    def get_champion(lb: list) -> dict:
        return lb[0][0]

    @staticmethod
    def choose_generation() -> int:
        chosenAnswer = False
        option = False

        while not chosenAnswer:
            print("Would you like to choose which generation to view?")
            answer = input("Y/N:")
            if answer in "ynYN":
                chosenAnswer = True
                if answer in "yY":
                    option = True
            else:
                print("You chose an invalid option, please try again.")
                print("---")

        if option:
            print("Which generation would you like to choose?")
            answer = None


# if __name__ == '__main__':
# 	foldername = "2018-03-20 01:48:19 (6ply 200 generations)"
# 	s = Playback(foldername)
# 	s.load_statistics_file()

# 	for gen_id in range(0,5):
# 		generation = s.load_generation(gen_id)
# 		leaderboard = s.process_leaderboard(generation['stats'])
# 		champ = s.get_champion(leaderboard)
# 		# find games played by the champ
# 		gameMoves = s.load_champ_games(generation,champ)
# 		s.run_replay_programme(gameMoves,champ=champ,gen=gen_id)
# 		# s.load_champ_games(1)

if __name__ == "__main__":
    print("--- PLAYBACK ---")

    import agentLoader as al

    al = al.agentLoader()
    system = al.load_agent_ui()
    print("Loading Statistics file.. ", end="")

    statistics = al.load_statistics_file(system)
    print("Done.")
    s = Playback()
    s.statistics = statistics

    for gen_id in range(0, 5):
        generation = s.load_generation(gen_id)
        leaderboard = s.process_leaderboard(generation["stats"])
        champ = s.get_champion(leaderboard)
        # find games played by the champ
        gameMoves = s.load_champ_games(generation, champ)
        s.run_replay_programme(gameMoves, champ=champ, gen=gen_id)
        # s.load_champ_games(1)
