from __future__ import annotations

"""
This program evaluates the performance of the simulator.
"""

import os
import json
import numpy as np
import multiprocessing

from slowpoke import play as p
import slowpoke.agents.agent as agent


class Evaluate:
    def __init__(self, date: str, ply: int, defaultResultsPath: str = None) -> None:
        self.date = date
        self.path = os.path.join("..", "results")
        if defaultResultsPath:
            self.path = defaultResultsPath
        self.directory = os.path.join(self.path, self.date)
        self.champ_folder_name = "champions"
        # container for the agents
        self.agents = {}
        # container for statistics
        self.statistics = {}
        # used as parameters for the sims.
        self.game_opts = {
            "show_dialog": False,
            "show_board": False,
            "human_white": False,
            "human_black": False,
            "preload_moves": [],
        }
        # cpu information
        self.cores = multiprocessing.cpu_count()
        # simulation specific information
        self.number_of_games = 10
        if self.cores > 64:
            self.number_of_games = 128

        self.ply = ply
        # split to every nth parttioned player.
        self.choice_range = 6
        # file save information
        self.filename = "gm_stats"

    def load_champions(self, extensions: bool = True) -> list:
        champsPath = os.path.join(self.directory, self.champ_folder_name)
        print("Loading Agents.. ", end="")
        if not os.path.isdir(champsPath):
            print(f"No champions directory at {champsPath}")
            return []
        files = os.listdir(champsPath)
        items = sorted([int(x.split(".json")[0]) for x in files])
        # get the id of the best agent and the worst.
        gmID = items[-1]
        agentCount = len(items)

        tests = []
        if self.choice_range > 0:
            tests.append(items[0])
        for i in range(self.choice_range - 1):
            tests.append(int(agentCount * (i + 1) / self.choice_range))

        self.gm_id = gmID
        # load the gold master agent.
        self.agents["gm"] = self.load_agent_file(self.ply, gmID, champsPath)

        # load the opponments
        for i in tests:
            # create agentString
            agent_ID = "gen-" + str(i)
            # load that agent
            self.agents[agent_ID] = self.load_agent_file(self.ply, i, champsPath)

        if extensions:
            self.load_other_agents()

        # now, we're done!
        print("Done.")
        print("There are", len(self.agents.keys()), "loaded.")

    def load_other_agents(self) -> list:
        # let's import other agents too
        self.agents["random"] = p.load_player_class("magikarp")[0]
        self.agents["pure_mcts"] = p.load_player_class("geodude")[0]

    def create_games(self) -> list:
        # we'll make a list of games that the GM will play against.
        games = []
        for agent in self.agents.keys():
            if agent != "gm":
                games.append(["gm", agent])
        return games

    def evaluate(self, games: list) -> None:
        ent = {}
        for x in games:
            # set colour codes
            black, white = 0, 1
            ev_ID = x[black] + "_vs_" + x[white]
            print("Calculating", ev_ID)
            # create score container
            ent[ev_ID] = {
                "player": x[black],
                "opp": x[white],
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "as_black": {"wins": 0, "losses": 0, "draws": 0},
                "as_white": {"wins": 0, "losses": 0, "draws": 0},
            }

            # iterate through games as black and white
            for j in range(0, 2):
                scores = []
                # make sure they switch for black and white
                gID = "as_black"
                if j == 1:
                    black, white = 1, 0
                    gID = "as_white"
                # initialise entry for the game
                ent[ev_ID][gID] = {}

                gamePool = []
                for i in range(0, int(self.number_of_games / 2)):
                    # add game to list of games to play
                    gamePool.append(
                        {
                            "black": x[black],
                            "white": x[white],
                            "gameOpt": self.game_opts,
                        }
                    )

                # create game pool.
                with multiprocessing.Pool(processes=self.cores) as pool:
                    scores = pool.map(self.game_worker, gamePool)

                # now that's done, we can now tally up the stats
                # count number of wins, losses, draws for given side.
                ent[ev_ID][gID]["wins"] = scores.count(black)
                ent[ev_ID][gID]["losses"] = scores.count(white)
                ent[ev_ID][gID]["draws"] = scores.count(-1)

                # add to overall w/l/d
                ent[ev_ID]["wins"] += ent[ev_ID][gID]["wins"]
                ent[ev_ID]["losses"] += ent[ev_ID][gID]["losses"]
                ent[ev_ID]["draws"] += ent[ev_ID][gID]["draws"]
                # now we need to add this to our results file.
                self.save_results_to_json(ent)
            print(ev_ID, ent[ev_ID])
        return ent

    """
  easy command to save to json.
  """

    def save_results_to_json(self, ent: dict) -> None:
        filename = self.filename + ".json"
        filepath = os.path.join(self.directory, filename)
        with open(filepath, "w") as outfile:
            json.dump(ent, outfile)

    """
  This gets called by the map (as part of multithread)
  """

    def game_worker(self, i: int) -> dict:
        black = self.init_agent_class(i["black"], self.agents[i["black"]])
        white = self.init_agent_class(i["white"], self.agents[i["white"]])
        return p.run_game(black, white, i["gameOpt"]).winner

    @staticmethod
    def load_agent_file(ply: int, pID: int, location: str) -> object:
        # returns an numpy array
        filetype = ".json"
        filename = str(pID) + filetype
        filepath = os.path.join(location, filename)
        # print("loading", filepath)
        data = json.load(open(filepath))[str(pID)]["coefficents"]
        coefs = np.array(data)
        # generate a class and return that
        return p.gen_slowpoke_class(plyCount=ply, weights=coefs)

    @staticmethod
    def init_agent_class(id: str, bot) -> object:
        return (agent.Agent(bot), id)


if __name__ == "__main__":
    date = "2018-03-21 18:41:51"
    ply = 3
    s = Evaluate(date, ply)
    s.load_champions()
    games = s.create_games()
    s.evaluate(games)
