from __future__ import annotations

import core.population as pop
import core.game as game
import core.storage as storage

import agents.slowpoke as sp
import agents.agent as agent
import core.mongo as mongo

# import libraries
import datetime
import json
import logging
import multiprocessing
import os
import random

import numpy as np
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table

# ignore runtime warnings
import warnings

warnings.filterwarnings("ignore")

import statistics
from core.constants import BLACK, WHITE, EMPTY, WIN_PT, DRAW_PT, LOSE_PT

# Champ points
CHAMP_WIN_PT, CHAMP_DRAW_PT, CHAMP_LOSE_PT = 1, 0, -1


def option_defaults(options):
    # adds default options if they are absent from options.
    defaultOptions = {
        "debugMode": False,
        "mongoConfigPath": "config.json",
        "ply_depth": 4,
        "NumberOfGenerations": 200,
        "Population": 15,
        "printStatus": True,
        "connectMongo": False,
        "resultsLocation": os.path.join("..", "results"),
        "use_parallel_mcts": None,  # None = auto (True when ply_depth > 1)
        "num_parallel": 4,  # Number of parallel threads for MCTS
    }
    for i in defaultOptions.keys():
        if i not in options:
            options[i] = defaultOptions[i]
    return options


class Generator:
    def __init__(self, options):
        # initialise default variables when needed.
        options = option_defaults(options)
        self.is_debugMode = options["debugMode"]
        # Declare base information
        self.ply_depth = options["ply_depth"]
        self.generations = options["NumberOfGenerations"]
        self.populationSize = options["Population"]  # number of players
        # generate the initial population.
        self.population = pop.Population(
            self.populationSize,
            self.ply_depth,
            self.is_debugMode,
            options["use_parallel_mcts"],
            options["num_parallel"],
            include_onix=True,
        )
        # time handlers
        self.StartTime = datetime.datetime.now().timestamp()
        self.AverageGameTime = 0
        self.AverageGenrationLength = 0
        self.RemainingTime = 0
        self.EstDateFinished = 0
        self.GenerationTimeLengths = np.array([])
        self.currentGenStartTime = datetime.datetime.now().timestamp()
        # current generation game counts
        self.GamesFinished = 0
        self.GamesQueued = 0
        self.CurrentGeneration = 0
        # champions
        self.AreChampionsPlaying = False
        self.LastChampionScore = 0
        self.cummulativeScore = 0
        self.AverageChampionGrowth = 0
        self.RecentChampionScores = 0
        self.playPreviousChampCount = 5
        self.champGamesRoundsCount = 6  # should always be even and at least 2.
        self.progress = []

        self.previousGenerationRankings = None
        self.previousChampPointList = None
        # Initiate other information
        self.processors = multiprocessing.cpu_count() - 1
        self.config = self.load_json_config(options["mongoConfigPath"])
        self.mongoConnected = options["connectMongo"]
        self.totalGamesPerGen = (options["Population"] ^ 2) - options["Population"]

        # placeholder values
        self.gameIDCounter = 0
        # once we have the config file we can proceed and initiate our MongoDB connection.
        self.init_mongo_connection()
        # we also want to save the stats offline
        self.generationStats = []
        self.folderName = (
            str(self.clean_date(self.StartTime, True))
            + " "
            + str(self.ply_depth)
            + "ply"
        )
        self.saveLocation = os.path.join(options["resultsLocation"], self.folderName)
        self.options = options  # Store for reference
        # Set up logging
        self.log_file = os.path.join(self.saveLocation, "training.log")
        self._setup_logging()
        # self.saveLocation = os.path.join(options['resultsLocation'],self.clean_date(self.StartTime, True))
        # generate charts as we go?
        self.generateChartsEveryRound = True

    def load_json_config(self, filepath: str) -> dict:
        """
        Loads config.json
        """
        try:
            with open(filepath) as json_file:
                data = json.load(json_file)
            return data
        except:
            data = {"MongoURI": ""}
            return data

    def init_mongo_connection(self) -> None:
        self.db = mongo.Mongo()
        try:
            if self.mongoConnected:
                self.db.initiate(self.config["MongoURI"])
        except:
            pass

    def _setup_logging(self) -> None:
        """Set up logging to both file and console."""
        # Ensure save directory exists
        if not os.path.isdir(self.saveLocation):
            os.makedirs(self.saveLocation)

        # Configure logging to file
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger(__name__)
        self.log("Training started")
        self.log(
            f"Population: {self.populationSize}, Ply Depth: {self.ply_depth}, Generations: {self.generations}"
        )

    def log(self, message: str) -> None:
        """Write a message to the log file."""
        if hasattr(self, "logger"):
            self.logger.info(message)

    def log_status_info(self) -> None:
        """Write current status info to log file."""
        for i in self.status_info():
            self.log(f"{i[0]}: {i[1]}")

    def Tournament(self) -> None:
        """
        Tournament; full round-robin where every pair plays both colours.
        Returns the players in order of how good they are.
        """
        self.log("=" * 60)
        self.log("STARTING TOURNAMENT")
        self.log(f"Generation: {self.currentGeneration}")
        self.log(f"Population size: {len(self.population.current_population)} players")

        # Full round-robin: each pair plays each colour exactly once
        gamePool = []
        players = self.population.current_population[:]
        self.log("Scheduling full round-robin games...")
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                pid_i, pid_j = players[i], players[j]
                # i as black, j as white
                gamePool.append({
                    "game_id": self.gameIDCounter,
                    "black": self.population.players[pid_i],
                    "white": self.population.players[pid_j],
                    "dbURI": False,
                    "debugInfo": False,
                })
                self.gameIDCounter += 1
                # j as black, i as white
                gamePool.append({
                    "game_id": self.gameIDCounter,
                    "black": self.population.players[pid_j],
                    "white": self.population.players[pid_i],
                    "dbURI": False,
                    "debugInfo": False,
                })
                self.gameIDCounter += 1
        self.GamesQueued = len(gamePool)
        self.log(f"Total games scheduled: {len(gamePool)}")

        # run game simulations.
        results = []
        # close number of processes when map is done.
        threadCount = self.processors
        if self.processors > len(gamePool):
            threadCount = len(gamePool)
        self.log(f"Running games with {threadCount} parallel processes...")
        with multiprocessing.Pool(processes=threadCount) as pool:
            chunksize = max(1, len(gamePool) // threadCount)
            results = pool.map(self.game_worker, gamePool, chunksize=chunksize)
            pool.close()
            pool.join()
        self.log("All games completed")

        # when the pool is done with processing, process the results.
        self.log("Processing game results...")
        for i in range(len(results)):
            self.population.allocate_points(
                results[i]["game"], results[i]["black"], results[i]["white"]
            )
            # merge winning players move caches
            if results[i]["game"]["Winner"] == Black:
                bCache = self.population.players[results[i]["black"]].bot.cache
                self.population.players[
                    results[i]["black"]
                ].bot.cache = self.merge_dicts(bCache, results[i]["black_cache"])
            elif results[i]["game"]["Winner"] == White:
                wCache = self.population.players[results[i]["white"]].bot.cache
                self.population.players[
                    results[i]["white"]
                ].bot.cache = self.merge_dicts(wCache, results[i]["white_cache"])
            # nullify the cache since its not needed anymore
            results[i]["black_cache"] = None
            results[i]["white_cache"] = None
        self.log("Results processed. Updating player rankings...")

        self.population.sort_population_by_points()
        self.population.add_champion()
        self.log("Tournament complete. Population sorted by Elo.")
        return (self.population, results)

    """ 
  This function is called for every generation.
  """

    def run_generations(self) -> None:
        # loop through the generations.
        for i in range(self.generations):
            print("Initiating generation", i)
            self.log(f"=" * 60)
            self.log(f"Starting generation {i}")
            # increment generation count
            self.currentGeneration = i
            self.currentGenStartTime = datetime.datetime.now().timestamp()
            # reset game count statistics prior to running
            self.GamesFinished = 0
            self.GamesQueued = 0
            # initiate timestamp
            startTime = datetime.datetime.now()
            # make bots play each other.
            self.population, generationResults = self.Tournament()
            self.previousGenerationRankings = (
                self.population.print_population_by_points()
            )

            # compute champion games (runs independently of others)
            self.log("Running champion games...")
            self.run_champions()
            # save champions to file
            self.log("Saving champions to file...")
            self.population.save_champions_to_file(self.saveLocation)
            # save genomic details
            self.log("Saving population genomes...")
            self.population.save_population_genomes(self.saveLocation)
            # get the best players and generate a new population from them.
            self.log("Generating next population...")
            self.population.generate_next_population()
            self.populationSize = self.population.count
            # initiate end timestamp and add time difference length to list.
            timeDifference = (datetime.datetime.now() - startTime).total_seconds()
            self.GenerationTimeLengths = np.hstack(
                (self.GenerationTimeLengths, timeDifference)
            )
            self.log(f"Generation complete ({timeDifference:.1f}s)")
            # need to store the results of this into a json file!
            self.generationStats.append(
                {
                    "stats": [(str(i[0]), str(i[1])) for i in self.status_info()],
                    "games": generationResults,
                    "durationInSeconds": str(timeDifference),
                }
            )
            self.save_training_stats_to_json(self.saveLocation, self.generationStats)
            if self.generateChartsEveryRound:
                self.generate_stats()
            # Display status at generation boundary
            self.display_status_info(force_display=True)

    def nuke_cache(self) -> None:
        for i in self.population:
            self.population[i].bot.cache = {}

    def generate_stats(self) -> None:
        # create statistics
        stats = statistics.Statistics(self.folderName)
        stats.loadStatisticsFile()
        stats.saveCharts()
        print("I made some charts!")

    def save_training_stats_to_json(self, saveLocation: str, stats) -> None:
        # check save directory exists prior to saving
        if not os.path.isdir(saveLocation):
            os.makedirs(saveLocation)

        filename = "statistics.json"
        with open(os.path.join(saveLocation, filename), "w") as outfile:
            json.dump(stats, outfile)

        # Also write Parquet for efficient columnar access
        try:
            storage.save_statistics_parquet(saveLocation, stats)
        except Exception as e:
            self.log(f"Parquet write failed (non-fatal): {e}")

    def pool_champ_game(self, info) -> None:
        blackPlayer = self.population.players[info["Players"][0]]
        whitePlayer = self.population.players[info["Players"][1]]
        results = game.tournament_match(blackPlayer, whitePlayer)
        if results["Winner"] == info["champColour"]:
            # champion won.
            return ChampWIN_PT
        elif results["Winner"] == empty:
            return ChampDRAW_PT
        else:
            return ChampLOSE_PT

    def create_champ_games(self) -> None:
        currentChampID = self.population.champions[-1]
        champGames = []
        gameRound = int(self.champGamesRoundsCount / 2)
        # playback counter
        playcounter = np.size(self.progress)
        if playcounter > self.playPreviousChampCount:
            playcounter = self.playPreviousChampCount

        for i in range(playcounter):
            previousChampID = self.population.champions[-i + 1]
            # set player colours
            info = {"Players": (currentChampID, previousChampID), "champColour": Black}

            for j in range(gameRound):
                champGames.append(info)
            # reverse players
            info = {"Players": (previousChampID, currentChampID), "champColour": White}
            for j in range(gameRound):
                champGames.append(info)
        return champGames

    def run_champions(self) -> None:
        """
        These champion games are called at the end of every generation
        and are used to determine the progress of the bots.
        """
        self.AreChampionsPlaying = True
        self.display_status_info()

        # check if theres more than 5 champions.
        if len(self.population.champions) > 2:
            # create list of games to play
            champGames = self.create_champ_games()
            # close number of processes when map is done.
            results = []

            numberOfChampgames = len(champGames)
            threadCount = self.processors
            if self.processors > numberOfChampgames:
                threadCount = numberOfChampgames

            with multiprocessing.Pool(processes=threadCount) as pool:
                chunksize = max(1, numberOfChampgames // threadCount)
                results = pool.map(
                    self.pool_champ_game, champGames, chunksize=chunksize
                )
                pool.close()
                pool.join()

            # split results into equal segments
            l = results
            n = self.playPreviousChampCount
            results = [l[i : i + n] for i in range(0, len(l), n)]
            # calculate the gradient of the scores.

            medians = []
            for i in results:
                medians.append(np.mean(i))

            self.previousChampPointList = medians

            # compute new champ points compared to previous champ
            newChampPoints = np.mean(medians)
            self.cummulativeScore += newChampPoints
            # store points.
            self.progress.append(newChampPoints)
            self.population.players[
                self.population.champions[-1]
            ].champ_score = newChampPoints
            self.population.players[self.population.champions[-1]].champ_range = results
        else:
            # theres only one champion, don't play.
            self.progress.append(0)
        self.AreChampionsPlaying = False
        self.display_status_info(force_display=True)

    def game_worker(self, i: int) -> dict:
        timeStart = datetime.datetime.now().timestamp()
        results = game.tournament_match(
            i["black"], i["white"], i["game_id"], i["dbURI"], i["debugInfo"]
        )
        bSubset = {}
        wSubset = {}
        # get a subset of the caches
        if i["black"].bot.enable_cache:
            bCache = i["black"].bot.cache
            wCache = i["white"].bot.cache
            # Sample from caches if they have entries
            for _ in range(100):
                if bCache:
                    randb = random.choice(list(bCache.keys()))
                    bSubset[randb] = bCache[randb]
                if wCache:
                    randw = random.choice(list(wCache.keys()))
                    wSubset[randw] = wCache[randw]
            # nuke cache
            i["black"].bot.cache = {}
            i["white"].bot.cache = {}
        data = {
            "game": results,
            "black": i["black"].id,
            "white": i["white"].id,
            "black_cache": bSubset,
            "white_cache": wSubset,
            "duration": str(
                self.clean_date(datetime.datetime.now().timestamp() - timeStart)
            ),
        }
        return data

    def status_info(self) -> dict:
        currentTime = datetime.datetime.now().timestamp()
        recent_scores = self.progress[-7:]

        averageGenTimeLength = np.mean(self.GenerationTimeLengths)

        PercentageEst = 0.0
        if not np.isnan(averageGenTimeLength) and averageGenTimeLength > 0:
            PercentageEst = min(
                (currentTime - self.currentGenStartTime) / averageGenTimeLength,
                1.0,
            )

        numGens = np.size(self.progress)
        remainingGenTime = max(
            0.0,
            averageGenTimeLength - (currentTime - self.currentGenStartTime),
        )
        RemainingGenCount = self.generations - numGens

        # calculate current run time
        currentRunTime = datetime.datetime.now() - datetime.datetime.fromtimestamp(
            self.StartTime
        )
        # calculate remaining time
        EstRemainingTime = (
            (RemainingGenCount * averageGenTimeLength)
            + np.sum(self.GenerationTimeLengths)
            - currentRunTime.total_seconds()
        )

        EstEndDate = EstRemainingTime + self.StartTime + currentRunTime.total_seconds()

        messsages = []

        messsages.append(["Generation", str(numGens) + "/" + str(self.generations)])
        messsages.append(["Population", self.populationSize])
        messsages.append(["Ply Depth", self.ply_depth])
        messsages.append(["Connected To Mongo", self.mongoConnected])
        messsages.append(["Cores Utilised", self.processors])
        # start and end dates
        messsages.append([" ", " "])
        messsages.append(["Test Start Date", self.clean_date(self.StartTime, True)])
        messsages.append(["Current Runtime", currentRunTime])
        messsages.append(["Test End Date*", self.clean_date(EstEndDate, True)])
        messsages.append(["Remaining Test Time*", self.clean_date(EstRemainingTime)])
        # Time info
        messsages.append([" ", " "])
        messsages.append(["Mean Game Time", self.clean_date(averageGenTimeLength)])
        messsages.append(["Gen. Progress*", str(round(PercentageEst * 100, 2)) + "%"])
        messsages.append(["Remaining Gen. Time*", self.clean_date(remainingGenTime)])
        # champion info
        messsages.append([" ", " "])
        messsages.append(["Champions Currently Playing?", self.AreChampionsPlaying])
        messsages.append(["Previous Score", self.LastChampionScore])
        messsages.append(["Cummulative Score", f"{self.cummulativeScore:.2f}"])

        avgRecentScores = 0.0
        if len(recent_scores) > 0:
            avgRecentScores = float(np.mean(recent_scores))
        messsages.append(["Average Growth", f"{avgRecentScores:.2f}"])
        try:
            messsages.append(
                [
                    "Recent Scores",
                    ", ".join("{:0.2f}".format(x) for x in recent_scores),
                ]
            )
            messsages.append(
                [
                    "Prev. Champ Point Range",
                    ["{:0.2f}".format(x) for x in self.previousChampPointList],
                ]
            )
        except:
            pass
        messsages.append([" ", " "])
        messsages.append(["Previous Scoreboard", " "])
        messsages.append([self.previousGenerationRankings, ""])
        messsages.append(["", ""])
        messsages.append(["Debug Mode:", self.is_debugMode])

        return messsages

    def display_status_info(self, force_display: bool = False) -> None:
        """Log status info to file. Display rich panels to console."""
        self.log_status_info()

        console = Console()
        layout = Layout()
        layout.split_column(
            Layout(name="info"),
            Layout(name="matrix"),
            Layout(name="ranking"),
        )

        # ── Info panel: single-line metrics ──
        info = Table.grid(padding=(1, 2))
        info.add_column("Metric", style="cyan", no_wrap=True)
        info.add_column("Value", style="white")
        ranking_lines = None
        for metric, value in self.status_info():
            metric_s = str(metric) if metric else ""
            value_s = str(value) if value else ""
            # Skip spacers (single-space metric)
            if metric_s.strip() == "" and value_s.strip() == "":
                continue
            # Capture ranking for its own panel
            if metric_s.startswith("Player"):
                ranking_lines = metric_s
                continue
            if metric_s == "Previous Scoreboard":
                continue
            info.add_row(metric_s, value_s)
        layout["info"].update(
            Panel(info, title=f"Generation {self.currentGeneration}")
        )

        # ── Player rankings ──
        if ranking_lines:
            rank_grid = Table.grid(padding=(0, 1))
            rank_grid.add_column()
            for line in ranking_lines.strip().split("\n"):
                rank_grid.add_row(line)
            layout["ranking"].update(Panel(rank_grid, title="Player Rankings (Elo)"))

        # ── Head-to-head matrix ──
        matrix = self.population.build_head_to_head_table()
        if matrix:
            layout["matrix"].update(Panel(matrix, title="Head-to-Head Results"))

        console.print(layout)

    @staticmethod
    def clean_date(timestamp, unixDefault=False):
        try:
            if unixDefault == True:
                k = datetime.datetime.fromtimestamp(timestamp)
                return k.strftime("%Y-%m-%d %H:%M:%S")
            else:
                start = datetime.datetime.fromtimestamp(0)
                k = datetime.datetime.fromtimestamp(timestamp)
                magic = k - start
                return magic
        except:
            return 0

    @staticmethod
    def generate_game_id(generationID, i, j, cpu1, cpu2):
        # choose a random number between 1 and the number of players.
        IDPadding = generationID + "_" + str(i) + "_" + str(j)
        game_id = IDPadding + cpu1.id + cpu2.id
        return game_id

    @staticmethod
    def merge_dicts(x, y):
        z = x.copy()  # start with x's keys and values
        z.update(y)  # modifies z with y's keys and values & returns None
        return z
