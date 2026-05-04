from __future__ import annotations

import slowpoke.core.population as pop
import slowpoke.core.game as game
import slowpoke.core.storage as storage

import slowpoke.core.mongo as mongo

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

from slowpoke import statistics

from slowpoke.core.constants import BLACK, WHITE, EMPTY

Black, White, empty = BLACK, WHITE, EMPTY
ChampWIN_PT, ChampDRAW_PT, ChampLOSE_PT = 1, 0, -1


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
    def __init__(self, options, tui=None, save_location=None):
        # initialise default variables when needed.
        options = option_defaults(options)
        self.is_debug_mode = options["debugMode"]
        self.tui = tui  # Optional TournamentApp for live display
        # Declare base information
        self.ply_depth = options["ply_depth"]
        self.generations = options["NumberOfGenerations"]
        self.population_size = options["Population"]  # number of players
        # generate the initial population.
        self.population = pop.Population(
            self.population_size,
            self.ply_depth,
            self.is_debug_mode,
            options["use_parallel_mcts"],
            options["num_parallel"],
            include_onix=True,
        )
        # time handlers
        self.start_time = datetime.datetime.now().timestamp()
        self.average_game_time = 0
        self.average_gen_length = 0
        self.remaining_time = 0
        self.est_date_finished = 0
        self.gen_time_lengths = np.array([])
        self.current_gen_start_time = datetime.datetime.now().timestamp()
        # current generation game counts
        self.games_finished = 0
        self.games_queued = 0
        self.currentGeneration = 0
        # champions
        self.are_champions_playing = False
        self.last_champion_score = 0
        self.cumulative_score = 0
        self.average_champion_growth = 0
        self.recent_champion_scores = 0
        self.play_previous_champ_count = 5
        self.champ_games_rounds_count = 6  # should always be even and at least 2.
        self.progress = []

        self.previous_champ_point_list = None
        # Initiate other information
        self.processors = multiprocessing.cpu_count() - 1
        self.config = self.load_json_config(options["mongoConfigPath"])
        self.mongo_connected = options["connectMongo"]
        self.total_games_per_gen = (options["Population"] ^ 2) - options["Population"]

        # placeholder values
        self.game_id_counter = 0
        # once we have the config file we can proceed and initiate our MongoDB connection.
        self.init_mongo_connection()
        # we also want to save the stats offline
        self.generation_stats = []
        if save_location:
            self.folder_name = os.path.basename(save_location.rstrip("/"))
            self.save_location = save_location
        else:
            self.folder_name = (
                str(self.clean_date(self.start_time, True))
                + " "
                + str(self.ply_depth)
                + "ply"
            )
            self.save_location = os.path.join(
                options["resultsLocation"], self.folder_name
            )
        self.options = options  # Store for reference
        # Set up logging
        self.log_file = os.path.join(self.save_location, "training.log")
        self._setup_logging()
        # generate charts as we go?
        self.generate_charts_every_round = True
        self._resume_from = 0  # generation to start from (set >0 on resume)

    def __getstate__(self) -> dict:
        """Strip unpicklable attributes for multiprocessing workers."""
        state = self.__dict__.copy()
        state["tui"] = None
        state["logger"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

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
            if self.mongo_connected:
                self.db.initiate(self.config["MongoURI"])
        except:
            pass

    def _setup_logging(self) -> None:
        """Set up logging to both file and console."""
        # Ensure save directory exists
        if not os.path.isdir(self.save_location):
            os.makedirs(self.save_location)

        # Remove any existing handlers so we can reconfigure
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Configure logging to file (append mode)
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            filemode="a",
        )
        self.logger = logging.getLogger(__name__)
        is_resume = self._resume_from > 0 if hasattr(self, "_resume_from") else False
        self.log("Training resumed" if is_resume else "Training started")
        self.log(
            f"Population: {self.population_size}, Ply Depth: {self.ply_depth}, Generations: {self.generations}"
        )

    def log(self, message: str) -> None:
        """Write a message to the log file."""
        if hasattr(self, "logger"):
            self.logger.info(message)

    def log_status_info(self) -> None:
        """Write current status info to log file."""
        data = self.status_info()
        for section, entries in data.items():
            self.log(f"[{section}]")
            for key, value in entries.items():
                self.log(f"  {key}: {value}")

    def tournament(self) -> None:
        """
        Tournament; full round-robin where every pair plays both colours.
        Returns the players in order of how good they are.
        """
        self.log("=" * 60)
        self.log("STARTING TOURNAMENT")
        self.log(f"Generation: {self.currentGeneration}")
        self.log(f"Population size: {len(self.population.current_population)} players")
        self.population.head_to_head.clear()

        # Full round-robin: each pair plays each colour exactly once
        gamePool = []
        players = self.population.current_population[:]
        self.log("Scheduling full round-robin games...")
        gen_idx = 0
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                pid_i, pid_j = players[i], players[j]
                gamePool.append(
                    {
                        "idx": gen_idx,
                        "game_id": self.game_id_counter,
                        "black": self.population.players[pid_i],
                        "white": self.population.players[pid_j],
                        "dbURI": False,
                        "debugInfo": False,
                    }
                )
                gen_idx += 1
                self.game_id_counter += 1
                gamePool.append(
                    {
                        "idx": gen_idx,
                        "game_id": self.game_id_counter,
                        "black": self.population.players[pid_j],
                        "white": self.population.players[pid_i],
                        "dbURI": False,
                        "debugInfo": False,
                    }
                )
                gen_idx += 1
                self.game_id_counter += 1
        self.games_queued = len(gamePool)
        self.log(f"Total games scheduled: {len(gamePool)}")

        # run game simulations.
        results = [None] * len(gamePool)
        threadCount = self.processors
        if self.processors > len(gamePool):
            threadCount = len(gamePool)
        self.log(f"Running games with {threadCount} parallel processes...")
        self.games_finished = 0
        total_games = len(gamePool)
        with multiprocessing.Pool(processes=threadCount) as pool:
            for result in pool.imap_unordered(self.game_worker, gamePool, chunksize=16):
                results[result["idx"]] = result
                self.games_finished += 1
                self.log(
                    f"Game {self.games_finished}/{total_games}: "
                    f"P{result['black']} vs P{result['white']} → "
                    f"{'Black' if result['game']['Winner'] == Black else 'White' if result['game']['Winner'] == White else 'Draw'}"
                )
                if self.tui:
                    self.tui.push_update()
            pool.close()
            pool.join()
        self.log("All games completed")

        # process results (cache merging)
        self.log("Processing game results...")
        for i in range(len(results)):
            self.population.allocate_points(
                results[i]["game"], results[i]["black"], results[i]["white"]
            )
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
        start_gen = self._resume_from if hasattr(self, "_resume_from") else 0
        for i in range(start_gen, self.generations):
            print("Initiating generation", i)
            self.log("=" * 60)
            self.log(f"Starting generation {i}")
            # increment generation count
            self.currentGeneration = i
            self.current_gen_start_time = datetime.datetime.now().timestamp()
            # reset game count statistics prior to running
            self.games_finished = 0
            self.games_queued = 0
            # initiate timestamp
            startTime = datetime.datetime.now()
            # make bots play each other.
            try:
                self.population, generationResults = self.tournament()
            except KeyboardInterrupt:
                self.log("Generation interrupted during tournament")
                raise

            # compute champion games (runs independently of others)
            self.log("Running champion games...")
            self.run_champions()
            # save champions to file
            self.log("Saving champions to file...")
            self.population.save_champions_to_file(self.save_location)
            # save genomic details
            self.log("Saving population genomes...")
            self.population.save_population_genomes(self.save_location)
            # get the best players and generate a new population from them.
            self.log("Generating next population...")
            self.population.generate_next_population()
            self.population_size = self.population.count
            # initiate end timestamp and add time difference length to list.
            timeDifference = (datetime.datetime.now() - startTime).total_seconds()
            self.gen_time_lengths = np.hstack((self.gen_time_lengths, timeDifference))
            self.log(f"Generation complete ({timeDifference:.1f}s)")
            # need to store the results of this into a json file!
            self.generation_stats.append(
                {
                    "stats": self.status_info(),
                    "games": generationResults,
                    "durationInSeconds": str(timeDifference),
                }
            )
            self.save_training_stats(self.save_location, self.generation_stats)
            # Save checkpoint for resume
            try:
                self.save_checkpoint()
            except Exception as e:
                self.log(f"save_checkpoint failed (non-fatal): {e}")
            if self.generate_charts_every_round:
                try:
                    self.generate_stats()
                except Exception as e:
                    self.log(f"generate_stats failed (non-fatal): {e}")
            # Display status at generation boundary
            self.display_status_info(force_display=True)

    def nuke_cache(self) -> None:
        for i in self.population:
            self.population[i].bot.cache = {}

    def generate_stats(self) -> None:
        # create statistics
        stats = statistics.Statistics(self.folder_name)
        stats.load_statistics_file()
        stats.save_charts()
        print("I made some charts!")

    def save_training_stats(self, save_location: str, stats) -> None:
        """Save generation game results as Parquet."""
        if not os.path.isdir(save_location):
            os.makedirs(save_location)

        try:
            storage.save_statistics_parquet(save_location, stats)
        except ImportError:
            self.log("pyarrow not available — skipping statistics export")
        except Exception as e:
            self.log(f"Parquet write failed (non-fatal): {e}")

    def save_checkpoint(self) -> None:
        """Save a checkpoint that can be used to resume training."""
        ckpt_dir = os.path.join(self.save_location, "checkpoint")
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)

        data = {
            "generator_version": 1,
            "generator": {
                "currentGeneration": self.currentGeneration,
                "generations": self.generations,
                "GenerationTimeLengths": self.gen_time_lengths.tolist(),
                "progress": self.progress,
                "cummulativeScore": self.cumulative_score,
                "AverageGameTime": self.average_game_time,
                "AverageGenrationLength": self.average_gen_length,
                "gameIDCounter": self.game_id_counter,
                "generationStats": self.generation_stats,
                "StartTime": self.start_time,
                "folderName": self.folder_name,
                "saveLocation": self.save_location,
                "options": self.options,
                "ply_depth": self.ply_depth,
                "populationSize": self.population_size,
                "processors": self.processors,
                "LastChampionScore": self.last_champion_score,
                "previousChampPointList": self.previous_champ_point_list,
            },
            "population": self.population.get_checkpoint_data(),
        }

        path = os.path.join(ckpt_dir, "latest.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self.log(f"Checkpoint saved ({path})")

    @classmethod
    def from_checkpoint(cls, folder_path: str, tui=None) -> "Generator":
        """Create a Generator from a saved checkpoint for resuming training."""
        ckpt_path = os.path.join(folder_path, "checkpoint", "latest.json")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

        with open(ckpt_path) as f:
            data = json.load(f)

        gen_data = data["generator"]
        options = gen_data["options"]

        # Create generator (will init a fresh population — we overwrite it)
        gen = cls(options, tui, save_location=gen_data["saveLocation"])

        # Restore generator scalar state
        gen.currentGeneration = gen_data["currentGeneration"]
        gen.generations = gen_data.get("generations", gen.generations)
        gen.gen_time_lengths = np.array(gen_data.get("GenerationTimeLengths", []))
        gen.progress = list(gen_data.get("progress", []))
        gen.cumulative_score = gen_data.get("cummulativeScore", 0)
        gen.average_game_time = gen_data.get("AverageGameTime", 0)
        gen.average_gen_length = gen_data.get("AverageGenrationLength", 0)
        gen.game_id_counter = gen_data.get("gameIDCounter", 0)
        gen.generation_stats = list(gen_data.get("generationStats", []))
        gen.start_time = gen_data.get("StartTime", gen.start_time)
        gen.population_size = gen_data.get("populationSize", gen.population_size)
        gen.last_champion_score = gen_data.get("LastChampionScore", 0)
        gen.previous_champ_point_list = gen_data.get("previousChampPointList")

        # Restore population
        gen.population.load_checkpoint_data(data["population"])

        # Set resume point (next generation after the last completed one)
        gen._resume_from = gen.currentGeneration + 1
        gen.currentGeneration = gen.currentGeneration  # last completed

        # Reconfigure logging for the existing folder
        gen.log_file = os.path.join(gen.save_location, "training.log")
        gen._setup_logging()

        return gen

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
        gameRound = int(self.champ_games_rounds_count / 2)
        # playback counter
        playcounter = np.size(self.progress)
        if playcounter > self.play_previous_champ_count:
            playcounter = self.play_previous_champ_count

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
        self.are_champions_playing = True
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
            n = self.play_previous_champ_count
            results = [l[i : i + n] for i in range(0, len(l), n)]
            # calculate the gradient of the scores.

            medians = []
            for i in results:
                medians.append(np.mean(i))

            self.previous_champ_point_list = medians

            # compute new champ points compared to previous champ
            newChampPoints = np.mean(medians)
            self.cumulative_score += newChampPoints
            # store points.
            self.progress.append(newChampPoints)
            self.population.players[
                self.population.champions[-1]
            ].champ_score = newChampPoints
            self.population.players[self.population.champions[-1]].champ_range = results
        else:
            # theres only one champion, don't play.
            self.progress.append(0)
        self.are_champions_playing = False
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
            "idx": i.get("idx", 0),
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
        """Return structured status data as a dict of {category: {metric: value}}.

        All values are pre-formatted strings.
        """
        current_time = datetime.datetime.now().timestamp()
        recent_scores = self.progress[-7:]

        average_gen_time = np.mean(self.gen_time_lengths)

        percentage_est = 0.0
        if not np.isnan(average_gen_time) and average_gen_time > 0:
            percentage_est = min(
                (current_time - self.current_gen_start_time) / average_gen_time,
                1.0,
            )

        num_gens = np.size(self.progress)
        remaining_gen_seconds = average_gen_time - (
            current_time - self.current_gen_start_time
        )
        remaining_gen_count = self.generations - num_gens

        current_run_time = datetime.datetime.now() - datetime.datetime.fromtimestamp(
            self.start_time
        )

        est_remaining_seconds = (
            (remaining_gen_count * average_gen_time)
            + np.sum(self.gen_time_lengths)
            - current_run_time.total_seconds()
        )

        est_end_timestamp = (
            est_remaining_seconds + self.start_time + current_run_time.total_seconds()
        )

        # -- helpers --
        def _nan(val: float) -> bool:
            try:
                return bool(np.isnan(val))
            except (TypeError, ValueError):
                return False

        def _dur(val: float) -> str:
            """Format a duration in seconds, or show dash for no data."""
            if val is None or _nan(val) or val < 0:
                return "—"
            return str(self.clean_date(float(val)))

        def _ts(val: float) -> str:
            """Format a unix timestamp, or show dash for no data."""
            if val is None or _nan(val) or val < 0:
                return "—"
            return str(self.clean_date(float(val), True))

        def _maybe_nan(val: float) -> str:
            """Return formatted float or dash."""
            if val is None or _nan(val):
                return "—"
            return f"{float(val):.2f}"

        # -- progress --
        progress = {
            "generation": f"{num_gens}/{self.generations}",
            "population": str(self.population_size),
            "ply depth": str(self.ply_depth),
            "mongo": "Yes" if self.mongo_connected else "No",
            "cores": str(self.processors),
            "debug": "Yes" if self.is_debug_mode else "No",
        }

        def _progress_bar(current: int, total: int, width: int = 20) -> str:
            """Build a text progress bar like '████████░░ 80% (42/210)'."""
            if total == 0:
                return "—"
            ratio = current / total
            filled = int(ratio * width)
            bar = "█" * filled + "░" * (width - filled)
            return f"{bar} {ratio:.0%} ({current}/{total})"

        # -- timing --
        timing = {
            "start": _ts(self.start_time),
            "runtime": str(current_run_time),
            "est end": _ts(est_end_timestamp),
            "est remaining": _dur(est_remaining_seconds),
            "games": _progress_bar(self.games_finished, self.games_queued),
            "mean game": _dur(average_gen_time),
            "gen progress": f"{round(percentage_est * 100, 2)}%",
            "remaining gen": _dur(remaining_gen_seconds),
        }

        # -- champion --
        avg_recent = 0.0
        if len(recent_scores) > 0:
            avg_recent = float(np.mean(recent_scores))

        recent_str = "—"
        if len(recent_scores) > 0:
            recent_str = "[{}]".format(
                ", ".join("{:0.2f}".format(x) for x in recent_scores)
            )

        champ_range_str = "—"
        if (
            self.previous_champ_point_list is not None
            and len(self.previous_champ_point_list) > 0
        ):
            champ_range_str = ", ".join(
                "{:0.2f}".format(x) for x in self.previous_champ_point_list
            )

        champion = {
            "playing": "Yes" if self.are_champions_playing else "No",
            "prev score": _maybe_nan(self.last_champion_score),
            "cumulative": f"{self.cumulative_score:.2f}",
            "avg growth": f"{avg_recent:.2f}",
            "recent scores": recent_str,
            "prev champ range": champ_range_str,
        }

        return {
            "progress": progress,
            "timing": timing,
            "champion": champion,
        }

    @staticmethod
    def _build_info_table(entries: dict[str, str], title: str) -> Panel:
        """Build a titled Panel containing a two-column key-value Rich Table."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        for key, value in entries.items():
            table.add_row(key, value)
        return Panel(table, title=title)

    def display_status_info(self, force_display: bool = False) -> None:
        """Log status info to file. Display rich panels or push to TUI."""
        self.log_status_info()

        if self.tui:
            self.tui.push_update()
            return

        console = Console()
        console.clear()
        layout = Layout()
        layout.split_column(
            Layout(name="info"),
            Layout(name="ranking"),
        )

        data = self.status_info()
        info = Layout()
        info.split_column(
            Layout(name="progress-section"),
            Layout(name="timing-section"),
            Layout(name="champion-section"),
        )
        info["progress-section"].update(
            self._build_info_table(data["progress"], "Progress")
        )
        info["timing-section"].update(self._build_info_table(data["timing"], "Timing"))
        info["champion-section"].update(
            self._build_info_table(data["champion"], "Champion")
        )
        layout["info"].update(Panel(info, title=f"Generation {self.currentGeneration}"))

        standings = self.population.build_standings_table()
        if standings:
            layout["ranking"].update(Panel(standings, title="Standings"))

        console.print(layout)

    @staticmethod
    def clean_date(timestamp, unixDefault=False):
        try:
            if unixDefault:
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
