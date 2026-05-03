"""Population management, Elo ratings, and evolution."""

from __future__ import annotations

import datetime
import json
import math
import multiprocessing
import operator
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

import agents.agent as agent
import agents.slowbro as sb
import core.neuroevolution as evo
import core.storage as storage
from agents.evaluator.neural import NeuralNetwork

BLACK, WHITE, EMPTY = 0, 1, -1
Black, White, empty = BLACK, WHITE, EMPTY
WIN_PT, DRAW_PT, LOSE_PT = 2, 0, -1
ONIX_ID = -2


class EloRating:
    """Elo rating system with anchoring and K-factor calibration.

    K=32 for new players (<10 games), K=24 for intermediate (10-50 games),
    K=10 for established (>50 games).
    """

    def __init__(self, k_factor: int = 32, initial_rating: int = 1200) -> None:
        self.k_factor = k_factor
        self.initial_rating = initial_rating

    def get_k_factor(self, games_played: int) -> int:
        """Get K-factor based on number of games played."""
        if games_played < 10:
            return 32  # New player - high volatility
        elif games_played < 50:
            return 24  # Intermediate - moderate volatility
        else:
            return 10  # Established - low volatility

    def expected_score(self, player_rating: float, opponent_rating: float) -> float:
        """Calculate expected score for a player against an opponent."""
        return 1 / (1 + 10 ** ((opponent_rating - player_rating) / 400))

    def update_rating(
        self,
        player_rating: float,
        opponent_rating: float,
        actual_score: float,
        games_played: int = 0,
    ) -> float:
        expected = self.expected_score(player_rating, opponent_rating)
        # Use dynamic K-factor if games_played > 0, otherwise use fixed K-factor
        if games_played > 0:
            k = self.get_k_factor(games_played)
        else:
            k = self.k_factor
        return player_rating + k * (actual_score - expected)


class Population:
    def __init__(
        self,
        num_players: int,
        ply_depth: int,
        is_debug: bool = False,
        use_parallel_mcts: Optional[bool] = None,
        num_parallel: int = 4,
        include_baseline: bool = True,
        baseline_elo: float = 500.0,
        include_onix: bool = False,
        use_neat: bool = True,
    ) -> None:
        self.is_debug = is_debug

        self.generation = 0
        self.count = num_players
        self.ply_depth = ply_depth
        self.mutationRate = 0.9
        self.players = {}
        self.champions = []
        self.player_counter = 0
        self.folderDirectory = os.path.join("..", "results", "champions")
        self.elo_system = EloRating(k_factor=32, initial_rating=1200)
        self.baseline_elo = baseline_elo

        if use_parallel_mcts is None:
            self.use_parallel_mcts = ply_depth > 1
        else:
            self.use_parallel_mcts = use_parallel_mcts
        self.parallel_threads = num_parallel
        self.use_neat = use_neat
        self.evolution = evo.NEATEvolution(self) if use_neat else evo.StandardGA(self)

        self.current_population = self.generate_players(self.count)

        self.baseline_entity = None
        if include_baseline:
            self.baseline_entity = self.generate_baseline_player()
            if self.baseline_entity.id not in self.current_population:
                self.current_population.append(self.baseline_entity.id)

        # Onix: permanent heuristic-bot fixture at ~900 Elo
        self.onixEntity = None
        if include_onix:
            self.onixEntity = self.generate_onix_player()
            if self.onixEntity.id not in self.current_population:
                self.current_population.append(self.onixEntity.id)

        nn = self.players[0].bot.nn
        self.num_weights = (
            nn.len_coefficients
            if hasattr(nn, "len_coefficients") and nn.len_coefficients > 0
            else 100
        )
        self.tau = 1 / math.sqrt(2 * math.sqrt(self.num_weights))

        # if safe mutations are enabled, we use it.
        self.safe_mutations = True
        self.debug = False
        self.crossoverMethod = 2

        # Head-to-head tracking: {(black_id, white_id): [black_wins, white_wins, draws]}
        self.head_to_head: Dict[Tuple[int, int], List[int]] = {}

    """
  Generates an individual player. This is only called in the
  generate_players() function!
  """

    def generate_player(self) -> "agent.Agent":
        bot = self.evolution.generate_bot(self.ply_depth, self.is_debug)
        human = agent.Agent(bot, initial_elo=self.baseline_elo)
        # generate ID
        human.set_id(self.player_counter)
        self.player_counter += 1
        return human

    """
  Generates a baseline player with uninitialized (random) weights.
  This player has 1200 Elo and serves as a reference point in tournaments.
  """

    def generate_baseline_player(self) -> object:
        if self.baseline_entity is not None:
            return self.baseline_entity
        bot = sb.Slowbro(ply_depth=self.ply_depth, debug=self.is_debug, use_mlx=True)
        human = agent.Agent(bot, initial_elo=self.baseline_elo)
        human.set_id(-1)
        human.isBaseline = True
        human.entity_name = "baseline"
        self.players[human.id] = human
        return human

    def generate_onix_player(self) -> object:
        from agents.onix import Onix

        onix_bot = Onix(ply_depth=self.ply_depth, debug=self.is_debug)
        ent = agent.Agent(onix_bot, initial_elo=600.0)
        ent.set_id(ONIX_ID)
        ent.entity_name = "Onix"
        ent.origin = [[0, 0, 0]]
        ent.parents = []
        self.players[ent.id] = ent
        return ent

    """
  Generates Players to participate in the tournament.
  This is only called at the beginning of the genetic algorithm.
  """

    def generate_players(self, count: int) -> list:
        players = []
        for _ in range(count):
            # generate a new human
            human = self.generate_player()
            # add it to the list of players
            self.players[human.id] = human
            # add it to the current population.
            players.append(human.id)
        return players

    """
  Self explanatory, prints the current population in order of
  Elo rating (now the primary ranking metric).
  """

    def print_population_by_points(self) -> str:
        if self.debug:
            print("Current Population:", self.current_population)
        elo_ratings = list(
            map(
                lambda x: (x, self.players[x].elo, self.players[x].points),
                self.current_population,
            )
        )
        elo_ratings = sorted(elo_ratings, key=operator.itemgetter(1), reverse=True)
        output = ""
        for i in elo_ratings:
            label = getattr(self.players[i[0]], "entity_name", None)
            player_label = f"Player {i[0]} ({label})" if label else f"Player {i[0]}"
            output += f"{player_label}\tElo: {i[1]:.1f}\tPts: {i[2]}\n"
        return output

    """
  Prints the current population in order of Elo rating.
  """

    def print_population_by_elo(self) -> str:
        if self.debug:
            print("Current Population:", self.current_population)
        elo_ratings = list(
            map(
                lambda x: (x, self.players[x].elo, self.players[x].points),
                self.current_population,
            )
        )
        elo_ratings = sorted(elo_ratings, key=operator.itemgetter(1), reverse=True)
        output = "Population by Elo Rating:\n"
        for i in elo_ratings:
            player_label = f"Player {i[0]}"
            if self.baseline_entity and i[0] == self.baseline_entity.id:
                player_label += " (baseline)"
            output += f"{player_label}\tElo: {i[1]:.1f}\tPts: {i[2]}\n"
        return output

    def printEloStats(self) -> str:
        """Print Elo statistics for the current population."""
        elos = [self.players[pid].elo for pid in self.current_population]
        avg_elo = sum(elos) / len(elos)
        best_elo = max(elos)
        worst_elo = min(elos)
        best_player = max(
            self.current_population, key=lambda pid: self.players[pid].elo
        )

        output = f"Elo Stats - Avg: {avg_elo:.1f} | Best: {best_elo:.1f} (P{best_player}) | Worst: {worst_elo:.1f}\n"
        return output

    """
  order the players by how good they are.
  Now sorts by Elo rating instead of points.
  """

    def sort_population_by_points(self) -> list:
        # create tuple of players and their elo ratings
        elo_ratings = list(
            map(lambda x: (x, self.players[x].elo), self.current_population)
        )
        # sort list of tuples by Elo (highest first)
        elo_ratings = sorted(elo_ratings, key=operator.itemgetter(1), reverse=True)
        # assign back the sorted player IDs
        self.current_population = [x[0] for x in elo_ratings]

    """
  Generate new population based on the player performance.
  Input: list of player ID's.
  Output: a new list of players.
  """

    def generate_next_population(self) -> None:
        start = datetime.datetime.now()
        self.generation += 1

        # Exclude Onix and baseline from parent/elite selection
        eligible = [pid for pid in self.current_population if pid not in (ONIX_ID, -1)]
        elites = eligible[:5]

        for i in elites:
            self.players[i].points = 0
            # games_played is preserved — K-factor decays naturally (32→24→10)

        offsprings = []
        for i in range(0, 2):
            parent_a_ID, parent_b_ID = eligible[i], eligible[i + 1]
            children = self.generate_players(4)

            # crossover from parents
            children[0], children[1] = self.crossover(
                parent_a_ID, parent_b_ID, children[0], children[1]
            )

            # assign evolution blocks to child 0 and child 1.
            self.inherit_origins(children[0], [parent_a_ID, parent_b_ID])
            self.inherit_origins(children[1], [parent_b_ID, parent_a_ID])
            # copy the caches of the parent to the child
            self.inherit_cache(children[0], parent_a_ID)
            self.inherit_cache(children[1], parent_b_ID)

            self.add_origins(children[0], [1, 1, 0])
            self.add_origins(children[1], [1, 1, 0])

            # for the last 2 offsprings, they obtain the same weights as their parents.
            self.set_weights(children[2], self.get_weights(parent_a_ID))
            self.set_weights(children[3], self.get_weights(parent_b_ID))
            self.add_origins(children[2], [0, 1, 0])
            self.add_origins(children[3], [0, 1, 0])
            self.inherit_origins(children[2], [parent_a_ID])
            self.inherit_origins(children[3], [parent_b_ID])
            # copy the caches of the parent to the child
            self.inherit_cache(children[2], parent_a_ID)
            self.inherit_cache(children[3], parent_b_ID)

            # mutate all offsprings
            # for offspring in children:
            #   self.mutate(offspring)
            # now we add children to the list of offsprings
            offsprings = offsprings + children

        # the last two children are mutations of 4th and 5th place bots.
        remainders = self.generate_players(2)
        self.set_weights(remainders[0], self.get_weights(eligible[3]))
        self.set_weights(remainders[1], self.get_weights(eligible[4]))
        self.add_origins(remainders[0], [0, 1, 0])
        self.add_origins(remainders[1], [0, 1, 0])
        self.inherit_origins(remainders[0], [self.current_population[3]])
        self.inherit_origins(remainders[1], [self.current_population[4]])
        # copy the caches of the parent to the child
        self.inherit_cache(remainders[0], self.current_population[3])
        self.inherit_cache(remainders[1], self.current_population[4])

        # add remainders to list of offsprings
        offsprings = offsprings + remainders
        if self.debug:
            print("offsprings:", offsprings)

        # mutate the offsprings. we should parallelise this.
        mutations = []
        print("Computing Mutations..")
        threadCount = multiprocessing.cpu_count()
        if len(offsprings) < threadCount:
            threadCount = len(offsprings)
        with multiprocessing.Pool(processes=threadCount) as pool:
            mutations = pool.map(self.mutate, offsprings)
            pool.close()
            pool.join()

        # for i in offsprings:
        #   mutations.append(self.mutate(i))
        print("Finished computing mutations.")

        # now that we have the mutations, load them to each agent.
        for mutation in mutations:
            self.evolution.load_mutation_result(*mutation)

        # All offspring start at initial Elo — they earn their rating through play
        for oid in offsprings:
            self.players[oid].elo = self.baseline_elo

        # Reset games_played for all offspring (they start fresh)
        for offspring_id in offsprings:
            self.players[offspring_id].games_played = 0
            self.players[
                offspring_id
            ].points = 0  # Also reset points for new generation

        newPopulation = offsprings + elites
        # Preserve Onix across generations (keep its Elo, never reset)
        if self.onixEntity is not None:
            newPopulation.append(self.onixEntity.id)
            self.players[ONIX_ID].points = 0
        # Preserve baseline entity across generations
        if self.baseline_entity is not None:
            newPopulation.append(self.baseline_entity.id)
            self.players[self.baseline_entity.id].elo = self.baseline_elo
            self.players[self.baseline_entity.id].points = 0
        self.current_population = newPopulation
        self.count = len(self.current_population)
        end = datetime.datetime.now() - start
        if self.debug:
            print("DONE, that took", end)

        self.kill_caches()

        print("Successfully computed offsprings for the next generation.")

    def heuristic_crossover(self, cpu1, cpu2, child1, child2) -> None:
        print("Processing Crossover")
        mother = self.players[cpu1].bot.nn.weights
        father = self.players[cpu2].bot.nn.weights

        randomlayer = random.randint(0, len(mother) - 1)
        lenWeightsRandlayer = len(father[randomlayer])
        maxLim = int(0.4 * lenWeightsRandlayer)
        randWeightIndexes = list(
            set([random.randint(0, lenWeightsRandlayer - 1) for i in range(maxLim)])
        )

        newWeightSetA = []
        newWeightSetB = []

        for i in range(lenWeightsRandlayer):
            if i in randWeightIndexes:
                newWeightSetA.append(father[randomlayer][i].tolist()[0])
                newWeightSetB.append(mother[randomlayer][i].tolist()[0])
            else:
                newWeightSetA.append(mother[randomlayer][i].tolist()[0])
                newWeightSetB.append(father[randomlayer][i].tolist()[0])

        # turn back into matrix
        newWeightSetA = np.matrix(newWeightSetA)
        newWeightSetB = np.matrix(newWeightSetB)

        # for i in newWeightSetA:
        #   print(i)

        # now to load them to the offspring
        self.set_weights(child1, self.get_weights(cpu1))
        self.set_weights(child2, self.get_weights(cpu2))

        self.players[child2].bot.nn.weights[randomlayer] = newWeightSetA
        self.players[child2].bot.nn.weights[randomlayer] = newWeightSetB

        print("Crossover Successful.")
        # return the pair of children
        return (child1, child2)

    """
  Crossover mechanism for creating offspring children
  Input: two parents, two children, two indexes to swap from
  """

    def crossover(self, cpu1, cpu2, child1, child2) -> None:
        """Crossover two parents into two children using current evolution strategy."""
        if self.debug:
            print(
                "Implementing Crossover for IDS " + str(child1) + "," + str(child2),
                end=".. ",
            )
        result = self.evolution.crossover(cpu1, cpu2, child1, child2)
        print("Crossover Successful.")
        return result

    """
  Mutate the weights of the neural network.
  """

    def mutate(self, cpu) -> None:
        """Mutate the agent using the current evolution strategy."""
        if self.debug:
            print("Generating mutations for player " + str(cpu))
        return self.evolution.mutate(cpu)

    """
  Static function to create safe mutations
  """

    def safe_mutation(self, cpu, static: bool = False) -> None:
        print("Computing Safe Mutations..")
        cache = self.get_move_cache(cpu)
        curreneWeight1D = self.players[cpu].bot.nn.get_all_coefficients()

        # get a subset of those cached moves.
        subset = {}
        subsetSize = int(len(cache.keys()) / 10)
        if subsetSize < 1000:
            subsetSize = len(cache.keys())
        for _ in range(subsetSize):
            rand = random.choice(list(cache.keys()))
            subset[rand] = np.array(rand)

        # now we find the safest mutation
        bestWeight = curreneWeight1D
        bestScore = 0

        for su in range(100):
            weights = None
            if static:
                # create a new mutation
                multipliers = np.random.random_sample([self.num_weights])
                multipliers = self.tau * multipliers
                multipliers = np.exp(multipliers)
                weights = multipliers * curreneWeight1D
                weights = np.clip(weights, -1, 1)
                self.players[cpu].bot.nn.load_coefficients(weights)
            else:
                for w in range(len(self.players[cpu].bot.nn.weights)):
                    weights = self.players[cpu].bot.nn.weights[w]
                    multipliers = np.random.random_sample(weights.shape)
                    multipliers = self.tau * multipliers
                    multipliers = np.exp(multipliers)
                    # print(weights.shape, multipliers.shape)
                    self.players[cpu].bot.nn.weights[w] = np.add(weights, multipliers)
                    self.players[cpu].bot.nn.weights[w] = np.clip(
                        self.players[cpu].bot.nn.weights[w], -1, 1
                    )

                    biases = self.players[cpu].bot.nn.biases[w]
                    multipliers = np.random.random_sample(biases.shape)
                    multipliers = self.tau * multipliers
                    multipliers = np.exp(multipliers)
                    # print(weights.shape, multipliers.shape)
                    self.players[cpu].bot.nn.biases[w] = np.add(biases, multipliers)
                    self.players[cpu].bot.nn.biases[w] = np.clip(
                        self.players[cpu].bot.nn.biases[w], -1, 1
                    )
                weights = self.players[cpu].bot.nn.get_all_coefficients()

            # calculate to see whether these weights are better
            qa = 0
            for i in subset:
                # evaluate this cached move
                eval_a = self.players[cpu].bot.nn.compute(subset[i])
                # compare it to the current value
                if eval_a / cache[i] >= 0.95:
                    qa += 1
            if qa > bestScore:
                bestWeight = weights
                bestScore = qa
                # print("New best:", qa, su)
            percentile = round(bestScore * 100 / len(subset), 2)
            print(
                str(cpu)
                + " - Count:"
                + str(su)
                + " Best:"
                + str(bestScore)
                + "/"
                + str(len(subset))
                + " - "
                + str(percentile)
                + "%\r",
                end="",
            )
        # print("\nDONE")
        return bestWeight

    """
  Saves champions to a file.
  """

    def save_champions_to_file(self, folderDirectory: str) -> None:
        folderDirectory = os.path.join(folderDirectory, "champions")
        if not os.path.isdir(folderDirectory):
            os.makedirs(folderDirectory)

        i = self.generation
        championID = self.champions[-1]
        coeffs = self.players[championID].bot.nn.get_all_coefficients()
        meta = {
            "pid": str(self.players[championID].id),
            "champ_range": json.dumps(self.players[championID].champ_range),
            "champ_score": float(self.players[championID].champ_score),
        }

        # Write .npz (compressed numpy, primary format)
        npz_path = os.path.join(folderDirectory, str(i) + ".npz")
        storage.save_champion_npz(npz_path, coeffs, meta)

        # Write .json (backward compat, lightweight metadata only)
        json_path = os.path.join(folderDirectory, str(i) + ".json")
        championJson = {
            str(i): {
                "pid": self.players[championID].id,
                "coefficents": coeffs.tolist(),
                "champ_range": self.players[championID].champ_range,
                "champ_score": self.players[championID].champ_score,
            }
        }
        with open(json_path, "w") as outfile:
            json.dump(championJson, outfile)

        print(f"saved champs to {i}.npz + {i}.json")

    """
  Saves genomic properties to a file. Each champion's properties
  gets saved in this file, such as whether they were made from
  mutation, their parents IDs, whether crossovers were used 
  and so on.
  """

    def save_population_genomes(self, folderDirectory: str) -> None:
        # check save directory exists prior to saving
        if not os.path.isdir(folderDirectory):
            os.makedirs(folderDirectory)

        agent = {}
        for player_id in self.current_population:
            # store player and its weights.
            agent[player_id] = {}
            agent[player_id]["score"] = self.players[player_id].points
            agent[player_id]["origin"] = self.players[player_id].origin
            agent[player_id]["parents"] = self.players[player_id].parents

        filename = "genomes.json"
        with open(os.path.join(folderDirectory, filename), "w") as outfile:
            json.dump(agent, outfile)

            # append to file.
        print("saved players genomic info.")

    """
  NOT USED
  """

    def save_population_to_db(self, db) -> None:
        population = self.current_population
        """
    Stores the population into Mongo. 
    """
        keys = []
        if db.connected:
            for i in population:
                if not db.check_player_exists(i.id):
                    entry = db.write("players", i.getDict())
                    keys.append(entry)
                else:
                    keys.append(i.id)
        return keys

    """
  Allocates points to players based on the game outcomes
  Also updates Elo ratings for both players.
  """

    def allocate_points(self, results, black, white) -> None:
        black_rating = self.players[black].elo
        white_rating = self.players[white].elo
        black_games = self.players[black].games_played
        white_games = self.players[white].games_played

        self.players[black].games_played += 1
        self.players[white].games_played += 1

        # Determine the result scores for Elo calculation
        if results["Winner"] == Black:
            black_score, white_score = 1.0, 0.0
            self.players[black].points += WIN_PT
            self.players[white].points += LOSE_PT
        elif results["Winner"] == White:
            black_score, white_score = 0.0, 1.0
            self.players[black].points += LOSE_PT
            self.players[white].points += WIN_PT
        else:
            black_score, white_score = 0.5, 0.5

        # Onix is a fixed anchor at 900, never moves
        if getattr(self.players[black], "entity_name", None) != "Onix":
            self.players[black].elo = self.elo_system.update_rating(
                black_rating, white_rating, black_score, black_games
            )
        if getattr(self.players[white], "entity_name", None) != "Onix":
            self.players[white].elo = self.elo_system.update_rating(
                white_rating, black_rating, white_score, white_games
            )

        # Track head-to-head
        self.record_match(black, white, results["Winner"])

    def record_match(self, black: int, white: int, winner: int) -> None:
        """Record a game result for head-to-head tracking.

        Args:
            black: Black player ID.
            white: White player ID.
            winner: Winner colour (BLACK=0, WHITE=1, EMPTY=-1 for draw).
        """
        from core.constants import BLACK, WHITE, EMPTY

        key = (black, white)
        if key not in self.head_to_head:
            self.head_to_head[key] = [0, 0, 0]  # black_wins, white_wins, draws
        if winner == BLACK:
            self.head_to_head[key][0] += 1
        elif winner == WHITE:
            self.head_to_head[key][1] += 1
        else:
            self.head_to_head[key][2] += 1

    def build_standings_table(self):
        """Build a compact standings table: Player | Elo | Pts | W-D-L | Score.

        Aggregates head-to-head data into per-player totals.
        """
        from rich.table import Table

        pids = [pid for pid in self.current_population]
        if not pids:
            return None

        totals = {}
        for a in pids:
            totals[a] = {"w": 0, "l": 0, "d": 0}
            for b in pids:
                if a == b:
                    continue
                rec = self.head_to_head.get((a, b), [0, 0, 0])
                totals[a]["w"] += rec[0]
                totals[a]["l"] += rec[1]
                totals[a]["d"] += rec[2]
                rec = self.head_to_head.get((b, a), [0, 0, 0])
                totals[a]["w"] += rec[1]
                totals[a]["l"] += rec[0]
                totals[a]["d"] += rec[2]

        by_elo = sorted(pids, key=lambda pid: self.players[pid].elo, reverse=True)

        t = Table(title="Standings")
        t.add_column("Player", style="cyan", no_wrap=True)
        t.add_column("Elo", justify="right")
        t.add_column("Pts", justify="right")
        t.add_column("W", justify="right")
        t.add_column("D", justify="right")
        t.add_column("L", justify="right")
        t.add_column("Score", justify="right")

        for pid in by_elo:
            w = totals[pid]["w"]
            l = totals[pid]["l"]
            d = totals[pid]["d"]
            total_g = w + l + d
            score = f"{w / total_g:.3f}" if total_g else "—"
            label = f"P{pid}"
            ent = getattr(self.players[pid], "entity_name", None)
            if ent:
                label = ent
            t.add_row(
                label,
                f"{self.players[pid].elo:.1f}",
                str(self.players[pid].points),
                str(w),
                str(d),
                str(l),
                score,
            )

        return t

    def build_matrix_table(self, max_rows: int = 8):
        """Build a compact per-player win matrix.

        Only shows the top ``max_rows`` players by Elo to keep the
        table readable. Cell shows row player's wins vs column player.
        """
        from rich.table import Table

        pids = [pid for pid in self.current_population]
        if not pids:
            return None
        # Sort by Elo descending, take top N
        by_elo = sorted(pids, key=lambda pid: self.players[pid].elo, reverse=True)
        pids = by_elo[:max_rows]

        t = Table(title="Win Matrix (top by Elo)")
        t.add_column("", style="cyan", no_wrap=True)
        for pid in pids:
            label = getattr(self.players[pid], "entity_name", None) or f"P{pid}"
            t.add_column(label, justify="center", max_width=5)

        for a in pids:
            label = getattr(self.players[a], "entity_name", None) or f"P{a}"
            row = [label]
            for b in pids:
                if a == b:
                    row.append("—")
                else:
                    rec = self.head_to_head.get((a, b), [0, 0, 0])
                    w = rec[0]
                    row.append(str(w) if w else ".")
            t.add_row(*row)

        return t

    def add_champion(self) -> None:
        for pid in self.current_population:
            if pid not in (ONIX_ID, -1):
                self.champions.append(pid)
                return

    """
  Assign weights to a bot's neural net.
  """

    def set_weights(self, botID, weights) -> None:
        self.evolution.set_weights(botID, weights)

    def get_weights(self, botID) -> object:
        return self.evolution.get_weights(botID)

    """
  Helper function to retrieve cache if it exists,
  otherwise return empty dict.
  """

    def get_move_cache(self, botID) -> dict:
        if self.players[botID].bot.enable_cache:
            return self.players[botID].bot.cache
        else:
            return {}  # Return empty dict instead of False

    """
  Kill the caches when we're done with mutations or whatever.
  This is really important!
  """

    def kill_caches(self) -> None:
        for i in range(self.player_counter):
            self.players[i].bot.cache = {}
        print("Killed all caches.")

    # Done
    def add_origins(self, botID, values) -> None:
        self.players[botID].origin = [values]

    # Done
    def inherit_origins(self, botID, parentIDs) -> None:
        for i in parentIDs:
            self.players[botID].parents.append(i)

    def inherit_cache(self, botID, parentID) -> None:
        self.players[botID].bot.cache = self.players[parentID].bot.cache

    @staticmethod
    def generate_fake_moves():
        print("Generating fake moves", end=".. ")
        # we create a dictionary of fake moves.
        nn = NeuralNetwork(layer_list=[32, 40, 10, 1])
        fakeMoves = {}
        for _ in range(10000):
            # generate a random list of nn inputs
            state = np.random.random_sample([32])
            stateID = tuple(state)
            evals = nn.compute(state)
            fakeMoves[stateID] = evals
            # evaluate it
        # get a copy of the current nn coefs.
        coefs = nn.get_all_coefficients()
        # return this.
        print("Generated Fake moves.")
        return (fakeMoves, coefs)


if __name__ == "__main__":
    # we can use this to test out the population program
    pass
    # x = Population(15,1)
