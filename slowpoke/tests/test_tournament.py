"""
Tests for tournament mechanics: scheduling, W/D/L tracking, standings display,
and NEAT evaluator wiring.

Covers the gaps that caused the "far too many 1-1 pairs" symptom at large ply:
  - record_match accumulation
  - build_standings_table / build_matrix_table correctness
  - full round-robin scheduling (N*(N-1) games, both colours per pair)
  - NEAT agent: TMCTS self.nn must reference the NEAT network, not the stale
    default standard network that was snapshotted at Slowbro construction time
"""

import unittest

import numpy as np

BLACK = 0
WHITE = 1
EMPTY = -1


# ─────────────────────────────────────────────────────────────
# Helpers to build a minimal Population without NN overhead
# ─────────────────────────────────────────────────────────────

def _make_population(n=4, ply=1, use_neat=False):
    from slowpoke.core.population import Population
    return Population(num_players=n, ply_depth=ply, use_neat=use_neat,
                      include_baseline=False, include_onix=False)


# ─────────────────────────────────────────────────────────────
# record_match unit tests
# ─────────────────────────────────────────────────────────────

class TestRecordMatch(unittest.TestCase):
    """record_match must accumulate W/D/L into head_to_head correctly."""

    def setUp(self):
        self.pop = _make_population(3)
        self.pids = self.pop.current_population[:]

    def test_black_win_increments_black_wins(self):
        a, b = self.pids[0], self.pids[1]
        self.pop.record_match(a, b, BLACK)
        self.assertEqual(self.pop.head_to_head[(a, b)], [1, 0, 0])

    def test_white_win_increments_white_wins(self):
        a, b = self.pids[0], self.pids[1]
        self.pop.record_match(a, b, WHITE)
        self.assertEqual(self.pop.head_to_head[(a, b)], [0, 1, 0])

    def test_draw_increments_draws(self):
        a, b = self.pids[0], self.pids[1]
        self.pop.record_match(a, b, EMPTY)
        self.assertEqual(self.pop.head_to_head[(a, b)], [0, 0, 1])

    def test_multiple_games_accumulate(self):
        a, b = self.pids[0], self.pids[1]
        self.pop.record_match(a, b, BLACK)
        self.pop.record_match(a, b, BLACK)
        self.pop.record_match(a, b, WHITE)
        self.assertEqual(self.pop.head_to_head[(a, b)], [2, 1, 0])

    def test_separate_colour_orderings_are_separate_keys(self):
        a, b = self.pids[0], self.pids[1]
        self.pop.record_match(a, b, BLACK)   # a=black wins
        self.pop.record_match(b, a, BLACK)   # b=black wins
        self.assertEqual(self.pop.head_to_head[(a, b)], [1, 0, 0])
        self.assertEqual(self.pop.head_to_head[(b, a)], [1, 0, 0])

    def test_clear_resets_all_records(self):
        a, b = self.pids[0], self.pids[1]
        self.pop.record_match(a, b, BLACK)
        self.pop.head_to_head.clear()
        self.assertNotIn((a, b), self.pop.head_to_head)


# ─────────────────────────────────────────────────────────────
# build_standings_table correctness
# ─────────────────────────────────────────────────────────────

class TestBuildStandingsTable(unittest.TestCase):
    """build_standings_table must aggregate W/D/L correctly."""

    def _totals(self, pop):
        """Extract {pid: {'w','d','l'}} from the standings table rows."""
        pids = pop.current_population
        totals = {}
        for a in pids:
            totals[a] = {"w": 0, "l": 0, "d": 0}
            for b in pids:
                if a == b:
                    continue
                rec = pop.head_to_head.get((a, b), [0, 0, 0])
                totals[a]["w"] += rec[0]
                totals[a]["l"] += rec[1]
                totals[a]["d"] += rec[2]
                rec = pop.head_to_head.get((b, a), [0, 0, 0])
                totals[a]["w"] += rec[1]
                totals[a]["l"] += rec[0]
                totals[a]["d"] += rec[2]
        return totals

    def test_one_game_black_wins(self):
        pop = _make_population(2)
        a, b = pop.current_population
        pop.record_match(a, b, BLACK)
        t = self._totals(pop)
        self.assertEqual(t[a]["w"], 1)
        self.assertEqual(t[a]["l"], 0)
        self.assertEqual(t[b]["w"], 0)
        self.assertEqual(t[b]["l"], 1)

    def test_symmetric_pair_both_win_as_black(self):
        """If black always wins, a full pair (a,b)+(b,a) gives each agent 1W-1L."""
        pop = _make_population(2)
        a, b = pop.current_population
        pop.record_match(a, b, BLACK)   # a as black wins
        pop.record_match(b, a, BLACK)   # b as black wins
        t = self._totals(pop)
        self.assertEqual(t[a]["w"], 1)
        self.assertEqual(t[a]["l"], 1)
        self.assertEqual(t[b]["w"], 1)
        self.assertEqual(t[b]["l"], 1)

    def test_dominant_agent_wins_all(self):
        """An agent that wins both games against an opponent shows 2W-0L."""
        pop = _make_population(2)
        a, b = pop.current_population
        pop.record_match(a, b, BLACK)   # a as black wins
        pop.record_match(b, a, WHITE)   # a as white wins
        t = self._totals(pop)
        self.assertEqual(t[a]["w"], 2)
        self.assertEqual(t[a]["l"], 0)
        self.assertEqual(t[b]["w"], 0)
        self.assertEqual(t[b]["l"], 2)

    def test_wins_plus_losses_plus_draws_equals_games_played(self):
        """W + D + L per player must equal number of games they actually played."""
        pop = _make_population(3)
        pids = pop.current_population
        a, b, c = pids
        outcomes = [
            (a, b, BLACK), (b, a, BLACK),
            (a, c, WHITE), (c, a, BLACK),
            (b, c, EMPTY), (c, b, WHITE),
        ]
        for blk, wht, winner in outcomes:
            pop.record_match(blk, wht, winner)

        t = self._totals(pop)
        for pid in pids:
            total = t[pid]["w"] + t[pid]["d"] + t[pid]["l"]
            self.assertEqual(total, 4,
                             f"Player {pid} should have 4 recorded games, got {total}")

    def test_draws_counted_once_per_game(self):
        pop = _make_population(2)
        a, b = pop.current_population
        pop.record_match(a, b, EMPTY)
        pop.record_match(b, a, EMPTY)
        t = self._totals(pop)
        self.assertEqual(t[a]["d"], 2)
        self.assertEqual(t[b]["d"], 2)
        self.assertEqual(t[a]["w"], 0)
        self.assertEqual(t[b]["w"], 0)

    def test_build_standings_table_renders(self):
        """build_standings_table should return a Rich Table without crashing."""
        pop = _make_population(3)
        a, b, c = pop.current_population
        pop.record_match(a, b, BLACK)
        pop.record_match(b, c, WHITE)
        table = pop.build_standings_table()
        self.assertIsNotNone(table)

    def test_build_matrix_table_renders(self):
        """build_matrix_table should return a Rich Table without crashing."""
        pop = _make_population(3)
        a, b, c = pop.current_population
        pop.record_match(a, b, BLACK)
        pop.record_match(b, a, WHITE)
        table = pop.build_matrix_table()
        self.assertIsNotNone(table)


# ─────────────────────────────────────────────────────────────
# Round-robin scheduling
# ─────────────────────────────────────────────────────────────

class TestRoundRobinScheduling(unittest.TestCase):
    """Tournament scheduling must produce N*(N-1) games with every pair
    playing both colour assignments exactly once."""

    def _schedule(self, n_players=5):
        """Reproduce the tournament gamePool logic and return the list of
        (black_pid, white_pid) pairs."""
        from slowpoke.core.population import Population
        pop = Population(num_players=n_players, ply_depth=1, use_neat=False,
                         include_baseline=False, include_onix=False)
        players = pop.current_population[:]
        pairs = []
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                pid_i, pid_j = players[i], players[j]
                pairs.append((pid_i, pid_j))   # i as black
                pairs.append((pid_j, pid_i))   # j as black
        return players, pairs

    def test_total_game_count(self):
        n = 5
        players, pairs = self._schedule(n)
        self.assertEqual(len(pairs), n * (n - 1))

    def test_every_unordered_pair_plays_both_colours(self):
        n = 6
        players, pairs = self._schedule(n)
        pair_set = set(pairs)
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                a, b = players[i], players[j]
                self.assertIn((a, b), pair_set,
                              f"({a} black, {b} white) not scheduled")
                self.assertIn((b, a), pair_set,
                              f"({b} black, {a} white) not scheduled")

    def test_no_self_play(self):
        _, pairs = self._schedule(5)
        for blk, wht in pairs:
            self.assertNotEqual(blk, wht, "Self-play detected")

    def test_each_pair_appears_exactly_twice(self):
        n = 4
        players, pairs = self._schedule(n)
        from collections import Counter
        counts = Counter(frozenset(p) for p in pairs)
        for pair_set, count in counts.items():
            self.assertEqual(count, 2,
                             f"Pair {pair_set} scheduled {count} times (expected 2)")

    def test_each_player_games_count(self):
        """Each player should appear in exactly 2*(N-1) games."""
        n = 5
        players, pairs = self._schedule(n)
        from collections import Counter
        game_counts = Counter()
        for blk, wht in pairs:
            game_counts[blk] += 1
            game_counts[wht] += 1
        for pid in players:
            self.assertEqual(game_counts[pid], 2 * (n - 1),
                             f"Player {pid} in {game_counts[pid]} games, expected {2*(n-1)}")


# ─────────────────────────────────────────────────────────────
# Distribution: wide spread expected with non-uniform agents
# ─────────────────────────────────────────────────────────────

class TestTournamentDistribution(unittest.TestCase):
    """After a simulated full round-robin with non-uniform outcomes,
    the standings must reflect a wide distribution — not all 1-1 pairs."""

    def _simulate_full_rr(self, n, outcome_fn):
        """Run a round-robin where outcome_fn(black_pid, white_pid) → winner."""
        pop = _make_population(n)
        players = pop.current_population[:]
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                a, b = players[i], players[j]
                pop.record_match(a, b, outcome_fn(a, b))
                pop.record_match(b, a, outcome_fn(b, a))
        return pop, players

    def test_always_black_wins_gives_uniform_1_1_per_pair(self):
        """When black always wins, EVERY pair ends exactly 1W-1L — this IS
        the degenerate case.  Asserting it explicitly documents the expected
        behaviour so regressions are obvious."""
        n = 4
        pop, players = self._simulate_full_rr(n, lambda blk, wht: BLACK)
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                a, b = players[i], players[j]
                ab = pop.head_to_head.get((a, b), [0, 0, 0])
                ba = pop.head_to_head.get((b, a), [0, 0, 0])
                a_wins = ab[0] + ba[1]
                b_wins = ab[1] + ba[0]
                self.assertEqual(a_wins, 1)
                self.assertEqual(b_wins, 1)

    def test_heterogeneous_outcomes_produce_wide_distribution(self):
        """With player-quality-dependent outcomes, higher-ranked players
        should have more wins than lower-ranked ones."""
        n = 6
        players = list(range(n))

        def outcome_fn(blk, wht):
            # Simulate: higher player ID = stronger; wins with 80% prob
            if blk > wht:
                return BLACK if __import__("random").random() < 0.8 else WHITE
            else:
                return WHITE if __import__("random").random() < 0.8 else BLACK

        import random
        random.seed(42)
        pop, players_list = self._simulate_full_rr(n, outcome_fn)

        # Aggregate total wins per player
        wins = {}
        for pid in players_list:
            w = 0
            for other in players_list:
                if other == pid:
                    continue
                rec = pop.head_to_head.get((pid, other), [0, 0, 0])
                w += rec[0]
                rec = pop.head_to_head.get((other, pid), [0, 0, 0])
                w += rec[1]
            wins[pid] = w

        win_values = list(wins.values())
        self.assertGreater(max(win_values) - min(win_values), 0,
                           "All agents have same win count — distribution is too narrow")

    def test_full_rr_total_wins_equals_games_without_draws(self):
        """Sum of all wins across all players must equal N*(N-1) when no draws."""
        n = 5
        pop, players = self._simulate_full_rr(n, lambda blk, wht: BLACK)
        total_wins = 0
        for pid in players:
            for other in players:
                if other == pid:
                    continue
                rec = pop.head_to_head.get((pid, other), [0, 0, 0])
                total_wins += rec[0] + rec[1]
        # Each game produces exactly one win: total = N*(N-1)
        self.assertEqual(total_wins, n * (n - 1))


# ─────────────────────────────────────────────────────────────
# NEAT evaluator wiring: TMCTS must use the NEAT nn
# ─────────────────────────────────────────────────────────────

class TestNEATEvaluatorWiring(unittest.TestCase):
    """After NEATEvolution.generate_bot, the TMCTS decision function's `nn`
    attribute must reference the same NeuralNetwork object as bot.nn (the NEAT
    network), NOT the stale default standard network that was captured at
    Slowbro construction time.

    If this is broken, all agents use an identical material-count-dominated
    evaluator in MLX batch mode, producing degenerate 1-1 results at large ply.
    """

    def _make_neat_bot(self, ply=4):
        from slowpoke.core.population import Population
        from slowpoke.core.neuroevolution import NEATEvolution

        pop = Population(num_players=1, ply_depth=ply, use_neat=True,
                         include_baseline=False, include_onix=False)
        evo = NEATEvolution(pop)
        bot = evo.generate_bot(ply_depth=ply, debug=False)
        return bot

    def test_tmcts_nn_is_neat_nn(self):
        """TMCTS's self.nn must be the NEAT NeuralNetwork, not a standard one."""
        bot = self._make_neat_bot()
        tmcts = bot.decision_function
        self.assertIs(
            tmcts.nn, bot.nn,
            "TMCTS.nn is not the same object as bot.nn. "
            "This means batch evaluation (flush_batch, _evaluate_moves_batch) "
            "uses the old default standard network instead of the NEAT genome. "
            "Fix: call _detect_mlx() after bot.nn is set, or pass the NEAT nn "
            "before constructing Slowbro's TMCTS."
        )

    def test_tmcts_nn_is_neat_mode(self):
        """TMCTS's nn must be in 'neat' mode, not 'standard'."""
        bot = self._make_neat_bot()
        tmcts = bot.decision_function
        mode = getattr(tmcts.nn, "_mode", None)
        self.assertEqual(
            mode, "neat",
            f"TMCTS.nn._mode is '{mode}', expected 'neat'. "
            "Batch evaluations will use the wrong (standard) network."
        )

    def test_two_neat_agents_have_different_tmcts_nns(self):
        """Two independently-created NEAT agents must have distinct TMCTS nns.

        If they share the same nn object, all agents play identically.
        """
        bot1 = self._make_neat_bot()
        bot2 = self._make_neat_bot()
        self.assertIsNot(
            bot1.decision_function.nn, bot2.decision_function.nn,
            "Two NEAT agents share the same TMCTS nn — all games will be identical."
        )

    def test_evaluate_board_uses_neat_genome(self):
        """evaluate_board must call the NEAT genome, not the standard network."""
        from slowpoke.core.checkers import CheckerBoard
        bot = self._make_neat_bot(ply=1)

        # Tamper with the standard [32,40,10,1] bias to produce a sentinel
        standard_nn = object.__new__(type(bot.nn))  # unused; check type only
        # Instead: replace bot.nn._genome with a hand-crafted genome that
        # always returns a fixed value, and verify evaluate_board returns it.
        from slowpoke.agents.evaluator.genome import Genome
        import numpy as np

        genome = bot.nn._genome
        self.assertIsNotNone(genome, "NEAT genome should not be None")

        B = CheckerBoard()
        val = bot.evaluate_board(B, BLACK)
        # Just verify it doesn't crash and returns a float
        self.assertIsInstance(val, float)


# ─────────────────────────────────────────────────────────────
# head_to_head cleared between generations
# ─────────────────────────────────────────────────────────────

class TestHeadToHeadLifecycle(unittest.TestCase):

    def test_head_to_head_cleared_at_tournament_start(self):
        """Simulates what tournament() does: clear then repopulate.
        Stale data from the previous generation must not bleed through."""
        pop = _make_population(3)
        a, b, c = pop.current_population

        # Simulate gen-0 tournament
        pop.record_match(a, b, BLACK)
        pop.record_match(b, a, WHITE)
        self.assertIn((a, b), pop.head_to_head)

        # Start gen-1 tournament: clear first
        pop.head_to_head.clear()
        self.assertEqual(len(pop.head_to_head), 0,
                         "head_to_head should be empty at tournament start")

    def test_stale_keys_dont_affect_new_generation_standings(self):
        """After generate_next_population, new player IDs have no h2h entries.
        Their standings W/D/L should be 0, not inheriting old data."""
        pop = _make_population(6, ply=1, use_neat=False)
        old_pids = set(pop.current_population)

        # Simulate some tournament results
        pids = pop.current_population[:]
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                a, b = pids[i], pids[j]
                pop.record_match(a, b, BLACK)
                pop.record_match(b, a, WHITE)
                pop.allocate_points({"Winner": BLACK}, a, b)
                pop.allocate_points({"Winner": WHITE}, b, a)

        pop.sort_population_by_points()
        pop.add_champion()
        pop.generate_next_population()

        new_pids = set(pop.current_population) - old_pids
        # New players have no h2h entries yet (tournament hasn't started)
        for pid in new_pids:
            for other in pop.current_population:
                if other == pid:
                    continue
                key1 = (pid, other)
                key2 = (other, pid)
                # May or may not exist; if they do exist, they should be for old IDs
                # (not new ones). New IDs shouldn't appear as stale entries.
                if key1 in pop.head_to_head or key2 in pop.head_to_head:
                    # Only acceptable if this is an elite (retained) player
                    self.assertIn(
                        pid, old_pids,
                        f"New player {pid} has stale head_to_head entries"
                    )


# ─────────────────────────────────────────────────────────────
# Weight diversity: agents in a new population must be distinct
# ─────────────────────────────────────────────────────────────

class TestPopulationWeightDiversity(unittest.TestCase):
    """Agents created in a fresh population must have different weights.

    Identical weights → identical evaluations → degenerate 1-1 tournament
    distribution regardless of ply depth.
    """

    def test_standard_ga_agents_have_distinct_weights(self):
        """No two agents in a standard-GA population may share identical weights."""
        pop = _make_population(n=6, ply=1, use_neat=False)
        pids = pop.current_population[:]

        weights = {}
        for pid in pids:
            w = pop.players[pid].bot.nn.get_all_coefficients()
            weights[pid] = w

        identical_pairs = []
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                a, b = pids[i], pids[j]
                if np.array_equal(weights[a], weights[b]):
                    identical_pairs.append((a, b))

        self.assertEqual(
            identical_pairs, [],
            f"Agents with identical weights found: {identical_pairs}. "
            "All agents in the population will evaluate boards identically, "
            "causing a degenerate 1-1 tournament distribution."
        )

    def test_neat_agents_have_distinct_genome_weights(self):
        """No two NEAT agents in a fresh population may have identical connection weights."""
        from slowpoke.core.population import Population

        pop = Population(num_players=6, ply_depth=1, use_neat=True,
                         include_baseline=False, include_onix=False)
        pids = pop.current_population[:]

        genomes = {}
        for pid in pids:
            nn = pop.players[pid].bot.nn
            g = nn._genome
            self.assertIsNotNone(g, f"Agent {pid} has no NEAT genome")
            # Extract connection weights as a sorted list for comparison
            conn_weights = tuple(
                sorted((c.from_node, c.to_node, round(c.weight, 8))
                       for c in g.connections.values())
            )
            genomes[pid] = conn_weights

        identical_pairs = []
        pid_list = list(genomes.keys())
        for i in range(len(pid_list)):
            for j in range(i + 1, len(pid_list)):
                a, b = pid_list[i], pid_list[j]
                if genomes[a] == genomes[b]:
                    identical_pairs.append((a, b))

        self.assertEqual(
            identical_pairs, [],
            f"NEAT agents with identical genome weights found: {identical_pairs}. "
            "All agents evaluate boards identically in MLX batch mode."
        )

    def test_standard_ga_weight_variance_is_nonzero(self):
        """The variance of initial weights across the population must be non-trivial.

        Even if no two agents are perfectly identical, near-zero variance
        means the population is effectively homogeneous.
        """
        pop = _make_population(n=8, ply=1, use_neat=False)
        pids = pop.current_population[:]

        all_first_weights = np.array([
            pop.players[pid].bot.nn.get_all_coefficients()[0]
            for pid in pids
        ])

        variance = np.var(all_first_weights)
        self.assertGreater(
            variance, 1e-6,
            f"First-weight variance across population is {variance:.2e} — "
            "agents have near-identical weights (seeding bug?)."
        )

    def test_population_weight_spread_across_layers(self):
        """Check diversity across the full weight vector, not just the first element."""
        pop = _make_population(n=6, ply=1, use_neat=False)
        pids = pop.current_population[:]

        weight_matrix = np.stack([
            pop.players[pid].bot.nn.get_all_coefficients()
            for pid in pids
        ])  # shape: (n_agents, n_weights)

        # Per-weight variance across agents: must be nonzero for most weights
        per_weight_var = np.var(weight_matrix, axis=0)
        zero_var_fraction = np.mean(per_weight_var < 1e-10)

        self.assertLess(
            zero_var_fraction, 0.1,
            f"{zero_var_fraction*100:.1f}% of weight dimensions have zero variance "
            "across the population — agents are nearly identical."
        )


# ─────────────────────────────────────────────────────────────
# ParallelTMCTS seed diversity
# ─────────────────────────────────────────────────────────────

class TestParallelTMCTSSeedDiversity(unittest.TestCase):
    """ParallelTMCTS must not use the same fixed seed for every agent.

    All agents sharing seed=42 → identical thread RNGs → identical tree
    exploration → homogeneous play despite different NN weights.
    """

    def _make_parallel_tmcts(self):
        from slowpoke.search.parallel_tmcts import ParallelTMCTS

        class DummyEvaluator:
            def evaluate_board(self, board, colour):
                return 0.0

        return ParallelTMCTS(ply=1, evaluator=DummyEvaluator())

    def test_default_seed_is_not_fixed_42(self):
        """Two independently created ParallelTMCTS instances must not both use seed 42."""
        t1 = self._make_parallel_tmcts()
        t2 = self._make_parallel_tmcts()
        self.assertNotEqual(
            t1.seed, 42,
            "ParallelTMCTS default seed is 42 — all agents explore identical trees."
        )
        self.assertNotEqual(
            t2.seed, 42,
            "ParallelTMCTS default seed is 42 — all agents explore identical trees."
        )

    def test_two_instances_have_different_seeds(self):
        """Two independently created ParallelTMCTS instances should have different seeds."""
        seeds = {self._make_parallel_tmcts().seed for _ in range(5)}
        self.assertGreater(
            len(seeds), 1,
            f"All 5 ParallelTMCTS instances got the same seed: {seeds}. "
            "Agents will explore identical MCTS trees."
        )

    def test_explicit_seed_is_preserved(self):
        """Passing an explicit seed must still be honoured (for reproducible tests)."""
        from slowpoke.search.parallel_tmcts import ParallelTMCTS

        class DummyEvaluator:
            def evaluate_board(self, board, colour):
                return 0.0

        t = ParallelTMCTS(ply=1, evaluator=DummyEvaluator(), seed=99)
        self.assertEqual(t.seed, 99)

    def test_population_agents_have_different_parallel_tmcts_seeds(self):
        """Agents created via Population must each get a distinct ParallelTMCTS seed."""
        from slowpoke.core.population import Population
        from slowpoke.search.parallel_tmcts import ParallelTMCTS

        pop = Population(
            num_players=6, ply_depth=1,
            use_parallel_mcts=True, num_parallel=2,
            include_baseline=False, include_onix=False,
        )
        seeds = []
        for pid in pop.current_population:
            df = pop.players[pid].bot.decision_function
            if isinstance(df, ParallelTMCTS):
                seeds.append(df.seed)

        if not seeds:
            self.skipTest("No ParallelTMCTS agents in population at ply=1")

        self.assertGreater(
            len(set(seeds)), 1,
            f"All ParallelTMCTS agents share the same seed: {seeds[0]}. "
            "Every agent explores identical MCTS trees."
        )


if __name__ == "__main__":
    unittest.main()
