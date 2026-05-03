"""MCTS — Monte Carlo Tree Search with UCB1."""

from __future__ import annotations

import datetime
import math
import random
from typing import Any, Dict, Optional

import numpy as np

from core.constants import MLX_AVAILABLE, mx
from core.constants import MLX_AVAILABLE, mx


class MCTS:
    """Monte Carlo Tree Search with UCB1 move selection."""

    def __init__(
        self,
        ply: int,
        evaluator: Optional[Any] = None,
        debug: bool = False,
        batch_size: int = 512,
    ) -> None:
        """Initialise MCTS.

        Args:
            ply: Search depth (used as time budget in seconds if > 0).
            evaluator: Board evaluation function.
            debug: Enable debug output.
            batch_size: Positions to accumulate before batch eval.
        """
        self.ply = ply
        self.evaluator = evaluator
        self.c = 1.4
        self.debug = debug
        self.batch_size = batch_size
        self.use_mlx = False
        if evaluator is not None and hasattr(evaluator, "nn"):
            self.use_mlx = getattr(evaluator.nn, "_use_mlx", False)
        self.mcts_plays: Dict[Any, int] = {}
        self.mcts_chances: Dict[Any, float] = {}

    def decide(self, B: Any, colour: int) -> Optional[int]:
        """Return the best move found by MCTS search."""
        return self.mcts_code(B, self.ply, colour)

    def mcts_code(self, B: Any, ply: int, colour: int) -> Optional[int]:
        """Run MCTS for a given number of seconds based on ply.

        Args:
            B: Board state.
            ply: Search depth (time budget = ply seconds).
            colour: Current player colour.

        Returns:
            Best move, or None if no moves available.
        """
        if self.debug:
            print("I AM:", colour)
        moves = B.get_moves()
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]

        self.mcts_plays, self.mcts_chances = {}, {}
        seconds = self.ply
        self.calculation_time = datetime.timedelta(seconds=seconds)

        max_rounds = 1
        if self.ply > 0:
            max_rounds = 200 * self.ply

        begin = datetime.datetime.utcnow()
        ply_depth = 100 * self.ply
        while datetime.datetime.utcnow() - begin < self.calculation_time:
            self.mcts_simulate(B, ply_depth, colour, max_rounds)

        move_states = []
        for i in moves:
            B.push_move(i)
            FEN_hash = hash(B.pdn["FEN"])
            move_states.append((i, FEN_hash))
            B.pop_move()

        percent_winchance, best_move = max(
            (
                self.mcts_chances.get((colour, S), 0)
                / self.mcts_plays.get((colour, S), 1),
                p,
            )
            for p, S in move_states
        )

        if colour == 1:
            percent_winchance, best_move = min(
                (
                    self.mcts_chances.get((colour, S), 0)
                    / self.mcts_plays.get((colour, S), 1),
                    p,
                )
                for p, S in move_states
            )

        if self.debug:
            goods = sorted(
                (
                    (
                        100
                        * self.mcts_chances.get((colour, S), 0)
                        / self.mcts_plays.get((colour, S), 1),
                        self.mcts_chances.get((colour, S), 0),
                        self.mcts_plays.get((colour, S), 0),
                        p,
                    )
                    for p, S in move_states
                ),
                reverse=True,
            )
            for i in goods:
                print(i[3], "Moves:", i[2], "Good Moves", i[1], str(i[0]) + "%")
            print("Maximum depth searched:", ply_depth)
            print(percent_winchance)

        return best_move

    def mcts_simulate(self, B: Any, ply: int, colour: int, rounds: int) -> None:
        """Simulate random games for MCTS statistics.

        Args:
            B: Board state.
            ply: Search depth limit.
            colour: Starting player colour.
            rounds: Number of rounds to simulate.
        """
        visited_states = set()
        player = colour
        move_stack = []

        expand = True
        winner = -1
        current_ply = ply

        position_batch = []
        batch_refs = []

        for t in range(1, rounds + 1):
            legal_moves = B.get_moves()
            if not legal_moves:
                break

            move_states = []
            for i in legal_moves:
                B.push_move(i)
                FEN_hash = hash(B.pdn["FEN"])
                move_states.append((i, FEN_hash))
                B.pop_move()

            if all(self.mcts_plays.get((player, S)) for p, S in move_states):
                all_move_states = [self.mcts_plays[(player, S)] for p, S in move_states]
                log_total = math.log(sum(all_move_states))
                value, move, FEN_hash = max(
                    (
                        self.mcts_chances[(player, S)] / self.mcts_plays[(player, S)]
                        + self.c * math.sqrt(log_total / self.mcts_plays[(player, S)]),
                        p,
                        S,
                    )
                    for p, S in move_states
                )
                B.push_move(move)
                move_stack.append(move)
            else:
                choice = random.choice(move_states)
                move, FEN_hash = choice
                B.push_move(move)
                move_stack.append(move)

            if B.is_over():
                winner = B.winner
                break

            su = FEN_hash
            if expand and (player, su) not in self.mcts_plays:
                expand = False
                self.mcts_plays[(player, su)] = 0
                self.mcts_chances[(player, su)] = 0
                if t > current_ply:
                    current_ply = t

                if self.use_mlx and self.evaluator:
                    boardStatus = B.get_board_pos_weighted(
                        B.current_player(),
                        {
                            "Black": 1,
                            "White": -1,
                            "empty": 0,
                            "blackKing": 1.5,
                            "whiteKing": -1.5,
                        },
                    )
                    if self.evaluator.layers[0] == 91:
                        boardStatus = self.evaluator.nn.subsquares(boardStatus)
                    position_batch.append(np.array(boardStatus, dtype=np.float32))
                    batch_refs.append((player, su))

            visited_states.add((player, su))
            player = B.current_player()
            if B.is_over():
                winner = B.winner

        if self.use_mlx and self.evaluator and len(position_batch) > 0:
            batch_results = self.evaluator.nn.compute_batch_mlx(position_batch)
            for (player, su), result in zip(batch_refs, batch_results):
                if (player, su) in self.mcts_plays:
                    self.mcts_plays[(player, su)] = 1
                    self.mcts_chances[(player, su)] = float(result)

        for p, x in visited_states:
            if (p, x) not in self.mcts_plays:
                continue
            self.mcts_plays[(p, x)] += 1
            if p == winner:
                self.mcts_chances[(p, x)] += 1

        for _ in range(len(move_stack)):
            B.pop_move()
