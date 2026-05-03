"""MCTS — Monte Carlo Tree Search with UCB1 (time-budgeted).

Inherits ``is_over``, ``_extract_position``, ``_detect_mlx``, ``_ucb1_score``
from ``MCTSBase``.
"""

from __future__ import annotations

import datetime
import math
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np

from core.constants import MLX_AVAILABLE, mx
from search.base import MCTSBase


class MCTS(MCTSBase):
    """Monte Carlo Tree Search with UCB1 move selection.

    Time-budgeted: runs for ``ply`` seconds of wall-clock time.
    """

    def __init__(
        self,
        ply: int,
        evaluator: Optional[Any] = None,
        debug: bool = False,
        batch_size: int = 512,
    ) -> None:
        super().__init__(ply, evaluator, debug, batch_size)
        self.c = 1.4
        self.mcts_plays: Dict[Tuple[int, int], int] = {}
        self.mcts_chances: Dict[Tuple[int, int], float] = {}

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
        calculation_time = datetime.timedelta(seconds=self.ply)

        max_rounds = 1
        if self.ply > 0:
            max_rounds = 200 * self.ply

        begin = datetime.datetime.utcnow()
        ply_depth = 100 * self.ply
        while datetime.datetime.utcnow() - begin < calculation_time:
            self.mcts_simulate(B, ply_depth, colour, max_rounds)

        move_states = []
        for m in moves:
            B.push_move(m)
            fen_hash = hash(B.pdn["FEN"])
            move_states.append((m, fen_hash))
            B.pop_move()

        if colour == 0:
            _, best_move = max(
                (self.mcts_chances.get((colour, s), 0) / max(self.mcts_plays.get((colour, s), 1), 1), m)
                for m, s in move_states
            )
        else:
            _, best_move = min(
                (self.mcts_chances.get((colour, s), 0) / max(self.mcts_plays.get((colour, s), 1), 1), m)
                for m, s in move_states
            )

        if self.debug:
            goods = sorted(
                (
                    (
                        100 * self.mcts_chances.get((colour, s), 0) / max(self.mcts_plays.get((colour, s), 1), 1),
                        self.mcts_chances.get((colour, s), 0),
                        self.mcts_plays.get((colour, s), 0),
                        m,
                    )
                    for m, s in move_states
                ),
                reverse=True,
            )
            for g in goods:
                print(g[3], "Moves:", g[2], "Good Moves", g[1], str(g[0]) + "%")
            print("Maximum depth searched:", ply_depth)

        return best_move

    def mcts_simulate(self, B: Any, max_ply: int, colour: int, rounds: int) -> None:
        """Simulate random games for MCTS statistics.

        Args:
            B: Board state.
            max_ply: Search depth limit.
            colour: Starting player colour.
            rounds: Number of rounds to simulate.
        """
        visited_states = set()
        player = colour
        move_stack = []

        expand = True
        winner = -1

        position_batch = []
        batch_refs = []

        for t in range(1, rounds + 1):
            legal_moves = B.get_moves()
            if not legal_moves:
                break

            move_states = []
            for m in legal_moves:
                B.push_move(m)
                fen_hash = hash(B.pdn["FEN"])
                move_states.append((m, fen_hash))
                B.pop_move()

            if all(self.mcts_plays.get((player, s)) for _, s in move_states):
                play_values = [self.mcts_plays[(player, s)] for _, s in move_states]
                log_total = math.log(sum(play_values))
                _, move, fen_hash = max(
                    (
                        self.mcts_chances[(player, s)] / self.mcts_plays[(player, s)]
                        + self.c * math.sqrt(log_total / self.mcts_plays[(player, s)]),
                        m,
                        s,
                    )
                    for m, s in move_states
                )
                B.push_move(move)
                move_stack.append(move)
            else:
                choice = random.choice(move_states)
                move, fen_hash = choice
                B.push_move(move)
                move_stack.append(move)

            if B.is_over():
                winner = B.winner
                break

            su = fen_hash
            if expand and (player, su) not in self.mcts_plays:
                expand = False
                self.mcts_plays[(player, su)] = 0
                self.mcts_chances[(player, su)] = 0

                if self.use_mlx and self.evaluator:
                    board_status = B.get_board_pos_weighted(
                        B.current_player(),
                        {"Black": 1, "White": -1, "empty": 0, "blackKing": 1.5, "whiteKing": -1.5},
                    )
                    if hasattr(self.evaluator, "layers") and self.evaluator.layers[0] == 91:
                        board_status = self.evaluator.nn.subsquares(board_status)
                    position_batch.append(np.array(board_status, dtype=np.float32))
                    batch_refs.append((player, su))

            visited_states.add((player, su))
            player = B.current_player()
            if B.is_over():
                winner = B.winner

        if self.use_mlx and self.evaluator and position_batch:
            batch_results = self.evaluator.nn.compute_batch_mlx(position_batch)
            for (pl, st), result in zip(batch_refs, batch_results):
                if (pl, st) in self.mcts_plays:
                    self.mcts_plays[(pl, st)] = 1
                    self.mcts_chances[(pl, st)] = float(result)

        for p, x in visited_states:
            if (p, x) not in self.mcts_plays:
                continue
            self.mcts_plays[(p, x)] += 1
            if p == winner:
                self.mcts_chances[(p, x)] += 1

        for _ in range(len(move_stack)):
            B.pop_move()
