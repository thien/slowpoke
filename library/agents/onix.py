"""Onix — heuristic-based checkers bot with TMCTS search."""

from __future__ import annotations

import math
from typing import Any, Dict

from core.constants import MINIMAX_WIN, MINIMAX_LOSE, MINIMAX_DRAW, MINIMAX_EMPTY

_ROW_MASKS: list = [0xFF, 0x1FE00, 0x3FC0000, 0x7F8000000]
_CENTRE_MASK: int = (1 << 12) | (1 << 13) | (1 << 21) | (1 << 22)


class Onix:
    """Heuristic-based checkers bot with TMCTS search."""

    def __init__(self, ply_depth: int = 4, debug: bool = False) -> None:
        self.ply = ply_depth
        self.debug = debug
        self.enable_cache = False
        self.cache: Dict[Any, float] = {}

        from decision.tmcts import TMCTS

        self.decision_function = TMCTS(ply_depth, self, debug=debug)

    def move_function(self, board: Any, colour: int) -> int:
        """Entry point called by Agent.make_move()."""
        return self.decision_function.decide(board, colour)

    def evaluate_board(self, board: Any, colour: int) -> float:
        """Heuristic evaluation from colour's perspective.

        Returns a float in [-1, 1] where positive means good for colour.
        """
        if board.is_over():
            if board.winner != MINIMAX_EMPTY:
                return MINIMAX_WIN if board.winner == colour else MINIMAX_LOSE
            return MINIMAX_DRAW

        # Read bitboards — Rust backend or Python fallback
        if board._has_core:
            pieces = board._core.get_pieces()
            fwd = board._core.get_forward()
            bwd = board._core.get_backward()
        else:
            pieces = board.pieces
            fwd = board.forward
            bwd = board.backward

        opp = 1 - colour
        my_pieces = pieces[colour]
        opp_pieces = pieces[opp]
        my_kings = bwd[0] if colour == 0 else fwd[1]
        opp_kings = bwd[1] if opp == 0 else fwd[0]
        my_men = my_pieces ^ my_kings
        opp_men = opp_pieces ^ opp_kings

        score = 0

        # ── Material: men worth 1, kings worth 2 ──
        my_mat = my_men.bit_count() + 2 * my_kings.bit_count()
        opp_mat = opp_men.bit_count() + 2 * opp_kings.bit_count()
        score += 100 * (my_mat - opp_mat)

        # ── Advancement: each row advanced = 10 points per piece ──
        my_adv = 0
        opp_adv = 0
        for row, mask in enumerate(_ROW_MASKS):
            if colour == 0:  # Black advances toward row 3
                my_adv += (my_men & mask).bit_count() * row
                opp_adv += (opp_men & mask).bit_count() * row
            else:  # White advances toward row 0
                my_adv += (my_men & mask).bit_count() * (3 - row)
                opp_adv += (opp_men & mask).bit_count() * (3 - row)
        score += 10 * (my_adv - opp_adv)

        # ── Centre control ──
        my_centre = (my_pieces & _CENTRE_MASK).bit_count()
        opp_centre = (opp_pieces & _CENTRE_MASK).bit_count()
        score += 15 * (my_centre - opp_centre)

        # ── Mobility: number of legal moves ──
        score += 5 * len(board.get_moves())

        # ── Back-rank defence: pieces protecting the back row ──
        my_back_mask = _ROW_MASKS[0] if colour == 0 else _ROW_MASKS[3]
        opp_back_mask = _ROW_MASKS[3] if colour == 0 else _ROW_MASKS[0]
        my_back = (my_pieces & my_back_mask).bit_count()
        opp_back = (opp_pieces & opp_back_mask).bit_count()
        score += 30 * (my_back - opp_back)

        # Normalise to [-1, 1] via tanh
        return math.tanh(score / 200)
