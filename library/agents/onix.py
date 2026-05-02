"""
Onix — heuristic-based checkers bot. No neural network, no MLX.

Uses hand-crafted heuristics (material, kings, advancement, centre,
mobility, back-rank defence) evaluated directly from bitboards.
Search is powered by serial TMCTS at the configured ply depth.
"""

import os
import sys
import math

_lib_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _lib_dir not in sys.path:
    sys.path.insert(0, _lib_dir)

from agents import minimax_draw, minimax_empty, minimax_lose, minimax_win

# Bitboard masks for each of the 4 rows of playable squares
# Row 0: bits 0-7  |  Row 1: bits 9-16  |  Row 2: bits 18-25  |  Row 3: bits 27-34
_ROW_MASKS = [0xFF, 0x1FE00, 0x3FC0000, 0x7F8000000]

# Centre squares (bit positions 12, 13, 21, 22)
_CENTRE_MASK = (1 << 12) | (1 << 13) | (1 << 21) | (1 << 22)


class Onix:
    """Heuristic-based checkers bot with TMCTS search."""

    def __init__(self, plyDepth=4, debug=False):
        self.ply = plyDepth
        self.debug = debug
        self.enableCache = False
        self.cache = {}

        from decision.tmcts import TMCTS
        self.decisionFunction = TMCTS(plyDepth, self, debug=debug)

    def move_function(self, board, colour):
        """Entry point called by Agent.make_move()."""
        return self.decisionFunction.Decide(board, colour)

    def evaluate_board(self, board, colour):
        """Heuristic evaluation from colour's perspective.

        Returns a float in [-1, 1] where positive means good for colour.
        """
        if board.is_over():
            if board.winner != minimax_empty:
                return minimax_win if board.winner == colour else minimax_lose
            return minimax_draw

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
