"""Shared constants for the Slowpoke codebase."""

from __future__ import annotations

from typing import Dict

# ── MLX availability ──────────────────────────────────────────────────────────
MLX_AVAILABLE: bool = False
mx = None
try:
    import mlx.core as mx  # noqa: F401

    MLX_AVAILABLE = True
except ImportError:
    pass

# ── Piece colours ─────────────────────────────────────────────────────────────
BLACK: int = 0
WHITE: int = 1
EMPTY: int = -1
BLACK_KING: int = 2
WHITE_KING: int = 3

# ── Minimax values ────────────────────────────────────────────────────────────
MINIMAX_WIN: int = 1
MINIMAX_LOSE: int = -1
MINIMAX_DRAW: int = 0
MINIMAX_EMPTY: int = -1

# ── Tournament points ─────────────────────────────────────────────────────────
WIN_PT: int = 2
DRAW_PT: int = 0
LOSE_PT: int = -1
CHAMP_WIN_PT: int = 1
CHAMP_DRAW_PT: int = 0
CHAMP_LOSE_PT: int = -1

# ── Board rules ───────────────────────────────────────────────────────────────
BORING_NO_EAT_LIMIT: int = 50
REPETITION_LIMITS: int = 12
UNUSED_BITS: int = 0b100000000100000000100000000100000000

# ── Default piece weights for NN evaluation ───────────────────────────────────
PIECE_WEIGHTS: Dict[str, float] = {
    "Black": 1.0,
    "White": -1.0,
    "empty": 0.0,
    "blackKing": 1.5,
    "whiteKing": -1.5,
}
