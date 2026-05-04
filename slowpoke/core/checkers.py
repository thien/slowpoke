"""CheckerBoard — wraps the Rust checkers_core extension.

All game state and logic lives in the Rust backend. This Python class
holds PDN metadata and provides ASCII board display.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from termcolor import colored

from checkers_core import CheckerBoard as _RustCB

from slowpoke.core.constants import (
    BLACK,
    WHITE,
    EMPTY,
    BLACK_KING,
    WHITE_KING,
    BORING_NO_EAT_LIMIT,
    REPETITION_LIMITS,
    UNUSED_BITS,
)

# Backward-compat aliases
Black, White = BLACK, WHITE
empty = EMPTY
blackKing = BLACK_KING
whiteKing = WHITE_KING
boring_no_eat_limit = BORING_NO_EAT_LIMIT
repetition_limits = REPETITION_LIMITS
unused_bits = UNUSED_BITS


def _reconstruct_board(
    core_state: Dict[str, Any], pdn: Dict[str, Any]
) -> "CheckerBoard":
    """Recreate a CheckerBoard from a pickled state + pdn."""
    B = object.__new__(CheckerBoard)
    B._core = _RustCB()
    B._core.__setstate__(core_state)
    B.pdn = pdn
    B.state = [[None for _ in range(8)] for _ in range(4)]
    B.black_pieces = []
    B.white_pieces = []
    return B


class CheckerBoard:
    """Wraps a Rust CheckerBoard with PDN metadata and display helpers."""

    __slots__ = ("_core", "pdn", "state", "black_pieces", "white_pieces")

    def __init__(self) -> None:
        self._core = _RustCB()
        self.pdn = self._init_pgn()
        self.state = [[None for _ in range(8)] for _ in range(4)]
        self.black_pieces = []
        self.white_pieces = []
        self.update_state()

    # ── PDN metadata ──

    def _init_pgn(self) -> Dict[str, Any]:
        """Initialise the PGN dictionary for game export."""
        now = datetime.datetime.now()
        return {
            "Event": "Some Event",
            "Date": now.strftime("%y/%m/%d"),
            "Time": now.strftime("%H:%M:%S"),
            "Result": "*",
            "FEN": "B:W21-32:B1-16",
            "Moves": [],
        }

    def print_pgn(self) -> Dict[str, Any]:
        return self.pdn

    def set_id(self, game_id: str) -> None:
        self.pdn["_id"] = game_id

    def set_colours(self, black_id: str, white_id: str) -> None:
        self.pdn["Black"] = black_id
        self.pdn["White"] = white_id

    # ── Rust-backed properties ──

    @property
    def turn_count(self) -> int:
        return self._core.get_turn_count()

    @property
    def active(self) -> int:
        return self._core.get_active()

    @property
    def passive(self) -> int:
        return self._core.get_passive()

    @property
    def winner(self) -> Optional[int]:
        w = self._core.get_winner()
        if w == -1:
            return None
        if w == -2:
            return EMPTY  # convert Rust draw (-2) to Python convention (-1)
        return w

    @winner.setter
    def winner(self, value: Optional[int]) -> None:
        if value is None:
            self._core.set_winner(-1)
        elif value == EMPTY:
            self._core.set_winner(-2)
        else:
            self._core.set_winner(value)

    @property
    def jump(self) -> bool:
        return self._core.get_jump_flag()

    @property
    def no_eat_count(self) -> int:
        return self._core.get_no_eat_count()

    @no_eat_count.setter
    def no_eat_count(self, value: int) -> None:
        self._core.set_no_eat_count(value)

    @property
    def moves(self) -> List[Tuple[int, str]]:
        return self._core.get_move_log()

    # ── Game lifecycle ──

    def new_game(self) -> None:
        self._core.new_game()
        self.pdn = self._init_pgn()
        self.state = [[None for _ in range(8)] for _ in range(4)]
        self.black_pieces = []
        self.white_pieces = []

    def make_move(self, move: int, full_update: bool = True) -> "CheckerBoard":
        self._core.make_move(move)
        if full_update:
            self.update_state()
        return self

    def push_move(self, move: int) -> None:
        self._core.push_move(move)

    def pop_move(self) -> None:
        self._core.pop_move()

    def get_moves(self) -> List[int]:
        return self._core.get_moves()

    def get_move_strings(self) -> List[str]:
        return list(self._core.get_move_strings())

    def is_over(self, check_repetition: bool = True) -> bool:
        result = self._core.is_over(check_repetition)
        if result:
            self._update_pdn_result()
        return result

    def check_winner(self) -> None:
        self._update_pdn_result()

    def _update_pdn_result(self) -> None:
        w = self._core.get_winner()
        if w == -2:
            self.pdn["Winner"] = EMPTY
            self.pdn["Result"] = "1/2-1/2"
        elif w == BLACK:
            self.pdn["Winner"] = BLACK
            self.pdn["Result"] = "1-0"
        elif w == WHITE:
            self.pdn["Winner"] = WHITE
            self.pdn["Result"] = "0-1"
        # replay is populated lazily when B.pdn is returned (see game.py)

    # ── Board evaluation (NN input) ──

    def get_board_pos(self, colour: int) -> List[int]:
        raw = self._core.get_rank()
        return [int(x) for x in raw]

    def get_board_pos_weighted(
        self, colour: int, weights: Dict[str, float]
    ) -> np.ndarray:
        return self._core.get_board_pos_weighted(
            colour,
            weights["empty"],
            weights["Black"],
            weights["White"],
            weights["blackKing"],
            weights["whiteKing"],
        )

    # ── Utilities ──

    def copy(self) -> "CheckerBoard":
        B = object.__new__(CheckerBoard)
        B._core = self._core.copy()
        B.pdn = dict(self.pdn) if self.pdn else None
        B.state = [row[:] for row in self.state] if self.state else None
        B.black_pieces = list(self.black_pieces) if self.black_pieces else []
        B.white_pieces = list(self.white_pieces) if self.white_pieces else []
        return B

    def __reduce__(self) -> Tuple[Any, Tuple[Dict[str, Any], Dict[str, Any]]]:
        return (_reconstruct_board, (self._core.__getstate__(), self.pdn))

    def current_player(self, board: Optional["CheckerBoard"] = None) -> int:
        if board:
            return board.current_player()
        return BLACK if self.turn_count % 2 == 0 else WHITE

    def get_winner_message(self) -> None:
        w = self._core.get_winner()
        if w == BLACK:
            print("Congrats Black, you win!")
        elif w == -2:
            print("It's a draw!")
        else:
            print("Congrats White, you win!")

    # ── Display (state grid, ASCII board) ──

    def update_state(self) -> None:
        """Update display state and PDN from Rust core."""
        self._core.inc_turn_count()
        raw = self._core.get_rank()
        state = [[None for _ in range(8)] for _ in range(4)]
        black_pieces = []
        white_pieces = []
        for i in range(4):
            for j in range(8):
                v = raw[8 * i + j]
                sq = str(1 + j + 8 * i)
                if v == BLACK:
                    state[i][j] = BLACK
                    black_pieces.append(sq)
                elif v == WHITE:
                    state[i][j] = WHITE
                    white_pieces.append(sq)
                elif v == BLACK_KING:
                    state[i][j] = BLACK_KING
                    black_pieces.append("K" + sq)
                elif v == WHITE_KING:
                    state[i][j] = WHITE_KING
                    white_pieces.append("K" + sq)
                else:
                    state[i][j] = EMPTY
        self.state = state
        self.black_pieces = black_pieces
        self.white_pieces = white_pieces
        c = "B" if self.active == BLACK else "W"
        self.pdn["FEN"] = f"{c}:W{','.join(white_pieces)}:B{','.join(black_pieces)}"
        self.pdn["Moves"] = self._core.get_pdn_moves()

    def generate_ascii_board(self, black_pov: bool = True) -> List[Any]:
        board = [None] * 17
        for i in range(9):
            board[2 * i] = ["+", " - "] + ["-", " - "] * 7 + ["+", "\n"]
            if i < 8:
                board[2 * i + 1] = (
                    ["|", "   "]
                    + [a for subl in [["|", "   "] for _ in range(7)] for a in subl]
                    + ["|", "\n"]
                )

        for i, chunk in enumerate(self.state):
            for j, cell in enumerate(chunk):
                x, y = -1, -1
                if j < 4:
                    x = 2 * (7 - 2 * i) + 1
                    y = 2 * (6 - 2 * j) + 1
                else:
                    x = 2 * (6 - 2 * i) + 1
                    y = 2 * (7 - 2 * j) - 1
                piece = " "
                if cell == BLACK:
                    piece = colored("b", "red", attrs=["reverse"])
                elif cell == WHITE:
                    piece = colored("w", "cyan", attrs=["reverse"])
                elif cell == BLACK_KING:
                    piece = colored("B", "red", attrs=["reverse"])
                elif cell == WHITE_KING:
                    piece = colored("W", "cyan", attrs=["reverse"])
                padding = " " if j + 8 * i < 9 else ""
                board[x][y] = piece + str(1 + j + 8 * i) + padding
        if not black_pov:
            board = board[::-1]
        return board

    def print_board(self, black_pov: bool = True) -> str:
        return "".join(map(lambda x: "".join(x), self.generate_ascii_board(black_pov)))

    def __str__(self) -> str:
        return "".join(map(lambda x: "".join(x), self.generate_ascii_board()))
