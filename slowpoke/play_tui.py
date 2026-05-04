"""Textual TUI for playing checkers via slowpoke agents."""
from __future__ import annotations

import re
from typing import Optional

from rich.segment import Segment
from rich.style import Style
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import (
    Button,
    Footer,
    Header,
    Label,
    Select,
    Static,
)

import slowpoke.agents.geodude as geo
import slowpoke.agents.magikarp as ma
import slowpoke.agents.slowpoke as sp
import slowpoke.agents.agent as agent_mod
from slowpoke.core import checkers
from slowpoke.core.constants import BLACK, WHITE, EMPTY, BLACK_KING, WHITE_KING
from slowpoke.play import coef_master


# ── Board coordinate mapping ──────────────────────────────────────────────────
# Derives (board_row 0–7, board_col 0–7) from the state[i][j] layout used in
# checkers.py's generate_ascii_board, so our widget matches that orientation.

_BOARD_TO_STATE: dict[tuple[int, int], tuple[int, int]] = {}
_SQ_TO_BOARD: dict[int, tuple[int, int]] = {}

for _i in range(4):
    for _j in range(8):
        _sq = 1 + _j + 8 * _i
        if _j < 4:
            _r, _c = 7 - 2 * _i, 6 - 2 * _j
        else:
            _r, _c = 6 - 2 * _i, 15 - 2 * _j
        _BOARD_TO_STATE[(_r, _c)] = (_i, _j)
        _SQ_TO_BOARD[_sq] = (_r, _c)


# ── Agent helpers ─────────────────────────────────────────────────────────────

AGENT_CHOICES: list[tuple[str, str]] = [
    ("Slowpoke (Neural Net/MCTS)", "slowpoke"),
    ("Slowpoke Rand (random weights)", "slowpoke_rand"),
    ("Geodude (pure MCTS)", "geodude"),
    ("Magikarp (random moves)", "magikarp"),
    ("Human", "human"),
]

PLY_CHOICES: list[tuple[str, int]] = [(str(i), i) for i in range(1, 8)]


def build_agent(name: str, ply: int) -> Optional[agent_mod.Agent]:
    """Construct an Agent. Returns None for human (handled in UI)."""
    if name == "slowpoke":
        bot = sp.Slowpoke(ply_depth=ply)
        bot.load_weights(coef_master)
        return agent_mod.Agent(bot)
    if name == "slowpoke_rand":
        return agent_mod.Agent(sp.Slowpoke(ply_depth=ply))
    if name == "geodude":
        return agent_mod.Agent(geo.Geodude())
    if name == "magikarp":
        return agent_mod.Agent(ma.Magikarp())
    if name == "human":
        return None
    raise ValueError(f"Unknown agent: {name!r}")


# ── Setup Screen ──────────────────────────────────────────────────────────────


class SetupScreen(Screen):
    """Agent and ply selection."""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="setup-box"):
            yield Label("Slowpoke Checkers", id="setup-title")
            yield Label("Black Player", classes="field-label")
            yield Select(AGENT_CHOICES, id="black-select", value="slowpoke")
            yield Label("White Player", classes="field-label")
            yield Select(AGENT_CHOICES, id="white-select", value="geodude")
            yield Label("Ply Depth (Slowpoke only)", classes="field-label")
            yield Select(PLY_CHOICES, id="ply-select", value=4)
            yield Button("Start Game", id="start-btn", variant="success")
        yield Footer()

    @on(Button.Pressed, "#start-btn")
    def start_game(self) -> None:
        black = self.query_one("#black-select", Select).value
        white = self.query_one("#white-select", Select).value
        ply = self.query_one("#ply-select", Select).value
        if black is Select.BLANK or white is Select.BLANK or ply is Select.BLANK:
            return
        self.app.push_screen(GameScreen(str(black), str(white), int(ply)))


# ── Board Widget ──────────────────────────────────────────────────────────────


class CheckersBoardWidget(Widget):
    """
    Renders an 8×8 checkers board using render_line / Strip / Segment.

    Each square is CELL_W×CELL_H terminal cells.  Dark squares are interactive
    when it is a human's turn: click a highlighted piece to select it, then
    click a green destination to play the move.
    """

    CELL_W = 6
    CELL_H = 3

    _BG_LIGHT = "#c8a46e"
    _BG_DARK = "#4a2c0a"
    _BG_MOVEABLE = "#6e4e00"   # can-move piece
    _BG_SELECTED = "#b8860b"   # selected piece
    _BG_DEST = "#1a4a1a"       # valid destination

    # Cached styles keyed by (bg_hex, fg_spec | None)
    _STYLE_CACHE: dict[tuple[str, str | None], Style] = {}

    class MoveSelected(Message):
        """Emitted when the user clicks a valid destination square."""

        def __init__(self, move: int) -> None:
            super().__init__()
            self.move = move

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._board: Optional[checkers.CheckerBoard] = None
        self._is_human_turn = False
        self._selected_sq: Optional[int] = None
        self._moveable: set[int] = set()
        self._dests: dict[int, int] = {}  # dest_sq → move int

    # ── Public API ────────────────────────────────────────────────────────────

    def set_state(self, board: checkers.CheckerBoard, is_human_turn: bool) -> None:
        self._board = board
        self._is_human_turn = is_human_turn
        self._selected_sq = None
        self._dests = {}
        self._moveable = set()
        if is_human_turn and not board.is_over():
            for ms in board.get_move_strings():
                src = int(re.split(r"[-x]", ms)[0])
                self._moveable.add(src)
        self.refresh()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _style(bg: str, fg: str | None = None) -> Style:
        key = (bg, fg)
        if key not in CheckersBoardWidget._STYLE_CACHE:
            s = Style.parse(f"on {bg}")
            if fg:
                s += Style.parse(f"bold {fg}")
            CheckersBoardWidget._STYLE_CACHE[key] = s
        return CheckersBoardWidget._STYLE_CACHE[key]

    def _sq_at(self, row: int, col: int) -> Optional[int]:
        key = (row, col)
        if key not in _BOARD_TO_STATE:
            return None
        i, j = _BOARD_TO_STATE[key]
        return 1 + j + 8 * i

    def _piece_at(self, row: int, col: int):
        if self._board is None:
            return None
        key = (row, col)
        if key not in _BOARD_TO_STATE:
            return None
        i, j = _BOARD_TO_STATE[key]
        v = self._board.state[i][j]
        return None if v == EMPTY else v

    def _bg(self, sq: Optional[int]) -> str:
        if sq is None:
            return self._BG_DARK
        if sq == self._selected_sq:
            return self._BG_SELECTED
        if sq in self._dests:
            return self._BG_DEST
        if sq in self._moveable:
            return self._BG_MOVEABLE
        return self._BG_DARK

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render_line(self, y: int) -> Strip:
        row = y // self.CELL_H
        if row >= 8:
            return Strip.blank(self.size.width)
        line = y % self.CELL_H
        return Strip([self._cell_seg(row, col, line) for col in range(8)])

    def _cell_seg(self, row: int, col: int, line: int) -> Segment:
        W = self.CELL_W
        is_dark = (row + col) % 2 == 1

        if not is_dark:
            return Segment(" " * W, self._style(self._BG_LIGHT))

        sq = self._sq_at(row, col)
        bg = self._bg(sq)

        if line == 0:
            # Top line: square number in dim text
            label = str(sq) if sq is not None else ""
            return Segment(label.ljust(W), self._style(bg, "dim white"))

        if line == 1:
            # Middle line: piece symbol
            piece = self._piece_at(row, col)
            if piece is None:
                # Show a faint dot on destination squares so they're obvious
                if sq in self._dests:
                    sym, fg = "·", "bright_white"
                else:
                    return Segment(" " * W, self._style(bg))
            elif piece == BLACK:
                sym, fg = "●", "#ff6b6b"
            elif piece == WHITE:
                sym, fg = "●", "bright_white"
            elif piece == BLACK_KING:
                sym, fg = "★", "#ff6b6b"
            else:  # WHITE_KING
                sym, fg = "★", "bright_white"
            left = (W - 1) // 2
            content = " " * left + sym + " " * (W - 1 - left)
            return Segment(content, self._style(bg, fg))

        # Bottom line: blank
        return Segment(" " * W, self._style(bg))

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def on_click(self, event: events.Click) -> None:
        if not self._is_human_turn or self._board is None:
            return
        col = event.x // self.CELL_W
        row = event.y // self.CELL_H
        if col >= 8 or row >= 8:
            return
        if (row + col) % 2 == 0:  # light square — not playable
            return
        sq = self._sq_at(row, col)
        if sq is None:
            return

        if sq in self._dests:
            self.post_message(self.MoveSelected(self._dests[sq]))
            return

        if sq in self._moveable:
            self._select(sq)
            self.refresh()
            return

        self._selected_sq = None
        self._dests = {}
        self.refresh()

    def _select(self, sq: int) -> None:
        self._selected_sq = sq
        self._dests = {}
        if self._board is None:
            return
        moves = self._board.get_moves()
        for idx, ms in enumerate(self._board.get_move_strings()):
            sqs = list(map(int, re.split(r"[-x]", ms)))
            if sqs[0] == sq:
                self._dests[sqs[-1]] = moves[idx]


# ── Game Screen ───────────────────────────────────────────────────────────────


class GameScreen(Screen):
    """Live board + side-panel."""

    BINDINGS = [("escape", "pop_screen", "Setup")]

    def __init__(self, black_name: str, white_name: str, ply: int) -> None:
        super().__init__()
        self.black_name = black_name
        self.white_name = white_name
        self.ply = ply
        self.board = checkers.CheckerBoard()
        self._human_waiting = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="game-layout"):
            with Vertical(id="board-panel"):
                yield CheckersBoardWidget(id="board")
            with Vertical(id="side-panel"):
                yield Static("", id="status-box", classes="status-box")
                yield Label("History", classes="panel-title")
                with ScrollableContainer(id="history-scroll"):
                    yield Static("", id="history-text")
                yield Label("Available Moves", id="moves-label", classes="panel-title")
                yield Static("", id="moves-text")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh()
        self._start_turn()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _player(self, colour: int) -> str:
        return self.black_name if colour == BLACK else self.white_name

    def _is_human(self, colour: int) -> bool:
        return self._player(colour) == "human"

    # ── Display ───────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        B = self.board
        human_turn = not B.is_over() and self._is_human(B.active)

        self.query_one("#board", CheckersBoardWidget).set_state(B, human_turn)

        if B.is_over():
            w = B.winner
            if w == BLACK:
                status = f"[bold green]Black ({self.black_name}) wins![/]"
            elif w == WHITE:
                status = f"[bold cyan]White ({self.white_name}) wins![/]"
            else:
                status = "[bold yellow]Draw![/]"
        else:
            side = "Black" if B.active == BLACK else "White"
            jump = "  [bold red][JUMP][/]" if B.jump else ""
            status = f"Turn [bold]{B.turn_count}[/] — {side}: [bold]{self._player(B.active)}[/]{jump}"
        self.query_one("#status-box", Static).update(status)

        history = (
            "\n".join(f"{i+1:>3}. {m[1]}" for i, m in enumerate(B.moves)) or "(none)"
        )
        self.query_one("#history-text", Static).update(history)

        ml = self.query_one("#moves-label", Label)
        mt = self.query_one("#moves-text", Static)
        if human_turn:
            mt.update("\n".join(f"[{i}] {ms}" for i, ms in enumerate(B.get_move_strings())))
            ml.display = mt.display = True
        else:
            ml.display = mt.display = False

    # ── Game loop ─────────────────────────────────────────────────────────────

    def _start_turn(self) -> None:
        if self.board.is_over():
            return
        colour = self.board.active
        if self._is_human(colour):
            self._human_waiting = True
        else:
            self._human_waiting = False
            ag = build_agent(self._player(colour), self.ply)
            self._run_ai(ag, colour)

    @work(thread=True)
    def _run_ai(self, ag: agent_mod.Agent, colour: int) -> None:
        move = ag.make_move(self.board, colour)
        self.app.call_from_thread(self._apply_move, move)

    def _apply_move(self, move: int) -> None:
        self.board.make_move(move)
        self._refresh()
        if not self.board.is_over():
            self._start_turn()

    @on(CheckersBoardWidget.MoveSelected)
    def on_board_move(self, event: CheckersBoardWidget.MoveSelected) -> None:
        if self._human_waiting:
            self._human_waiting = False
            self._apply_move(event.move)


# ── App ───────────────────────────────────────────────────────────────────────


class PlayApp(App):
    TITLE = "Slowpoke Checkers"

    CSS = """
    /* ── Setup ── */
    SetupScreen { align: center middle; }
    #setup-box {
        width: 64; height: auto;
        border: thick $primary; padding: 1 3;
    }
    #setup-title {
        text-align: center; text-style: bold;
        color: $accent; margin-bottom: 1;
    }
    #start-btn { margin-top: 2; width: 100%; }
    .field-label { color: $text-disabled; margin-top: 1; }

    /* ── Game ── */
    #game-layout { height: 1fr; }
    #board-panel {
        width: auto; min-width: 50;
        border: solid $primary;
        padding: 1; align: center middle;
    }
    CheckersBoardWidget { width: 48; height: 24; }
    #side-panel { width: 1fr; padding: 0 1; }
    .status-box {
        border: solid $accent; background: $surface;
        padding: 1; height: auto; margin-bottom: 1;
    }
    .panel-title { color: $accent; text-style: bold; margin-top: 1; }
    #history-scroll { height: 1fr; border: solid $surface-lighten-2; }
    #moves-text { padding: 0 1; }
    """

    def on_mount(self) -> None:
        self.push_screen(SetupScreen())


def main() -> None:
    PlayApp().run()


if __name__ == "__main__":
    main()
