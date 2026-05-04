"""Textual TUI for playing checkers via slowpoke agents."""
from __future__ import annotations

from typing import Optional

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    Select,
    Static,
)

import slowpoke.agents.geodude as geo
import slowpoke.agents.magikarp as ma
import slowpoke.agents.slowpoke as sp
import slowpoke.agents.agent as agent_mod
from slowpoke.core import checkers
from slowpoke.core.constants import BLACK, WHITE
from slowpoke.play import coef_master


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
    """Agent selection and ply configuration."""

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


# ── Game Screen ───────────────────────────────────────────────────────────────


class GameScreen(Screen):
    """Live board display with move controls."""

    BINDINGS = [("escape", "pop_screen", "Setup")]

    def __init__(self, black_name: str, white_name: str, ply: int) -> None:
        super().__init__()
        self.black_name = black_name
        self.white_name = white_name
        self.ply = ply
        self.board: checkers.CheckerBoard = checkers.CheckerBoard()
        self._human_waiting = False

    # ── Layout ────────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="game-layout"):
            with Vertical(id="board-panel"):
                yield Label("Board", classes="panel-title")
                yield Static("", id="board-display")
            with Vertical(id="side-panel"):
                yield Static("", id="status-box", classes="status-box")
                yield Label("History", classes="panel-title")
                with ScrollableContainer(id="history-scroll"):
                    yield Static("", id="history-text")
                yield Label("Your Move", id="move-label", classes="panel-title")
                yield ListView(id="move-list")
        yield Footer()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        self._refresh_display()
        self._start_turn()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _player_name(self, colour: int) -> str:
        return self.black_name if colour == BLACK else self.white_name

    def _is_human(self, colour: int) -> bool:
        return self._player_name(colour) == "human"

    # ── Display ───────────────────────────────────────────────────────────────

    def _refresh_display(self) -> None:
        B = self.board

        # Board (convert termcolor ANSI to Rich)
        self.query_one("#board-display", Static).update(
            Text.from_ansi(B.print_board(black_pov=True))
        )

        # Status
        if B.is_over():
            w = B.winner
            if w == BLACK:
                msg = f"[bold green]Black ({self.black_name}) wins![/]"
            elif w == WHITE:
                msg = f"[bold cyan]White ({self.white_name}) wins![/]"
            else:
                msg = "[bold yellow]Draw![/]"
        else:
            colour = B.active
            side = "Black" if colour == BLACK else "White"
            pname = self._player_name(colour)
            jump = "  [bold red][JUMP][/]" if B.jump else ""
            msg = f"Turn [bold]{B.turn_count}[/] — {side}: [bold]{pname}[/]{jump}"
        self.query_one("#status-box", Static).update(msg)

        # Move history
        moves = B.moves
        history = (
            "\n".join(f"{i + 1:>3}. {m[1]}" for i, m in enumerate(moves))
            or "(none)"
        )
        self.query_one("#history-text", Static).update(history)

        # Human move list (only when it's a human's turn)
        move_list = self.query_one("#move-list", ListView)
        move_label = self.query_one("#move-label", Label)
        move_list.clear()
        show_moves = not B.is_over() and self._is_human(B.active)
        move_label.display = show_moves
        move_list.display = show_moves
        if show_moves:
            for i, ms in enumerate(B.get_move_strings()):
                move_list.append(ListItem(Label(f"[{i}]  {ms}"), id=f"move-{i}"))

    # ── Game loop ─────────────────────────────────────────────────────────────

    def _start_turn(self) -> None:
        B = self.board
        if B.is_over():
            return
        colour = B.active
        if self._is_human(colour):
            self._human_waiting = True
        else:
            self._human_waiting = False
            ag = build_agent(self._player_name(colour), self.ply)
            self._run_ai(ag, colour)

    @work(thread=True)
    def _run_ai(self, ag: agent_mod.Agent, colour: int) -> None:
        move = ag.make_move(self.board, colour)
        self.app.call_from_thread(self._apply_move, move)

    def _apply_move(self, move: int) -> None:
        self.board.make_move(move)
        self._refresh_display()
        if not self.board.is_over():
            self._start_turn()

    # ── Human input ───────────────────────────────────────────────────────────

    @on(ListView.Selected, "#move-list")
    def on_human_move(self, event: ListView.Selected) -> None:
        if not self._human_waiting:
            return
        item_id = event.item.id or ""
        if not item_id.startswith("move-"):
            return
        idx = int(item_id.removeprefix("move-"))
        legal = self.board.get_moves()
        if idx < len(legal):
            self._human_waiting = False
            self._apply_move(legal[idx])


# ── App ───────────────────────────────────────────────────────────────────────


class PlayApp(App):
    """Slowpoke Checkers TUI."""

    TITLE = "Slowpoke Checkers"

    CSS = """
    /* ── Setup screen ── */
    SetupScreen {
        align: center middle;
    }
    #setup-box {
        width: 64;
        height: auto;
        border: thick $primary;
        padding: 1 3;
    }
    #setup-title {
        text-align: center;
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    #start-btn {
        margin-top: 2;
        width: 100%;
    }

    /* ── Game screen ── */
    #game-layout {
        height: 1fr;
    }
    #board-panel {
        width: 40;
        border: solid $primary;
        padding: 0 1;
    }
    #side-panel {
        width: 1fr;
        padding: 0 1;
    }
    .panel-title {
        color: $accent;
        text-style: bold;
        margin-top: 1;
    }
    .status-box {
        border: solid $accent;
        background: $surface;
        padding: 1;
        height: auto;
        margin-bottom: 1;
    }
    #history-scroll {
        height: 1fr;
        border: solid $surface-lighten-2;
        margin-bottom: 1;
    }
    #move-list {
        height: auto;
        max-height: 14;
        border: solid $success;
    }

    /* ── Shared ── */
    .field-label {
        color: $text-disabled;
        margin-top: 1;
    }
    """

    def on_mount(self) -> None:
        self.push_screen(SetupScreen())


def main() -> None:
    PlayApp().run()


if __name__ == "__main__":
    main()
