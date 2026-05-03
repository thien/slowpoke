"""Textual TUI for live tournament dashboard.

Architecture: Textual runs in a background thread; the tournament runs
in the main thread (required by ``multiprocessing.Pool``).
"""

from __future__ import annotations

import threading
from typing import Any, List, Optional

from rich.table import Table as RichTable
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.reactive import reactive
from textual.widgets import DataTable, Footer, Header, Label, Static


class TournamentApp(App):
    """Textual app showing live tournament progress.

    Call ``run_and_wait()`` to start the TUI in a background thread
    and block until the app is mounted and ready. Then run the
    tournament in the main thread, calling ``push_*`` methods
    via ``call_from_thread``.
    """

    CSS = """
    Screen {
        layout: vertical;
    }
    #info-row {
        layout: horizontal;
        height: 7;
        margin: 0 1;
    }
    #info-left, #info-right {
        width: 1fr;
        height: auto;
    }
    #info-left Label, #info-right Label {
        padding: 0 1;
    }
    #standings {
        height: 1fr;
        margin: 0 1;
    }
    #game-progress {
        height: 1;
        content-align: center middle;
        background: $surface;
    }
    """

    def __init__(self, generator: Any) -> None:
        self.generator = generator
        self._tui_ready = threading.Event()
        self._thread: Optional[threading.Thread] = None
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="info-row"):
            with Vertical(id="info-left"):
                yield Static(id="gen-info", markup="Loading...")
            with Vertical(id="info-right"):
                yield Static(id="time-info", markup="")
        yield DataTable(id="standings")
        yield Label(id="game-progress")
        yield Footer()

    # ── Lifecycle ──

    def on_mount(self) -> None:
        """Signal that the app is ready to receive updates."""
        self._tui_ready.set()

    def run_and_wait(self) -> None:
        """Start the TUI in a background thread and block until mounted.

        After this returns, the main thread may run the tournament and
        push updates via ``call_from_thread``.
        """
        self._thread = threading.Thread(target=self.run, daemon=False)
        self._thread.start()
        self._tui_ready.wait()

    def join(self) -> None:
        """Wait for the TUI thread to exit."""
        if self._thread is not None:
            self._thread.join()

    # ── Public API called from main thread via call_from_thread ──
    # NOTE: names do NOT start with "on_" because Textual intercepts
    # on_* methods as event handlers.

    def push_game_completed(
        self,
        gen: int,
        total_gens: int,
        game_idx: int,
        total_games: int,
    ) -> None:
        """Update the display after a game finishes."""
        self._update_info(gen, total_gens)
        self._update_progress(game_idx, total_games)
        self._update_standings()

    def push_generation_completed(self, gen: int, total_gens: int) -> None:
        """Update the display at the end of a generation."""
        self._update_info(gen, total_gens)
        self._update_standings()
        self._update_progress(0, 0)

    # ── Internal helpers ──

    def _update_info(self, gen: int, total_gens: int) -> None:
        left_lines: List[str] = []
        right_lines: List[str] = []
        mid = len(self.generator.status_info()) // 2
        for idx, item in enumerate(self.generator.status_info()):
            metric = str(item[0]) if item[0] else ""
            value = str(item[1]) if item[1] else ""
            if not metric.strip() and not value.strip():
                continue
            line = f"[bold cyan]{metric}[/bold cyan]  [white]{value}[/white]"
            if idx < mid:
                left_lines.append(line)
            else:
                right_lines.append(line)

        self.query_one("#gen-info", Static).update("\n".join(left_lines))
        self.query_one("#time-info", Static).update("\n".join(right_lines))

    def _update_standings(self) -> None:
        dt = self.query_one("#standings", DataTable)
        dt.clear()
        dt.add_columns("Player", "Elo", "Pts", "W", "D", "L", "Score")
        table = self.generator.population.build_standings_table()
        if table is None:
            return
        for row in table.rows:
            cells = [c for c in row.cells]
            if len(cells) >= 7:
                dt.add_row(*[str(c) for c in cells[:7]])

    def _update_progress(self, game_idx: int, total_games: int) -> None:
        label = self.query_one("#game-progress", Label)
        if total_games > 0:
            pct = int(game_idx / total_games * 20)
            bar = "█" * pct + "░" * (20 - pct)
            label.update(f" {bar}  {game_idx}/{total_games}")
        else:
            label.update("")
