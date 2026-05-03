"""Textual TUI for live tournament dashboard."""

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

    Runs the tournament in a background thread and updates the display
    after each game via ``call_from_thread``.
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
        self._ready = threading.Event()
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

    def on_mount(self) -> None:
        """Start the tournament in a background thread."""
        self._ready.set()
        thread = threading.Thread(target=self._run_tournament, daemon=True)
        thread.start()

    def _run_tournament(self) -> None:
        """Run generations loop and exit app when done."""
        try:
            self.generator.run_generations()
        finally:
            self.call_from_thread(self.exit)

    # ── Public API called from tournament thread ──

    def on_game_completed(
        self,
        gen: int,
        total_gens: int,
        game_idx: int,
        total_games: int,
    ) -> None:
        """Update the display after a game finishes.

        Called from the tournament thread via ``call_from_thread``.
        """
        self._update_info(gen, total_gens)
        self._update_progress(game_idx, total_games)
        self._update_standings()

    def on_generation_completed(self, gen: int, total_gens: int) -> None:
        """Update the display at the end of a generation."""
        self._update_info(gen, total_gens)
        self._update_standings()
        self._update_progress(0, 0)

    def _update_info(self, gen: int, total_gens: int) -> None:
        """Refresh the two-column info panel from generator.status_info()."""
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
        """Rebuild the standings DataTable from population data."""
        dt = self.query_one("#standings", DataTable)
        dt.clear()
        dt.add_columns("Player", "Elo", "Pts", "W", "D", "L", "Score")

        table = self.generator.population.build_standings_table()
        if table is None:
            return

        # Extract rows from the rich Table
        for row in table.rows:
            cells = [c for c in row.cells]
            if len(cells) >= 7:
                dt.add_row(*[str(c) for c in cells[:7]])

    def _update_progress(self, game_idx: int, total_games: int) -> None:
        """Update the game progress label."""
        label = self.query_one("#game-progress", Label)
        if total_games > 0:
            pct = int(game_idx / total_games * 20)
            bar = "█" * pct + "░" * (20 - pct)
            label.update(f" {bar}  {game_idx}/{total_games}")
        else:
            label.update("")
