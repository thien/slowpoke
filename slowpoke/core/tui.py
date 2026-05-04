"""Live tournament dashboard using ``rich.live.Live``."""

from __future__ import annotations

import os
import signal
import sys
from typing import Any, Optional

from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


class TournamentDisplay:
    """Rich Live display for tournament progress.

    Usage:
        with TournamentDisplay(generator):
            generator.run_generations()
    """

    def __init__(self, generator: Any) -> None:
        self.generator = generator
        self.generator.tui = self  # wire up so display_status_info pushes to TUI
        self._live: Optional[Live] = None
        self._started = False

    def start(self) -> None:
        """Enter the Live context and show initial layout."""
        self._live = Live(
            auto_refresh=False,
            refresh_per_second=4,
            screen=True,
            transient=True,
        )
        self._live.__enter__()
        self._started = True
        self._old_sigwinch = signal.signal(signal.SIGWINCH, self._on_resize)
        self.push_update()

    def stop(self) -> None:
        """Exit the Live context."""
        self._started = False
        signal.signal(signal.SIGWINCH, self._old_sigwinch)
        if self._live:
            try:
                self._live.__exit__(None, None, None)
            except Exception:
                pass
            self._live = None

    def _on_resize(self, signum: int, frame: object) -> None:
        """Handle terminal resize (SIGWINCH) by refreshing the display."""
        self.push_update()

    def push_update(self) -> None:
        """Refresh the display (called from tournament thread after games/champs)."""
        if not self._started or self._live is None:
            return
        try:
            layout = self._build_layout()
            self._live.update(layout)
            self._live.refresh()
        except Exception as e:
            print(f"[TUI error] {e}", file=sys.stderr)

    @staticmethod
    def _build_sub_table(entries: dict[str, str], title: str) -> Panel:
        """Build a titled Panel with a two-column key-value Rich Table.

        Uses tight padding (0, 1) to conserve screen space.
        """
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="white", no_wrap=True)
        for key, value in entries.items():
            table.add_row(key, value)
        return Panel(table, title=title)

    def _build_layout(self) -> Layout:
        """Build a 4-panel layout: three sub-tables left, matrix+standings right."""
        layout = Layout()
        layout.split_row(
            Layout(name="left-stats", ratio=2),
            Layout(name="right-panels", ratio=3),
        )
        layout["right-panels"].split_column(
            Layout(name="matrix"),
            Layout(name="standings"),
        )

        # ── Left: three sub-tables (progress / timing / champion) ──
        data = self.generator.status_info()
        left = Layout()
        left.split_column(
            Layout(name="progress-section"),
            Layout(name="timing-section"),
            Layout(name="champion-section"),
        )
        left["progress-section"].update(
            self._build_sub_table(data["progress"], "Progress")
        )
        left["timing-section"].update(self._build_sub_table(data["timing"], "Timing"))
        left["champion-section"].update(
            self._build_sub_table(data["champion"], "Champion")
        )
        g = self.generator.currentGeneration
        layout["left-stats"].update(Panel(left, title=f"Generation {g}"))

        # ── Right top: Win matrix ──
        try:
            term = os.get_terminal_size()
            avail_rows = (term.lines - 16) // 2
            avail_cols = (term.columns - 8) // 12
            matrix_rows = max(4, min(avail_rows, avail_cols))
        except (ValueError, OSError):
            matrix_rows = 8
        matrix = self.generator.population.build_matrix_table(max_rows=matrix_rows)
        if matrix:
            layout["matrix"].update(Panel(matrix, title="Win Matrix"))
        else:
            layout["matrix"].update(Panel("Waiting for games..."))

        # ── Right bottom: Standings ──
        standings = self.generator.population.build_standings_table()
        if standings:
            layout["standings"].update(Panel(standings, title="Standings"))

        return layout

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
