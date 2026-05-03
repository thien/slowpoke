"""Live tournament dashboard using ``rich.live.Live``."""

from __future__ import annotations

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
        self.push_update()

    def stop(self) -> None:
        """Exit the Live context."""
        self._started = False
        if self._live:
            try:
                self._live.__exit__(None, None, None)
            except Exception:
                pass
            self._live = None

    def push_update(self) -> None:
        """Refresh the display (called from tournament thread after games/champs)."""
        if not self._started or self._live is None:
            return
        try:
            layout = self._build_layout()
            self._live.update(layout)
            # Force an immediate refresh (auto_refresh is False)
            self._live.refresh()
        except Exception as e:
            # Print to stderr so it shows up even in alt-screen mode
            print(f"[TUI error] {e}", file=sys.stderr)

    def _build_layout(self) -> Layout:
        """Build a 3-panel layout: stats left, matrix+standings right."""
        layout = Layout()
        layout.split_row(
            Layout(name="left-stats", ratio=2),
            Layout(name="right-panels", ratio=3),
        )
        layout["right-panels"].split_column(
            Layout(name="matrix"),
            Layout(name="standings"),
        )

        # ── Left: Info metrics ──
        info = Table(show_header=False, box=None, padding=(0, 2))
        info.add_column("Metric", style="cyan")
        info.add_column("Value", style="white")
        for metric, value in self.generator.status_info():
            ms = str(metric) if metric is not None else ""
            vs = str(value) if value is not None else ""
            if not ms.strip() or not vs.strip():
                continue
            if ms.startswith("Player") or ms == "Previous Scoreboard":
                continue
            info.add_row(ms, vs)
        g = self.generator.currentGeneration
        layout["left-stats"].update(Panel(info, title=f"Generation {g}"))

        # ── Right top: Win matrix ──
        matrix = self.generator.population.build_matrix_table()
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
