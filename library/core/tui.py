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
        """Build the rich Layout with info and standings panels."""
        layout = Layout()
        layout.split_column(
            Layout(name="info"),
            Layout(name="standings"),
        )

        info_left = Table.grid(padding=(0, 2))
        info_right = Table.grid(padding=(0, 2))
        info_left.add_column("Metric", style="cyan", no_wrap=True)
        info_left.add_column("Value", style="white")
        info_right.add_column("Metric", style="cyan", no_wrap=True)
        info_right.add_column("Value", style="white")

        items = self.generator.status_info()
        mid = len(items) // 2
        for idx, (metric, value) in enumerate(items):
            ms = str(metric) if metric else ""
            vs = str(value) if value else ""
            if not ms.strip() and not vs.strip():
                continue
            if ms.startswith("Player") or ms == "Previous Scoreboard":
                continue
            (info_left if idx < mid else info_right).add_row(ms, vs)

        info_panel = Layout()
        info_panel.split_row(Panel(info_left), Panel(info_right))
        g = self.generator.currentGeneration
        layout["info"].update(Panel(info_panel, title=f"Generation {g}"))

        standings = self.generator.population.build_standings_table()
        if standings:
            layout["standings"].update(Panel(standings, title="Standings"))

        return layout

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
