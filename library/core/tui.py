"""Live tournament dashboard using ``rich.live.Live``."""

from __future__ import annotations

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
        self._initialized = False

    def start(self) -> None:
        """Enter the Live context and show initial layout."""
        self._live = Live(refresh_per_second=4, screen=True)
        self._live.__enter__()
        self._initialized = True
        # Show initial layout immediately
        try:
            self._live.update(self._build_layout())
        except Exception:
            pass

    def stop(self) -> None:
        """Exit the Live context."""
        self._initialized = False
        if self._live:
            try:
                self._live.__exit__(None, None, None)
            except Exception:
                pass
            self._live = None

    def push_update(self) -> None:
        """Refresh the display (called from tournament thread after games/champs)."""
        if not self._initialized or self._live is None:
            return
        try:
            self._live.update(self._build_layout())
        except Exception:
            pass

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
        layout["info"].update(
            Panel(info_panel, title=f"Generation {self.generator.currentGeneration}")
        )

        standings = self.generator.population.build_standings_table()
        if standings:
            layout["standings"].update(Panel(standings, title="Standings"))

        return layout

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
