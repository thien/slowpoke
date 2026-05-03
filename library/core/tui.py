"""Live tournament dashboard using ``rich.live.Live``.

Lightweight alternative to Textual — no threading issues, works from
any thread without signal-handling conflicts.
"""

from __future__ import annotations

import threading
from typing import Any, List, Optional

from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


class TournamentDisplay:
    """Rich Live display for tournament progress.

    Usage:
        display = TournamentDisplay(generator)
        display.start()          # starts Live in a thread
        display.push_update(...) # called after each game batch
        display.stop()           # stops when tournament ends
    """

    def __init__(self, generator: Any) -> None:
        self.generator = generator
        self._live: Optional[Live] = None
        self._latest: str = "Starting..."
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the live display in a background thread."""
        self._live = Live(refresh_per_second=4, screen=True)
        self._live.__enter__()
        # Start a refresh loop
        self._running = True
        self._refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._refresh_thread.start()

    def _refresh_loop(self) -> None:
        """Periodically refresh the display."""
        while self._running:
            layout = self._build_layout()
            if self._live:
                self._live.update(layout)
            import time
            time.sleep(0.25)

    def stop(self) -> None:
        """Stop the live display."""
        self._running = False
        if self._live:
            try:
                self._live.__exit__(None, None, None)
            except Exception:
                pass
            self._live = None

    def push_update(self) -> None:
        """Signal that new data is available (called from tournament thread)."""
        pass  # The refresh loop picks up changes automatically

    def _build_layout(self) -> Layout:
        """Build the rich Layout with info and standings panels."""
        layout = Layout()
        layout.split_column(
            Layout(name="info"),
            Layout(name="standings"),
        )

        # ── Info panel: 2-column metrics ──
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
        info_panel.split_row(
            Panel(info_left),
            Panel(info_right),
        )
        layout["info"].update(
            Panel(info_panel, title=f"Generation {self.generator.currentGeneration}")
        )

        # ── Standings table ──
        standings = self.generator.population.build_standings_table()
        if standings:
            layout["standings"].update(Panel(standings, title="Standings"))

        return layout

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
