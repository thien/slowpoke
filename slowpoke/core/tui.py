"""Textual-based live tournament dashboard."""

from __future__ import annotations

import queue
from typing import Any, Dict, List

from rich.panel import Panel
from rich.table import Table as RichTable
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Static


class MetricsPanel(Static):
    """Panel showing key-value metric pairs as a Rich table."""

    def __init__(self, title: str, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self._title = title

    def set_data(self, entries: Dict[str, str]) -> None:
        """Rebuild the panel from a dict of metric -> value strings."""
        table = RichTable(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        for key, value in entries.items():
            table.add_row(key, value)
        self.update(Panel(table, title=self._title))


class TournamentApp(App[None]):
    """Textual live display for tournament progress.

    The generation loop runs in a background worker thread.
    The tournament loop pushes game results through a thread-safe queue;
    a periodic timer drains the queue and updates the widgets on the
    Textual event loop.
    """

    CSS = """
    Screen {
        background: $surface;
    }
    .metrics-col {
        height: 1fr;
    }
    MetricsPanel {
        height: 1fr;
        border: solid $primary;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    DataTable {
        height: 1fr;
    }
    #standings {
        height: 1fr;
        margin: 1 0 0 0;
    }
    """

    BINDINGS = [("q", "quit", "Quit"), ("ctrl+c", "quit", "Quit")]

    def __init__(self, generator: Any) -> None:
        super().__init__()
        self.generator = generator
        self.generator.tui = self
        # Thread-safe event queue: worker thread pushes, event loop drains
        self._event_queue: queue.Queue = queue.Queue()
        self.generations_complete = False

    # ── Composition ──────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(classes="metrics-col"):
                yield MetricsPanel(id="progress", title="Progress")
                yield MetricsPanel(id="timing", title="Timing")
                yield MetricsPanel(id="champion", title="Champion")
            with Vertical():
                yield DataTable(id="matrix", show_cursor=False, zebra_stripes=True)
                yield DataTable(id="standings", show_cursor=False, zebra_stripes=True)
        yield Footer()

    # ── Life cycle ──────────────────────────────────────────────

    def on_mount(self) -> None:
        self.title = "Slowpoke — Training"
        self._prep_matrix()
        self._prep_standings()
        self._refresh_metrics()
        self.set_interval(1.0, self._tick)
        self.run_worker(self._worker_run, thread=True, exclusive=True)

    # ── Worker thread ───────────────────────────────────────────

    def _worker_run(self) -> None:
        """Run the blocking generation loop in a worker thread."""
        try:
            self.generator.run_generations()
        except KeyboardInterrupt:
            pass
        finally:
            self._event_queue.put(("done",))

    # ── Periodic timer ──────────────────────────────────────────

    def _tick(self) -> None:
        """Drain the event queue and refresh metrics."""
        interrupted = False
        try:
            while True:
                event = self._event_queue.get_nowait()
                if event[0] == "game":
                    self._refresh_matrix()
                    self._refresh_standings()
                elif event[0] == "refresh":
                    self._refresh_all()
                elif event[0] == "done":
                    interrupted = True
                    break
        except queue.Empty:
            pass
        self._refresh_metrics()
        if interrupted:
            self.generations_complete = True
            self.exit()

    # ── Public API (called from worker thread) ──────────────────

    def after_game(self, black: int, white: int, winner: int) -> None:
        """Queue a game-completed event (thread-safe)."""
        self._event_queue.put(("game", black, white, winner))

    def request_refresh(self) -> None:
        """Queue a full refresh event (thread-safe)."""
        self._event_queue.put(("refresh",))

    # ── Panel builders ──────────────────────────────────────────

    def _refresh_all(self) -> None:
        self._refresh_matrix()
        self._refresh_standings()
        self._refresh_metrics()

    def _refresh_metrics(self) -> None:
        data = self.generator.status_info()
        for key in ("progress", "timing", "champion"):
            self.query_one(f"#{key}", MetricsPanel).set_data(data[key])

    def _prep_matrix(self) -> None:
        """Initialise the matrix DataTable columns."""
        pop = self.generator.population
        pids = self._sorted_pids(pop)
        dt = self.query_one("#matrix", DataTable)
        dt.add_column("", key="_label")
        for pid in pids:
            label = self._player_label(pop, pid)
            dt.add_column(label, key=str(pid))

    def _refresh_matrix(self) -> None:
        """Rebuild matrix DataTable rows from head-to-head data.

        Columns are stable (set once in _prep_matrix), so we clear rows
        and re-add them rather than using update_cell, which is simpler
        for the matrix structure.
        """
        pop = self.generator.population
        pids = self._sorted_pids(pop)
        dt = self.query_one("#matrix", DataTable)
        dt.clear()
        for a in pids:
            cols = [self._player_label(pop, a)]
            for b in pids:
                if a == b:
                    cols.append("—")
                else:
                    cols.append(self._cell_text(pop, a, b))
            dt.add_row(*cols, key=str(a))

    def _prep_standings(self) -> None:
        """Initialise the standings DataTable columns."""
        dt = self.query_one("#standings", DataTable)
        dt.add_column("Player", key="_player")
        dt.add_column("Elo", key="_elo")
        dt.add_column("Pts", key="_pts")
        dt.add_column("W", key="_w")
        dt.add_column("D", key="_d")
        dt.add_column("L", key="_l")
        dt.add_column("Score", key="_score")

    def _refresh_standings(self) -> None:
        """Rebuild standings DataTable from population data."""
        pop = self.generator.population
        pids = self._sorted_pids(pop)
        dt = self.query_one("#standings", DataTable)
        dt.clear()
        for pid in pids:
            p = pop.players[pid]
            label = self._player_label(pop, pid)
            h2h = pop.head_to_head
            w = d = loss = 0
            for opp in self._all_pids(pop):
                if pid == opp:
                    continue
                rec_ab = h2h.get((pid, opp), [0, 0, 0])
                rec_ba = h2h.get((opp, pid), [0, 0, 0])
                w += rec_ab[0] + rec_ba[1]
                d += rec_ab[2] + rec_ba[2]
                loss += rec_ab[1] + rec_ba[0]
            total = w + d + loss
            score = f"{w / total:.2f}" if total > 0 else "—"
            dt.add_row(
                label,
                f"{p.elo:.0f}",
                f"{p.points:.0f}",
                str(w),
                str(d),
                str(loss),
                score,
                key=str(pid),
            )

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _sorted_pids(pop: Any) -> List[int]:
        pids = list(pop.current_population)
        pids.sort(key=lambda pid: pop.players[pid].elo, reverse=True)
        return pids

    @staticmethod
    def _all_pids(pop: Any) -> List[int]:
        return list(pop.current_population)

    @staticmethod
    def _player_label(pop: Any, pid: int) -> str:
        return getattr(pop.players[pid], "entity_name", None) or f"P{pid}"

    @staticmethod
    def _cell_text(pop: Any, a: int, b: int) -> str:
        rec_ab = pop.head_to_head.get((a, b), [0, 0, 0])
        rec_ba = pop.head_to_head.get((b, a), [0, 0, 0])
        w = rec_ab[0] + rec_ba[1]
        d = rec_ab[2] + rec_ba[2]
        loss = rec_ab[1] + rec_ba[0]
        if w + d + loss == 0:
            return "."
        return f"{w}-{d}-{loss}"
