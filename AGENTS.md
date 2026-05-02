# Slowpoke — Agent Guide

## Quick start

```bash
# Build once (required after cloning)
make install

# Train (generations, 15 players) — run from library/
(cd library && ../.venv/bin/python train.py light)   # ply=1, 200 gen
(cd library && ../.venv/bin/python train.py medium)  # ply=3, 200 gen
(cd library && ../.venv/bin/python train.py heavy)   # ply=6, 200 gen
(cd library && ../.venv/bin/python train.py heavy --parallel 8)

# Play against a champion
(cd library && ../.venv/bin/python play.py)

# Lint / format
uv run ruff check .
uv run ruff format .

# Test / bench (after make install)
make test
make bench

# Quick smoke test
make smoke
```

## Architecture

All source lives in `library/`. Imports use the `library`-relative path (e.g., `from core.checkers import CheckerBoard`). Tests use `sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))` to find `core/`.

| Location | Purpose |
|---|---|
| `library/core/` | CheckerBoard, game loop, tournament, population |
| `library/agents/` | Bots (Slowbro, Slowpoke, Geodude, Magikarp) |
| `library/decision/` | MCTS: `tmcts.py`, `parallel_tmcts.py`, `minimax.py` |
| `src/lib.rs` | Rust `checkers_core` — bitboard ops (hot path) |
| `Cargo.toml` | Rust build config |

**Rust backend**: `library/core/checkers.py` delegates `get_moves`, `push_move`, `pop_move`, `getBoardPosWeighted`, and `make_move` to `checkers_core.CheckerBoard` (a Rust PyO3 extension). Falls back to pure Python if the Rust module isn't installed.

Agent hierarchy:
- `Agent` wraps a bot with Elo rating, ID, match history
- `Slowbro` — tournament agent, fused 32-input NN (no subsquares)
- `Slowpoke` — legacy agent, 91-input NN with subsquares

## Board mechanics

- Bitboard representation: 36-bit `u64` per colour (forward/backward/pieces)
- `make_move(move, full_update=True)` — game path, computes PDN + display state
- `push_move` / `pop_move` — MCTS search path. Bitboard state managed by Rust core.
  History stores `mandatoryJumps` and `multipleJumpStack` on Python side; bitboard state in Rust `Vec<HistoryEntry>`.
- `getBoardPosWeighted()` — Rust computes weighted float32[32] directly from bitboards in ~0.7us.
- Move generation (`get_moves`, `get_jumps`, `jumps_from`) — Rust uses `u64::trailing_zeros()` (ARM `cls` instruction) for bit iteration.
- `is_over` / `checkWinner` use bitboard checks (`pieces[color] != 0`).

## Building

The Rust extension requires Rust (install via `rustup`). Build once after cloning:

```bash
make install    # maturin build --release + pip install
make test       # run all 125 tests
make bench      # benchmark hot functions
```

## Concurrency

| Pattern | Where | What |
|---|---|---|
| `ThreadPoolExecutor` | `ParallelTMCTS` | MCTS tree search (no pickling) |
| `multiprocessing.Pool` | `tournament.py`, `population.py` | Game matches, mutations (full object pickling) |
| Lock | `SharedBatchAccumulator` | Only write ops lock; reads are lock-free (GIL-safe dict `.get()`) |

## Key conventions

- **Tests use `unittest`**, not pytest-style functions. Run from repo root: `uv run pytest`.
- **Agents**: `Slowbro` for tournament use; `Slowpoke` for backward compat with legacy 91-input weights.
- **Piece constants** defined in `agents/__init__.py`: `minimax_win=1`, `minimax_lose=-1`, `minimax_draw=0`, `minimax_empty=-1`.
- **`_set_bits(n)`** — module-level helper in `checkers.py`, iterates LSB-to-MSB using `n & -n`.
- **Results** go to `results/` (gitignored). Champions saved as JSON per generation.
- **MLX** optional — GPU batch evaluation for neural network. Enabled per-agent.
- **`CheckerBoard.__slots__`** is defined; do not add ad-hoc attributes.
- **Spelling**: Use British English throughout (colour, behaviour, centre, etc.).
