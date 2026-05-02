# Slowpoke — Agent Guide

## Quick start

```bash
# All Python commands must run from library/
cd library

# Train (generations, 15 players)
uv run python train.py light     # ply=1, 200 gen
uv run python train.py medium    # ply=3, 200 gen
uv run python train.py heavy     # ply=6, 200 gen
uv run python train.py debug     # ply=1, random play
uv run python train.py heavy --parallel 8

# Play against a champion
uv run python play.py

# Lint / format / test
uv run ruff check .
uv run ruff format .
uv run pytest              # from repo root
```

## Architecture

All source lives in `library/`. Imports use the `library`-relative path (e.g., `from core.checkers import CheckerBoard`). Tests use `sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))` to find `core/`.

| Directory | Contents |
|---|---|
| `library/core/` | CheckerBoard (bitboard logic), game loop, tournament, population |
| `library/agents/` | Player bots (Slowbro, Slowpoke, Geodude, Magikarp) + evaluator |
| `library/decision/` | MCTS variants: `tmcts.py`, `parallel_tmcts.py`, `minimax.py` |

Agent hierarchy:
- `Agent` (in `agent.py`) wraps a bot with Elo rating, ID, match history
- `Slowbro` is the tournament agent — fused 32-input NN (no subsquares call)
- `Slowpoke` is the legacy agent — 91-input NN with subsquares

## Board mechanics

- Bitboard representation: 36-bit integers per colour (forward/backward/pieces)
- `make_move(move, full_update=True)` — game moves use `full_update=True` (computes PDN + display state)
- `push_move(move)` / `pop_move()` — MCTS search uses `full_update=False` internally, skipping PDN/display and state update.
  The search path does NOT maintain an intermediate rank list — NN evaluation reads bitboards directly.
- `getBoardPosWeighted()` reads bitboards directly in a single pass (no intermediate rank list or dict lookup).
- `get_moves()`, `get_jumps()`, `jumps_from()`, `make_move()` — all use `_set_bits()` helper (bit-twiddling, not `bin()`).
- `is_over()` calls `checkWinner()` which uses bitboard checks (`self.pieces[color] != 0`).

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
