"""Storage backends for training data and champion agents.

Supports:
- Parquet for structured game statistics (columnar, compressed)
- .npz for champion NN coefficients (compressed numpy archives)
- JSON for backward compatibility / migration
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

HAS_PARQUET = False
try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    HAS_PARQUET = True
except ImportError:
    pass


# ── Champions (.npz) ──


def save_champion_npz(
    filepath: str, coefficients: np.ndarray, meta: Optional[Dict[str, Any]] = None
) -> str:
    """Save champion NN coefficients as a compressed .npz file.

    Args:
        filepath: Output path (``.npz`` suffix added if missing).
        coefficients: Flat float32 array of NN weights and biases.
        meta: Optional dict of metadata (elo, points, champ_score, etc.).

    Returns:
        Actual path written.
    """
    if not filepath.endswith(".npz"):
        filepath += ".npz"
    data: Dict[str, Any] = {"coefficients": coefficients.astype(np.float32)}
    if meta:
        for k, v in meta.items():
            data[k] = v
    np.savez_compressed(filepath, **data)
    return filepath


def load_champion_npz(filepath: str) -> Dict[str, Any]:
    """Load champion data from a .npz file.

    Args:
        filepath: Path to ``.npz`` file.

    Returns:
        Dict with ``coefficients`` (ndarray) and any stored metadata.
    """
    data = np.load(filepath, allow_pickle=True)
    result = dict(data)
    data.close()
    return result


# ── Statistics (Parquet) ──


def _flatten_game(game: Dict[str, Any], gen: int) -> Dict[str, Any]:
    """Flatten a single game result dict into a row dict for Parquet storage."""
    row = {
        "gen": gen,
        "game_id": game.get("game", {}).get("_id", ""),
        "black_id": game.get("black", ""),
        "white_id": game.get("white", ""),
        "winner": game.get("game", {}).get("Winner", -1),
        "num_moves": len(game.get("game", {}).get("Moves", [])),
        "duration": game.get("duration", ""),
        "replay": json.dumps(game.get("game", {}).get("Moves", [])),
    }
    # Black/White Elo at time of game (if present)
    for side in ("black", "white"):
        elo_key = f"{side}_elo"
        elo_val = game.get(elo_key)
        if elo_val is not None:
            row[elo_key] = elo_val
    return row


def _game_schema() -> pa.Schema:
    """Return the PyArrow schema for game statistics."""
    return pa.schema([
        pa.field("gen", pa.int32()),
        pa.field("game_id", pa.string()),
        pa.field("black_id", pa.string()),
        pa.field("white_id", pa.string()),
        pa.field("winner", pa.int32()),
        pa.field("num_moves", pa.int32()),
        pa.field("duration", pa.string()),
        pa.field("replay", pa.string()),
        pa.field("black_elo", pa.float32(), nullable=True),
        pa.field("white_elo", pa.float32(), nullable=True),
    ])


def save_statistics_parquet(
    directory: str, generation_stats: List[Dict[str, Any]]
) -> str:
    """Save per-generation game results as Parquet files.

    Writes one file per generation to ``{directory}/statistics/gen_{n:04d}.parquet``
    and a combined ``{directory}/statistics/all.parquet``.

    Args:
        directory: Base results directory (e.g. ``results/2024-01-01_4ply/``).
        generation_stats: List of per-generation stats dicts, each containing
            ``games`` (list of game results) and optionally ``durationInSeconds``.

    Returns:
        Path to the combined parquet file.
    """
    if not HAS_PARQUET:
        raise ImportError("pyarrow is required for Parquet storage")

    stats_dir = os.path.join(directory, "statistics")
    os.makedirs(stats_dir, exist_ok=True)

    all_rows = []
    for gen_idx, gen_data in enumerate(generation_stats):
        games = gen_data.get("games", [])
        rows = [_flatten_game(g, gen_idx) for g in games]
        if not rows:
            continue

        table = pa.Table.from_pylist(rows, schema=_game_schema())
        gen_path = os.path.join(stats_dir, f"gen_{gen_idx:04d}.parquet")
        pq.write_table(table, gen_path, compression="zstd")
        all_rows.extend(rows)

    combined_path = os.path.join(stats_dir, "all.parquet")
    if all_rows:
        table = pa.Table.from_pylist(all_rows, schema=_game_schema())
        pq.write_table(table, combined_path, compression="zstd")

    return combined_path


def load_statistics_parquet(
    directory: str, gen: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load game results from Parquet files.

    Args:
        directory: Base results directory.
        gen: If set, load only that generation's file; otherwise load combined.

    Returns:
        List of game result dicts.
    """
    if not HAS_PARQUET:
        raise ImportError("pyarrow is required for Parquet storage")

    stats_dir = os.path.join(directory, "statistics")
    if gen is not None:
        path = os.path.join(stats_dir, f"gen_{gen:04d}.parquet")
    else:
        path = os.path.join(stats_dir, "all.parquet")

    if not os.path.isfile(path):
        return []

    table = pq.read_table(path)
    return table.to_pylist()


# ── Migration helpers ──


def migrate_statistics_json_to_parquet(
    directory: str,
) -> str:
    """Migrate legacy ``statistics.json`` to Parquet.

    Reads ``{directory}/statistics.json`` (written by the old JSON writer),
    converts each game entry to a Parquet row, and writes per-gen + combined
    Parquet files.

    Args:
        directory: Base results directory containing ``statistics.json``.

    Returns:
        Path to the combined parquet file, or empty string if no JSON found.
    """
    json_path = os.path.join(directory, "statistics.json")
    if not os.path.isfile(json_path):
        return ""

    with open(json_path, "r") as f:
        stats = json.load(f)

    return save_statistics_parquet(directory, stats)


def migrate_champions_json_to_npz(
    champion_dir: str,
) -> int:
    """Migrate legacy champion JSON files to ``.npz`` format.

    Reads all ``*.json`` files in ``champion_dir``, extracts coefficients,
    and writes equivalent ``.npz`` files. Skips files that already have
    a corresponding ``.npz``.

    Args:
        champion_dir: Path to the champions directory.

    Returns:
        Number of files migrated.
    """
    count = 0
    if not os.path.isdir(champion_dir):
        return count

    for fname in os.listdir(champion_dir):
        if not fname.endswith(".json"):
            continue
        base = fname[: -len(".json")]
        npz_path = os.path.join(champion_dir, base + ".npz")
        if os.path.isfile(npz_path):
            continue  # already migrated

        json_path = os.path.join(champion_dir, fname)
        with open(json_path, "r") as f:
            data = json.load(f)

        # Champion JSON has format: { "gen_number": { "pid": ..., "coefficents": [...], ... } }
        for gen_key, champ_data in data.items():
            coeffs = np.array(champ_data.get("coefficents", []), dtype=np.float32)
            meta = {
                "pid": str(champ_data.get("pid", "")),
                "champ_range": json.dumps(champ_data.get("champ_range", [])),
                "champ_score": float(champ_data.get("champ_score", 0)),
            }
            save_champion_npz(npz_path, coeffs, meta)
            count += 1

    return count
