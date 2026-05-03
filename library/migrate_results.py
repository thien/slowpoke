#!/usr/bin/env python3
"""Migrate legacy JSON results to compressed formats.

Reads existing ``statistics.json`` and champion ``.json`` files in a
results directory and writes equivalent Parquet / ``.npz`` files.

Usage:
    python library/migrate_results.py [<results_dir>]

If no directory is given, migrates all directories under ``results/``.
"""

from __future__ import annotations

import argparse
import os
import sys

_lib_dir = os.path.dirname(os.path.abspath(__file__))
if _lib_dir not in sys.path:
    sys.path.insert(0, _lib_dir)

import core.storage as storage


def migrate_one(directory: str, dry_run: bool = False) -> dict:
    """Migrate a single results directory.

    Args:
        directory: Path to results directory (e.g. ``results/2024-01-01_4ply/``).
        dry_run: If True, only report what would be done.

    Returns:
        Dict with ``stats`` and ``champions`` migration counts.
    """
    result: dict = {"stats": 0, "champions": 0, "dir": directory}

    # ── Migrate statistics.json → Parquet ──
    json_path = os.path.join(directory, "statistics.json")
    if os.path.isfile(json_path):
        if dry_run:
            print(f"  [DRY RUN] Would migrate: {json_path}")
            result["stats"] = 1
        else:
            try:
                out = storage.migrate_statistics_json_to_parquet(directory)
                if out:
                    print(f"  Migrated statistics → {out}")
                    result["stats"] = 1
                else:
                    print(f"  No statistics to migrate in {directory}")
            except Exception as e:
                print(f"  ERROR migrating statistics: {e}")

    # ── Migrate champions/*.json → .npz ──
    champ_dir = os.path.join(directory, "champions")
    if os.path.isdir(champ_dir):
        if dry_run:
            jsons = [f for f in os.listdir(champ_dir) if f.endswith(".json")]
            print(f"  [DRY RUN] Would migrate {len(jsons)} champion files")
            result["champions"] = len(jsons)
        else:
            try:
                count = storage.migrate_champions_json_to_npz(champ_dir)
                if count:
                    print(f"  Migrated {count} champion files → .npz")
                result["champions"] = count
            except Exception as e:
                print(f"  ERROR migrating champions: {e}")
    else:
        print(f"  No champions directory found in {directory}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate legacy JSON results to Parquet/.npz"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=None,
        help="Results directory to migrate (default: migrate all under results/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be migrated",
    )
    args = parser.parse_args()

    results_dir = os.path.join(_lib_dir, "..", "results")
    if args.directory:
        results_dir = args.directory

    if not os.path.isdir(results_dir):
        print(f"Directory not found: {results_dir}")
        sys.exit(1)

    targets = []
    if args.directory or os.path.isfile(os.path.join(results_dir, "statistics.json")):
        targets.append(results_dir)
    else:
        for entry in sorted(os.listdir(results_dir)):
            path = os.path.join(results_dir, entry)
            if os.path.isdir(path):
                targets.append(path)

    totals = {"stats": 0, "champions": 0}
    for t in targets:
        name = os.path.basename(t)
        print(f"\n{'=' * 50}")
        print(f"Migrating: {name}")
        print(f"{'=' * 50}")
        r = migrate_one(t, dry_run=args.dry_run)
        totals["stats"] += r["stats"]
        totals["champions"] += r["champions"]

    print(f"\n{'=' * 50}")
    if args.dry_run:
        print(
            f"DRY RUN complete. Would migrate {totals['stats']} stat files "
            f"and {totals['champions']} champion files."
        )
    else:
        print(
            f"Migration complete. Migrated {totals['stats']} stat files "
            f"and {totals['champions']} champion files."
        )


if __name__ == "__main__":
    main()
