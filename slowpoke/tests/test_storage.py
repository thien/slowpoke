"""Tests for storage backends (Parquet, .npz, migration)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from slowpoke.core.storage import (
    HAS_PARQUET,
    load_champion_npz,
    load_statistics_parquet,
    migrate_champions_json_to_npz,
    migrate_statistics_json_to_parquet,
    save_champion_npz,
    save_statistics_parquet,
)


class TestChampionNPZ(unittest.TestCase):
    """Test .npz champion storage."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        for f in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    def test_save_and_load_roundtrip(self):
        """Save and load coefficients: should match exactly."""
        coeffs = np.random.random(100).astype(np.float32)
        path = os.path.join(self.tmpdir, "test.npz")
        save_champion_npz(path, coeffs, {"pid": "42", "champ_score": 0.5})

        loaded = load_champion_npz(path)
        np.testing.assert_array_equal(loaded["coefficients"], coeffs)
        self.assertEqual(loaded["pid"], "42")
        self.assertEqual(float(loaded["champ_score"]), 0.5)

    def test_save_adds_npz_suffix(self):
        """Saving without .npz suffix should add it."""
        path = os.path.join(self.tmpdir, "test")
        result = save_champion_npz(path, np.zeros(5, dtype=np.float32))
        self.assertTrue(result.endswith(".npz"))
        self.assertTrue(os.path.isfile(result))

    def test_load_nonexistent_file(self):
        """Loading a nonexistent .npz should raise."""
        with self.assertRaises(FileNotFoundError):
            load_champion_npz("/nonexistent/file.npz")

    def test_meta_optional(self):
        """Saving without metadata should still work."""
        coeffs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        path = os.path.join(self.tmpdir, "meta.npz")
        save_champion_npz(path, coeffs)
        loaded = load_champion_npz(path)
        np.testing.assert_array_equal(loaded["coefficients"], coeffs)


@unittest.skipIf(not HAS_PARQUET, "pyarrow not installed")
class TestStatisticsParquet(unittest.TestCase):
    """Test Parquet statistics storage."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir)

    def _make_gen_stats(self, num_gens=2, games_per_gen=3):
        """Create sample generation stats for testing."""
        stats = []
        for g in range(num_gens):
            games = []
            for i in range(games_per_gen):
                games.append(
                    {
                        "game": {
                            "Winner": i % 3 - 1,
                            "_id": f"g{g}_game{i}",
                            "Moves": [f"m{j}" for j in range(i + 1)],
                        },
                        "black": f"black_{i}",
                        "white": f"white_{i}",
                        "duration": f"00:00:{i:02d}",
                        "black_elo": 1200.0 + i,
                        "white_elo": 1200.0 - i,
                    }
                )
            stats.append({"games": games, "durationInSeconds": str(g * 10)})
        return stats

    def test_save_and_load_roundtrip(self):
        """Save then load: game count and winner should match."""
        stats = self._make_gen_stats(2, 3)
        combined = save_statistics_parquet(self.tmpdir, stats)
        self.assertTrue(os.path.isfile(combined))

        rows = load_statistics_parquet(self.tmpdir)
        self.assertEqual(len(rows), 6)  # 2 gen x 3 games

    def test_load_single_gen(self):
        """Loading a single generation should return only its games."""
        stats = self._make_gen_stats(3, 2)
        save_statistics_parquet(self.tmpdir, stats)

        gen1_rows = load_statistics_parquet(self.tmpdir, gen=1)
        self.assertEqual(len(gen1_rows), 2)
        for r in gen1_rows:
            self.assertEqual(r["gen"], 1)

    def test_per_gen_files_created(self):
        """Each generation should have its own parquet file."""
        stats = self._make_gen_stats(3, 2)
        save_statistics_parquet(self.tmpdir, stats)

        for g in range(3):
            path = os.path.join(self.tmpdir, "statistics", f"gen_{g:04d}.parquet")
            self.assertTrue(os.path.isfile(path))

    def test_all_parquet_created(self):
        """Combined all.parquet should exist."""
        stats = self._make_gen_stats(2, 2)
        save_statistics_parquet(self.tmpdir, stats)
        combined = os.path.join(self.tmpdir, "statistics", "all.parquet")
        self.assertTrue(os.path.isfile(combined))

    def test_empty_generation(self):
        """Generation with no games should not crash."""
        stats = [{"games": [], "durationInSeconds": "0"}]
        save_statistics_parquet(self.tmpdir, stats)


@unittest.skipIf(not HAS_PARQUET, "pyarrow not installed")
class TestMigration(unittest.TestCase):
    """Test JSON → Parquet / .npz migration."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir)

    def test_migrate_statistics_json(self):
        """Migrate statistics.json to parquet."""
        json_path = os.path.join(self.tmpdir, "statistics.json")
        legacy_stats = [
            {
                "stats": [],
                "games": [
                    {
                        "game": {"Winner": 0, "_id": "g0_0", "Moves": ["a-b"]},
                        "black": "b0",
                        "white": "w0",
                        "duration": "00:00:01",
                    }
                ],
                "durationInSeconds": "1",
            }
        ]
        with open(json_path, "w") as f:
            json.dump(legacy_stats, f)

        result = migrate_statistics_json_to_parquet(self.tmpdir)
        self.assertTrue(result.endswith("all.parquet"))
        self.assertTrue(os.path.isfile(result))

    def test_migrate_champions_json(self):
        """Migrate champion JSON files to .npz."""
        champ_dir = os.path.join(self.tmpdir, "champions")
        os.makedirs(champ_dir)

        # Write a legacy champion JSON
        champ_data = {
            "0": {
                "pid": 42,
                "coefficents": [0.1, 0.2, 0.3],
                "champ_range": [[1, 2]],
                "champ_score": 0.75,
            }
        }
        with open(os.path.join(champ_dir, "0.json"), "w") as f:
            json.dump(champ_data, f)

        count = migrate_champions_json_to_npz(champ_dir)
        self.assertEqual(count, 1)
        self.assertTrue(os.path.isfile(os.path.join(champ_dir, "0.npz")))

    def test_migrate_skips_existing_npz(self):
        """Already-migrated champion files should be skipped."""
        champ_dir = os.path.join(self.tmpdir, "champions")
        os.makedirs(champ_dir)

        # Create both .json and .npz
        with open(os.path.join(champ_dir, "0.json"), "w") as f:
            json.dump({"0": {"coefficents": [1.0]}}, f)
        np.savez_compressed(
            os.path.join(champ_dir, "0.npz"), coefficients=np.array([1.0])
        )

        count = migrate_champions_json_to_npz(champ_dir)
        self.assertEqual(count, 0)  # nothing new migrated

    def test_no_statistics_json(self):
        """No statistics.json should be harmless."""
        result = migrate_statistics_json_to_parquet(self.tmpdir)
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
