"""Tests for fixture materialization and Space parity coverage.

Each test is mapped to a Space parity feature ID in contracts/space-parity.yaml.
"""

from __future__ import annotations

import json

import pyarrow.parquet as pq
import pytest


class TestV30FixtureStructure:
    """Validate that v3.0 fixtures have the structure the Space expects."""

    def test_v30_fixture_episodes_parquet(self, v30_fixture):
        """SP-02: episode parquet loading."""
        ep_path = v30_fixture / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        assert ep_path.exists()
        table = pq.read_table(ep_path)
        assert "episode_index" in table.column_names
        assert "length" in table.column_names
        assert "dataset_from_index" in table.column_names
        assert "dataset_to_index" in table.column_names
        assert table.num_rows == 3  # 3 episodes

    def test_v30_fixture_data_parquet(self, v30_fixture):
        """SP-03: data parquet loading."""
        data_path = v30_fixture / "data" / "chunk-000" / "file-000.parquet"
        assert data_path.exists()
        table = pq.read_table(data_path)
        assert "episode_index" in table.column_names
        assert "frame_index" in table.column_names
        assert "timestamp" in table.column_names
        assert "observation.state" in table.column_names
        assert "action" in table.column_names
        assert table.num_rows == 30  # 30 total frames

    def test_v30_fixture_tasks(self, v30_fixture):
        """SP-04: tasks metadata."""
        tasks_path = v30_fixture / "meta" / "tasks.parquet"
        assert tasks_path.exists()
        table = pq.read_table(tasks_path)
        assert "task_index" in table.column_names
        assert "task" in table.column_names
        assert table.num_rows == 2  # 2 tasks

    def test_v30_image_backed_fixture_has_null_video_path(self, v30_fixture):
        """SP-05: official writers use null when no feature is video-backed."""
        info = json.loads((v30_fixture / "meta" / "info.json").read_text())
        assert info["video_path"] is None

    def test_v30_fixture_episode_filtering(self, v30_fixture):
        """SP-07: episode index filtering."""
        data_path = v30_fixture / "data" / "chunk-000" / "file-000.parquet"
        table = pq.read_table(data_path)
        ep_col = table.column("episode_index").to_pylist()
        unique_eps = set(ep_col)
        assert unique_eps == {0, 1, 2}

    def test_v30_fixture_timestamps(self, v30_fixture):
        """SP-08: timestamp column."""
        data_path = v30_fixture / "data" / "chunk-000" / "file-000.parquet"
        table = pq.read_table(data_path)
        timestamps = table.column("timestamp").to_pylist()
        # Timestamps should start at 0 for each episode
        assert timestamps[0] == 0.0
        # Should be monotonically increasing within an episode
        ep_indices = table.column("episode_index").to_pylist()
        prev_ts = -1.0
        prev_ep = -1
        for ts, ep in zip(timestamps, ep_indices):
            if ep != prev_ep:
                prev_ts = -1.0
                prev_ep = ep
            assert ts > prev_ts
            prev_ts = ts

    def test_v30_fixture_frame_index(self, v30_fixture):
        """SP-09: frame_index column."""
        data_path = v30_fixture / "data" / "chunk-000" / "file-000.parquet"
        table = pq.read_table(data_path)
        frame_indices = table.column("frame_index").to_pylist()
        ep_indices = table.column("episode_index").to_pylist()
        # frame_index should start at 0 for each episode
        prev_ep = -1
        for fi, ep in zip(frame_indices, ep_indices):
            if ep != prev_ep:
                assert fi == 0
                prev_ep = ep

    def test_v30_fixture_stats(self, v30_fixture):
        """SP-10: stats.json loading."""
        stats_path = v30_fixture / "meta" / "stats.json"
        assert stats_path.exists()
        stats = json.loads(stats_path.read_text())
        assert "observation.state" in stats
        assert "action" in stats
        for key in ("observation.state", "action"):
            assert "min" in stats[key]
            assert "max" in stats[key]
            assert "mean" in stats[key]
            assert "std" in stats[key]

    def test_v30_fixture_multi_task(self, v30_fixture):
        """SP-13: multi-task dataset."""
        info = json.loads((v30_fixture / "meta" / "info.json").read_text())
        assert info["total_tasks"] == 2

        data_path = v30_fixture / "data" / "chunk-000" / "file-000.parquet"
        table = pq.read_table(data_path)
        task_indices = table.column("task_index").to_pylist()
        unique_tasks = set(task_indices)
        assert len(unique_tasks) == 2


class TestV30AnnotatedFixture:
    """Validate annotated fixture structure."""

    def test_v30_annotated_language_persistent(self, v30_annotated_fixture):
        """SP-11: language_persistent column."""
        data_path = v30_annotated_fixture / "data" / "chunk-000" / "file-000.parquet"
        table = pq.read_table(data_path)
        assert "language_persistent" in table.column_names
        # Check that some rows have non-empty annotations
        persistent = table.column("language_persistent").to_pylist()
        non_empty = [p for p in persistent if p != "[]"]
        assert len(non_empty) > 0

    def test_v30_annotated_language_events(self, v30_annotated_fixture):
        """SP-12: language_events column."""
        data_path = v30_annotated_fixture / "data" / "chunk-000" / "file-000.parquet"
        table = pq.read_table(data_path)
        assert "language_events" in table.column_names
        events = table.column("language_events").to_pylist()
        non_empty = [e for e in events if e != "[]"]
        assert len(non_empty) > 0

    def test_v30_annotated_language_content_structure(self, v30_annotated_fixture):
        """Verify language content is valid JSON with correct structure."""
        data_path = v30_annotated_fixture / "data" / "chunk-000" / "file-000.parquet"
        table = pq.read_table(data_path)
        persistent = table.column("language_persistent").to_pylist()
        non_empty = [p for p in persistent if p != "[]"]
        assert len(non_empty) > 0

        # Arrow stores canonical list<struct> rows directly (not JSON strings).
        rows = non_empty[0]
        assert isinstance(rows, list)
        assert len(rows) > 0
        row = rows[0]
        assert "role" in row
        assert "content" in row
        assert "style" in row
        assert row["style"] in ("subtask", "plan", "memory", "motion", "task_aug")


class TestV21FixtureStructure:
    """Validate v2.1 fixture structure."""

    def test_v21_fixture_has_image_feature(self, v21_fixture):
        """v2.1 datasets use image dtype (not video)."""
        info = json.loads((v21_fixture / "meta" / "info.json").read_text())
        image_features = [k for k, v in info["features"].items() if v["dtype"] == "image"]
        assert len(image_features) >= 1
        assert "observation.images.top" in image_features

    def test_v21_fixture_uses_null_video_path_for_embedded_images(self, v21_fixture):
        """Official v0.3.3 writes null when no feature is video-backed."""
        info = json.loads((v21_fixture / "meta" / "info.json").read_text())
        assert info["total_videos"] == 0
        assert info["video_path"] is None

    def test_v21_fixture_data_parquet_valid(self, v21_fixture):
        """v2.1 stores one loadable parquet file per episode."""
        paths = sorted((v21_fixture / "data" / "chunk-000").glob("episode_*.parquet"))
        assert [path.name for path in paths] == ["episode_000000.parquet", "episode_000001.parquet"]
        tables = [pq.read_table(path) for path in paths]
        assert [table.num_rows for table in tables] == [10, 10]
        assert sum(table.num_rows for table in tables) == 20


class TestFixtureDeterminism:
    """Verify fixture generation is deterministic."""

    def test_v30_fixture_is_deterministic(self, tmp_path):
        """Same seed should produce identical fixtures."""
        from lerobot_dataset_editor.fixtures import materialize_v30_fixture

        dir1 = materialize_v30_fixture(tmp_path / "run1")
        dir2 = materialize_v30_fixture(tmp_path / "run2")

        info1 = (dir1 / "meta" / "info.json").read_text()
        info2 = (dir2 / "meta" / "info.json").read_text()
        assert info1 == info2

        # Compare parquet content
        t1 = pq.read_table(dir1 / "data" / "chunk-000" / "file-000.parquet")
        t2 = pq.read_table(dir2 / "data" / "chunk-000" / "file-000.parquet")
        assert t1.equals(t2)

    def test_v21_fixture_is_deterministic(self, tmp_path):
        """Same seed should produce identical v2.1 fixtures."""
        from lerobot_dataset_editor.fixtures import materialize_v21_fixture

        dir1 = materialize_v21_fixture(tmp_path / "run1")
        dir2 = materialize_v21_fixture(tmp_path / "run2")

        info1 = (dir1 / "meta" / "info.json").read_text()
        info2 = (dir2 / "meta" / "info.json").read_text()
        assert info1 == info2


class TestFixtureProvenance:
    """Verify provenance documentation."""

    def test_v21_provenance(self, v21_fixture):
        prov = (v21_fixture / "PROVENANCE.md").read_text()
        assert "CONTRACT-DERIVED" in prov
        assert "NOT generated by the official" in prov

    def test_v30_provenance(self, v30_fixture):
        prov = (v30_fixture / "PROVENANCE.md").read_text()
        assert "CONTRACT-DERIVED" in prov
        assert "NOT generated by the official" in prov

    def test_corrupt_provenance(self, corrupt_fixture):
        prov = (corrupt_fixture / "PROVENANCE.md").read_text()
        assert "INTENTIONALLY CORRUPT" in prov
