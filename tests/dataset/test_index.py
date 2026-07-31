"""Tests for incremental SQLite dataset index."""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from lerobot_dataset_editor.dataset.index import DatasetIndex


class TestDatasetIndex:
    """Incremental index creates, caches, and invalidates correctly."""

    def test_index_creates_db_at_specified_path(self, tmp_path: Path) -> None:
        db_path = tmp_path / "subdir" / "index.sqlite"
        idx = DatasetIndex(db_path=db_path)
        assert db_path.exists()

    def test_index_schema_migration(self, tmp_path: Path) -> None:
        db_path = tmp_path / "index.sqlite"
        idx = DatasetIndex(db_path=db_path)
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT schema_version FROM index_meta").fetchone()
        assert row is not None
        assert row[0] == 2
        conn.close()

    def test_migrates_pre_release_info_fingerprint_cache(self, tmp_path: Path) -> None:
        db_path = tmp_path / "legacy-index.sqlite"
        connection = sqlite3.connect(db_path)
        connection.executescript(
            """
            CREATE TABLE index_meta (schema_version INTEGER NOT NULL, created_at TEXT NOT NULL);
            INSERT INTO index_meta VALUES (1, '2024-01-01T00:00:00Z');
            CREATE TABLE datasets (
                id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL UNIQUE,
                info_fingerprint TEXT NOT NULL,
                detected_version TEXT NOT NULL,
                document_json TEXT NOT NULL,
                indexed_at TEXT NOT NULL
            );
            """
        )
        connection.close()

        index = DatasetIndex(db_path=db_path)
        columns = {
            row[1] for row in index._conn.execute("PRAGMA table_info(datasets)")
        }
        version = index._conn.execute(
            "SELECT schema_version FROM index_meta"
        ).fetchone()
        assert "source_fingerprint" in columns
        assert "info_fingerprint" not in columns
        assert version == (2,)
        index.close()

    def test_index_caches_document(self, v30_fixture: Path, tmp_path: Path) -> None:
        db_path = tmp_path / "index.sqlite"
        idx = DatasetIndex(db_path=db_path)
        doc1 = idx.get_or_build(v30_fixture)
        doc2 = idx.get_or_build(v30_fixture)
        assert doc1.version == doc2.version
        assert doc1.total_frames == doc2.total_frames
        assert doc1.source_path == doc2.source_path

    def test_index_invalidates_on_fingerprint_change(self, v21_fixture: Path, tmp_path: Path) -> None:
        db_path = tmp_path / "index.sqlite"
        idx = DatasetIndex(db_path=db_path)
        doc1 = idx.get_or_build(v21_fixture)
        # Touch info.json to change its mtime
        info_path = v21_fixture / "meta" / "info.json"
        time.sleep(0.05)
        info_path.write_text(info_path.read_text())
        doc2 = idx.get_or_build(v21_fixture)
        # Should still work (re-indexed)
        assert doc2.total_frames == doc1.total_frames
        # indexed_at should differ
        assert doc2.indexed_at != doc1.indexed_at

    def test_index_stores_parquet_metadata(self, v30_fixture: Path, tmp_path: Path) -> None:
        db_path = tmp_path / "index.sqlite"
        idx = DatasetIndex(db_path=db_path)
        idx.get_or_build(v30_fixture)
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT relative_path, num_rows FROM parquet_files").fetchall()
        conn.close()
        assert len(rows) > 0
        # At least the main data parquet should be recorded
        paths = [r[0] for r in rows]
        assert any("chunk-000" in p for p in paths)

    def test_index_stores_video_metadata_table(self, tmp_path: Path) -> None:
        """Even with no videos, the video_files table should exist."""
        db_path = tmp_path / "index.sqlite"
        idx = DatasetIndex(db_path=db_path)
        conn = sqlite3.connect(db_path)
        # Table must exist
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='video_files'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    @patch("pyarrow.parquet.read_table", side_effect=AssertionError("read_table called"))
    def test_index_reuses_cache_on_unchanged_files(self, mock_rt, v30_fixture: Path, tmp_path: Path) -> None:
        """Second load should use cache — no pq.read_metadata if fingerprint matches."""
        db_path = tmp_path / "index.sqlite"
        idx = DatasetIndex(db_path=db_path)
        # First build (need real pq access, so unpatch for first call)
        mock_rt.stop()
        doc1 = idx.get_or_build(v30_fixture)
        # Re-patch and verify second call is cache hit
        with patch("pyarrow.parquet.read_table", side_effect=AssertionError("read_table called")):
            doc2 = idx.get_or_build(v30_fixture)
        assert doc2.total_frames == doc1.total_frames

    def test_index_uses_xdg_cache_default(self, tmp_path: Path) -> None:
        """When no db_path specified, uses XDG_CACHE_HOME."""
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path / "xdg_cache")}):
            from lerobot_dataset_editor.dataset.index import default_index_path

            path = default_index_path()
            assert "lerobot-dataset-editor" in str(path)
            assert str(path).startswith(str(tmp_path / "xdg_cache"))


    def test_index_invalidates_when_data_parquet_changes(
        self, v21_fixture: Path, tmp_path: Path
    ) -> None:
        import shutil

        dataset = tmp_path / "dataset"
        shutil.copytree(v21_fixture, dataset)
        idx = DatasetIndex(db_path=tmp_path / "index.sqlite")
        first = idx.get_or_build(dataset)
        parquet = next((dataset / "data").rglob("*.parquet"))
        stat = parquet.stat()
        os.utime(parquet, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        second = idx.get_or_build(dataset)
        assert second.indexed_at != first.indexed_at
        idx.close()

    def test_index_ffprobes_video_metadata_without_decode(
        self, v21_fixture: Path, tmp_path: Path
    ) -> None:
        import json
        import shutil

        dataset = tmp_path / "video-dataset"
        shutil.copytree(v21_fixture, dataset)
        info_path = dataset / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        info["total_videos"] = 2
        info_path.write_text(json.dumps(info), encoding="utf-8")
        for episode in (0, 1):
            video = dataset / "videos" / "chunk-000" / "observation.images.top" / f"episode_{episode:06d}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"metadata-probe-only")

        log = tmp_path / "ffprobe-args.txt"
        ffprobe = tmp_path / "ffprobe"
        ffprobe.write_text(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" >> '" + str(log) + "'\n"
            "printf '%s\\n' '{\"format\":{\"duration\":\"1.25\"},\"streams\":[{\"codec_type\":\"video\",\"codec_name\":\"h264\",\"width\":640,\"height\":480}]}'\n",
            encoding="utf-8",
        )
        ffprobe.chmod(0o755)

        idx = DatasetIndex(db_path=tmp_path / "video-index.sqlite", ffprobe_path=str(ffprobe))
        idx.get_or_build(dataset)
        rows = idx._conn.execute(
            "SELECT relative_path, duration_sec, codec, width, height FROM video_files ORDER BY relative_path"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0][1:] == (1.25, "h264", 640, 480)
        arguments = log.read_text(encoding="utf-8")
        assert "-show_streams" in arguments
        assert "-show_format" in arguments
        assert "-show_frames" not in arguments
        assert "-read_intervals" not in arguments
        idx.close()
