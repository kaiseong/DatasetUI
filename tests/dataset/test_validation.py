"""Tests for dataset structural validation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot_dataset_editor.dataset.loader import load_document
from lerobot_dataset_editor.dataset.validation import validate_dataset
from lerobot_dataset_editor.dataset.version import UnsupportedVersionError


class TestValidationPasses:
    """Valid fixtures must pass all checks."""

    def test_valid_v21_passes(self, v21_fixture: Path) -> None:
        result = validate_dataset(v21_fixture)
        assert result.valid is True
        assert result.errors == ()

    def test_valid_v30_passes(self, v30_fixture: Path) -> None:
        result = validate_dataset(v30_fixture)
        assert result.valid is True
        assert result.errors == ()

    def test_valid_v30_annotated_passes(self, v30_annotated_fixture: Path) -> None:
        result = validate_dataset(v30_annotated_fixture)
        assert result.valid is True
        assert result.errors == ()


class TestValidationDetectsProblems:
    """Structural problems are detected and reported."""

    def test_unknown_version_rejected(self, corrupt_fixture: Path) -> None:
        with pytest.raises(UnsupportedVersionError, match="v99.0"):
            validate_dataset(corrupt_fixture)

    def test_corrupt_parquet_detected(self, tmp_path: Path) -> None:
        """Fixture with valid info.json but unreadable Parquet."""
        meta = tmp_path / "meta"
        meta.mkdir()
        info = {
            "codebase_version": "v3.0",
            "fps": 10,
            "total_episodes": 1,
            "total_frames": 5,
            "chunks_size": 1000,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": None,
            "features": {
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            },
        }
        (meta / "info.json").write_text(json.dumps(info))
        # Write corrupt parquet
        data_dir = tmp_path / "data" / "chunk-000"
        data_dir.mkdir(parents=True)
        (data_dir / "file-000.parquet").write_bytes(b"NOT_PARQUET")
        # Write minimal episodes parquet
        ep_dir = tmp_path / "meta" / "episodes" / "chunk-000"
        ep_dir.mkdir(parents=True)
        ep_table = pa.table({"episode_index": [0], "length": [5], "tasks": [["pick"]], "data/chunk_index": [0], "data/file_index": [0]})
        pq.write_table(ep_table, ep_dir / "file-000.parquet")

        result = validate_dataset(tmp_path)
        assert result.valid is False
        assert any("parquet" in e.lower() or "corrupt" in e.lower() for e in result.errors)

    def test_missing_episode_detected(self, tmp_path: Path) -> None:
        """Synthetic fixture with episode 0 present but episode 1 missing from sequence."""
        meta = tmp_path / "meta"
        meta.mkdir()
        info = {
            "codebase_version": "v2.1",
            "fps": 10,
            "total_episodes": 3,
            "total_frames": 15,
            "total_tasks": 1,
            "total_videos": 0,
            "total_chunks": 1,
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": None,
            "features": {
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            },
        }
        (meta / "info.json").write_text(json.dumps(info))
        # episodes.jsonl missing episode 1
        episodes = [
            {"episode_index": 0, "length": 5, "tasks": ["pick"]},
            {"episode_index": 2, "length": 5, "tasks": ["pick"]},
        ]
        (meta / "episodes.jsonl").write_text(
            "\n".join(json.dumps(e) for e in episodes) + "\n"
        )
        # Write data files for ep 0 and 2
        data_dir = tmp_path / "data" / "chunk-000"
        data_dir.mkdir(parents=True)
        for ep_idx in (0, 2):
            table = pa.table({
                "index": pa.array(range(5), type=pa.int64()),
                "episode_index": pa.array([ep_idx] * 5, type=pa.int64()),
                "frame_index": pa.array(range(5), type=pa.int64()),
                "timestamp": pa.array([i / 10.0 for i in range(5)], type=pa.float32()),
            })
            pq.write_table(table, data_dir / f"episode_{ep_idx:06d}.parquet")

        result = validate_dataset(tmp_path)
        assert result.valid is False
        assert any("missing" in e.lower() or "gap" in e.lower() for e in result.errors)

    def test_duplicate_episode_detected(self, tmp_path: Path) -> None:
        """Fixture with ep 0 appearing twice in episodes list."""
        meta = tmp_path / "meta"
        meta.mkdir()
        info = {
            "codebase_version": "v2.1",
            "fps": 10,
            "total_episodes": 2,
            "total_frames": 10,
            "total_tasks": 1,
            "total_videos": 0,
            "total_chunks": 1,
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": None,
            "features": {
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            },
        }
        (meta / "info.json").write_text(json.dumps(info))
        episodes = [
            {"episode_index": 0, "length": 5, "tasks": ["pick"]},
            {"episode_index": 0, "length": 5, "tasks": ["pick"]},
        ]
        (meta / "episodes.jsonl").write_text(
            "\n".join(json.dumps(e) for e in episodes) + "\n"
        )
        data_dir = tmp_path / "data" / "chunk-000"
        data_dir.mkdir(parents=True)
        table = pa.table({
            "index": pa.array(range(5), type=pa.int64()),
            "episode_index": pa.array([0] * 5, type=pa.int64()),
            "frame_index": pa.array(range(5), type=pa.int64()),
            "timestamp": pa.array([i / 10.0 for i in range(5)], type=pa.float32()),
        })
        pq.write_table(table, data_dir / "episode_000000.parquet")

        result = validate_dataset(tmp_path)
        assert result.valid is False
        assert any("duplicate" in e.lower() for e in result.errors)

    def test_nonmonotonic_timestamp_detected(self, tmp_path: Path) -> None:
        """Fixture with non-monotonic timestamps."""
        meta = tmp_path / "meta"
        meta.mkdir()
        info = {
            "codebase_version": "v2.1",
            "fps": 10,
            "total_episodes": 1,
            "total_frames": 3,
            "total_tasks": 1,
            "total_videos": 0,
            "total_chunks": 1,
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": None,
            "features": {
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            },
        }
        (meta / "info.json").write_text(json.dumps(info))
        episodes = [{"episode_index": 0, "length": 3, "tasks": ["pick"]}]
        (meta / "episodes.jsonl").write_text(json.dumps(episodes[0]) + "\n")
        data_dir = tmp_path / "data" / "chunk-000"
        data_dir.mkdir(parents=True)
        # Non-monotonic: 0.0, 0.3, 0.2
        table = pa.table({
            "index": pa.array([0, 1, 2], type=pa.int64()),
            "episode_index": pa.array([0, 0, 0], type=pa.int64()),
            "frame_index": pa.array([0, 1, 2], type=pa.int64()),
            "timestamp": pa.array([0.0, 0.3, 0.2], type=pa.float32()),
        })
        pq.write_table(table, data_dir / "episode_000000.parquet")

        result = validate_dataset(tmp_path)
        assert result.valid is False
        assert any("monoton" in e.lower() or "timestamp" in e.lower() for e in result.errors)

    def test_schema_drift_detected(self, tmp_path: Path) -> None:
        """info.json declares a feature column not present in Parquet."""
        meta = tmp_path / "meta"
        meta.mkdir()
        info = {
            "codebase_version": "v2.1",
            "fps": 10,
            "total_episodes": 1,
            "total_frames": 3,
            "total_tasks": 1,
            "total_videos": 0,
            "total_chunks": 1,
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": None,
            "features": {
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "ghost_column": {"dtype": "float32", "shape": [1], "names": None},
            },
        }
        (meta / "info.json").write_text(json.dumps(info))
        episodes = [{"episode_index": 0, "length": 3, "tasks": ["pick"]}]
        (meta / "episodes.jsonl").write_text(json.dumps(episodes[0]) + "\n")
        data_dir = tmp_path / "data" / "chunk-000"
        data_dir.mkdir(parents=True)
        # Parquet lacks ghost_column
        table = pa.table({
            "index": pa.array([0, 1, 2], type=pa.int64()),
            "episode_index": pa.array([0, 0, 0], type=pa.int64()),
            "frame_index": pa.array([0, 1, 2], type=pa.int64()),
            "timestamp": pa.array([0.0, 0.1, 0.2], type=pa.float32()),
        })
        pq.write_table(table, data_dir / "episode_000000.parquet")

        result = validate_dataset(tmp_path)
        assert result.valid is False
        assert any("drift" in e.lower() or "ghost_column" in e.lower() for e in result.errors)

    def test_missing_video_detected(self, tmp_path: Path) -> None:
        """Fixture with video_path set but no .mp4 files exist."""
        meta = tmp_path / "meta"
        meta.mkdir()
        info = {
            "codebase_version": "v2.1",
            "fps": 10,
            "total_episodes": 1,
            "total_frames": 3,
            "total_tasks": 1,
            "total_videos": 1,
            "total_chunks": 1,
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "observation.images.top": {"dtype": "image", "shape": [2, 2, 3], "names": None},
            },
        }
        (meta / "info.json").write_text(json.dumps(info))
        episodes = [{"episode_index": 0, "length": 3, "tasks": ["pick"]}]
        (meta / "episodes.jsonl").write_text(json.dumps(episodes[0]) + "\n")
        data_dir = tmp_path / "data" / "chunk-000"
        data_dir.mkdir(parents=True)
        table = pa.table({
            "index": pa.array([0, 1, 2], type=pa.int64()),
            "episode_index": pa.array([0, 0, 0], type=pa.int64()),
            "frame_index": pa.array([0, 1, 2], type=pa.int64()),
            "timestamp": pa.array([0.0, 0.1, 0.2], type=pa.float32()),
        })
        pq.write_table(table, data_dir / "episode_000000.parquet")
        # No videos directory at all

        result = validate_dataset(tmp_path)
        assert result.valid is False
        assert any("video" in e.lower() or "missing" in e.lower() for e in result.errors)


class TestValidationWithOfficialFixtures:
    """Official fixture directories should validate cleanly."""

    @pytest.fixture
    def official_v21(self) -> Path | None:
        p = Path(__file__).resolve().parents[1] / "fixtures" / "official" / "v21_v033"
        if not p.exists():
            pytest.skip("official v21 fixture not present")
        return p

    @pytest.fixture
    def official_v30(self) -> Path | None:
        p = Path(__file__).resolve().parents[1] / "fixtures" / "official" / "v30_v060"
        if not p.exists():
            pytest.skip("official v30 fixture not present")
        return p

    def test_official_v21_passes(self, official_v21: Path) -> None:
        result = validate_dataset(official_v21)
        assert result.valid is True

    def test_official_v30_passes(self, official_v30: Path) -> None:
        result = validate_dataset(official_v30)
        assert result.valid is True
