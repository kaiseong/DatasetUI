from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from datasetui import merge_writer
from datasetui.merge_writer import write_preserved_merge
from datasetui.transform_errors import CurationTransformError


VIDEO_KEY = "observation.images.top"


def _data(episode: int, timestamps: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "episode_index": np.full(len(timestamps), episode, dtype=np.int64),
            "frame_index": np.arange(10, 10 + len(timestamps), dtype=np.int64),
            "timestamp": np.asarray(timestamps, dtype=np.float32),
            "index": np.arange(100, 100 + len(timestamps), dtype=np.int64),
            "task_index": np.zeros(len(timestamps), dtype=np.int64),
            "action": [[float(value)] for value in range(len(timestamps))],
        }
    )


class _FakeMergedSource:
    def __init__(self, root: Path, *, version: str) -> None:
        self.root = root
        self.version = version
        self.fps = 10.0
        self.video_keys = [VIDEO_KEY]
        self.tasks = {0: "pick"}
        self._episodes = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.sources = [SimpleNamespace(root=root), SimpleNamespace(root=root)]
        self.total_episodes = len(self._episodes)
        self.info = {
            "codebase_version": version,
            "fps": self.fps,
            "robot_type": "rby1",
            "chunks_size": 1000,
            "total_episodes": 4,
            "features": {
                "timestamp": {"dtype": "float32", "shape": [1]},
                "index": {"dtype": "int64", "shape": [1]},
                "episode_index": {"dtype": "int64", "shape": [1]},
                "frame_index": {"dtype": "int64", "shape": [1]},
                "task_index": {"dtype": "int64", "shape": [1]},
                "action": {"dtype": "float32", "shape": [1]},
                VIDEO_KEY: {
                    "dtype": "video",
                    "shape": [2, 2, 3],
                    "info": {"video.codec": "av1", "video.fps": 10},
                },
            },
        }
        (root / "meta").mkdir(exist_ok=True)
        (root / "meta/info.json").write_text(json.dumps(self.info))
        self._videos: dict[tuple[int, int], Path] = {}
        if version == "v3.0":
            for source_number in (0, 1):
                path = root / f"source-{source_number}.mp4"
                path.write_bytes((f"av1-source-{source_number}-" * 100).encode())
                self._videos[(source_number, 0)] = path
                self._videos[(source_number, 1)] = path
        else:
            for source_number, local_index in self._episodes:
                path = root / f"source-{source_number}-episode-{local_index}.mp4"
                path.write_bytes(
                    (f"av1-source-{source_number}-episode-{local_index}-" * 50).encode()
                )
                self._videos[(source_number, local_index)] = path

    def episode(self, output_index: int):
        source_number, local_index = self._episodes[output_index]
        timestamps = [1.25, 1.35] if local_index == 0 else [4.5, 4.6]
        metadata = {
            "episode_index": local_index,
            "length": 2,
            "tasks": ["pick"],
            "stats/index/min": [100],
        }
        if self.version == "v3.0":
            prefix = f"videos/{VIDEO_KEY}"
            metadata.update(
                {
                    f"{prefix}/chunk_index": 7,
                    f"{prefix}/file_index": 9,
                    f"{prefix}/from_timestamp": 0.0 if local_index == 0 else 0.2,
                    f"{prefix}/to_timestamp": 0.2 if local_index == 0 else 0.4,
                }
            )
        return _data(local_index, timestamps), metadata

    def video_source(self, output_index: int, video_key: str, metadata):
        assert video_key == VIDEO_KEY
        source_number, local_index = self._episodes[output_index]
        return self._videos[(source_number, local_index)], local_index * 2


@pytest.fixture(autouse=True)
def _avoid_video_decode(monkeypatch):
    monkeypatch.setattr(merge_writer, "_write_stats", lambda *args, **kwargs: None)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v3_copies_each_shared_source_shard_once_and_preserves_offsets(
    tmp_path: Path, monkeypatch
) -> None:
    source = _FakeMergedSource(tmp_path, version="v3.0")
    destination = tmp_path / "output"
    monkeypatch.setattr(
        "datasetui.transforms._slice_video",
        lambda *args, **kwargs: pytest.fail("merge must not invoke the video encoder"),
    )

    write_preserved_merge(source=source, destination=destination)

    output_videos = sorted((destination / "videos").rglob("*.mp4"))
    assert len(output_videos) == 2
    assert [_sha(path) for path in output_videos] == [
        _sha(source._videos[(0, 0)]),
        _sha(source._videos[(1, 0)]),
    ]
    metadata = pd.read_parquet(destination / "meta/episodes/chunk-000/file-000.parquet")
    prefix = f"videos/{VIDEO_KEY}"
    assert metadata[f"{prefix}/file_index"].tolist() == [0, 0, 1, 1]
    assert metadata[f"{prefix}/from_timestamp"].tolist() == [0.0, 0.2, 0.0, 0.2]
    assert metadata[f"{prefix}/to_timestamp"].tolist() == [0.2, 0.4, 0.2, 0.4]
    assert not any(column.startswith("stats/") for column in metadata.columns)
    assert metadata["meta/episodes/chunk_index"].tolist() == [0, 0, 0, 0]
    assert metadata["meta/episodes/file_index"].tolist() == [0, 0, 0, 0]
    info = json.loads((destination / "meta/info.json").read_text())
    assert info["total_videos"] == 2
    written = pd.read_parquet(destination / "data/chunk-000/file-000.parquet")
    assert written["frame_index"].tolist() == [10, 11]
    assert written["timestamp"].tolist() == pytest.approx([1.25, 1.35])


def test_v21_copies_each_episode_video_byte_for_byte(tmp_path: Path) -> None:
    source = _FakeMergedSource(tmp_path, version="v2.1")
    destination = tmp_path / "output"

    write_preserved_merge(source=source, destination=destination)

    output_videos = sorted((destination / "videos").rglob("*.mp4"))
    assert len(output_videos) == 4
    info = json.loads((destination / "meta/info.json").read_text())
    assert info["total_videos"] == 4
    expected = [
        _sha(source._videos[source_episode]) for source_episode in source._episodes
    ]
    assert [_sha(path) for path in output_videos] == expected


def test_v21_merge_prefers_episode_statistics_without_full_recompute(
    tmp_path: Path, monkeypatch
) -> None:
    from datasetui import official_operations

    source = _FakeMergedSource(tmp_path, version="v2.1")
    destination = tmp_path / "output"
    captured = []

    def aggregate(actual_destination, episodes):
        captured.extend(episodes)
        return True

    monkeypatch.setattr(
        official_operations, "write_legacy_aggregated_statistics", aggregate
    )
    monkeypatch.setattr(
        merge_writer,
        "_write_stats",
        lambda *args, **kwargs: pytest.fail("must not fully recompute statistics"),
    )

    result = write_preserved_merge(source=source, destination=destination)

    assert result["statistics_reused"] is True
    assert result["statistics"] == {
        "policy": "lerobot-official-aggregate-v1",
        "source": "legacy-episode-statistics",
        "fallback": False,
    }
    assert [(root, index) for root, index, _ in captured] == [
        (tmp_path, 0),
        (tmp_path, 1),
        (tmp_path, 0),
        (tmp_path, 1),
    ]


def test_v3_rejects_missing_segment_timestamps(tmp_path: Path) -> None:
    source = _FakeMergedSource(tmp_path, version="v3.0")
    original_episode = source.episode

    def missing_timestamp(output_index: int):
        data, metadata = original_episode(output_index)
        metadata.pop(f"videos/{VIDEO_KEY}/to_timestamp")
        return data, metadata

    source.episode = missing_timestamp
    with pytest.raises(CurationTransformError, match="timestamps are missing"):
        write_preserved_merge(source=source, destination=tmp_path / "output")


def test_video_copy_rejects_symlink_source(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    link = tmp_path / "link.mp4"
    link.symlink_to(source)

    with pytest.raises(CurationTransformError, match="unavailable"):
        merge_writer._copy_videos([(link, tmp_path / "output.mp4", tmp_path)], None)


def test_video_copy_rejects_symlinked_parent_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "source.mp4").write_bytes(b"source")
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "videos").symlink_to(outside, target_is_directory=True)

    with pytest.raises(CurationTransformError, match="unavailable"):
        merge_writer._copy_videos(
            [
                (
                    root / "videos/source.mp4",
                    tmp_path / "output.mp4",
                    root,
                )
            ],
            None,
        )


def test_video_copy_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "source.mp4"
    os.mkfifo(fifo)

    with pytest.raises(CurationTransformError, match="unavailable"):
        merge_writer._copy_videos([(fifo, tmp_path / "output.mp4", tmp_path)], None)


def test_progress_failure_aborts_current_video_copy(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"x" * (merge_writer._COPY_CHUNK_BYTES + 1))
    destination = tmp_path / "output.mp4"

    def lose_lease(progress):
        if progress["completed"] > 0:
            raise RuntimeError("lease lost")

    with pytest.raises(RuntimeError, match="lease lost"):
        merge_writer._copy_videos([(source, destination, tmp_path)], lose_lease)
    assert not destination.exists()
