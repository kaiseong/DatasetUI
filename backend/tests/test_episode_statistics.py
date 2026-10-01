from __future__ import annotations

import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from datasetui.episode_statistics import (
    EPISODE_STATISTICS_ENGINE,
    write_episode_statistics,
)
from datasetui.transform_errors import CurationTransformError
from datasetui.visual_statistics import (
    recompute_visual_statistics,
    recompute_visual_statistics_with_episodes,
)
from test_transforms import _write_v21
from test_validation_conversion import _add_shared_v3_video, _write_v3


def _hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _jsonlines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_v21_episode_statistics_jsonl_is_complete_and_loader_shaped(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    _write_v21(root)
    immutable_before = {
        name: digest
        for name, digest in _hashes(root).items()
        if name != "meta/episodes_stats.jsonl"
    }
    events: list[dict] = []

    result = write_episode_statistics(root, on_progress=events.append)

    path = root / "meta/episodes_stats.jsonl"
    rows = _jsonlines(path)
    assert result == {
        "engine": EPISODE_STATISTICS_ENGINE,
        "version": "v2.1",
        "episodes": 2,
        "format": "jsonl",
        "paths": ["meta/episodes_stats.jsonl"],
    }
    assert [row["episode_index"] for row in rows] == [0, 1]
    for row in rows:
        assert set(row) == {"episode_index", "stats"}
        assert {
            "min",
            "max",
            "mean",
            "std",
            "count",
            "q01",
            "q10",
            "q50",
            "q90",
            "q99",
        } <= set(row["stats"]["action"])
        assert row["stats"]["action"]["count"] == [10]
        assert row["stats"]["episode_index"]["min"] == [float(row["episode_index"])]
    assert rows[0]["stats"]["frame_index"]["min"] == [0.0]
    assert rows[0]["stats"]["frame_index"]["max"] == [9.0]
    assert immutable_before == {
        name: digest
        for name, digest in _hashes(root).items()
        if name != "meta/episodes_stats.jsonl"
    }
    assert events[-1]["completed"] == events[-1]["total"] == 2
    assert events[-1]["unit"] == "episodes"

    first_content = path.read_bytes()
    write_episode_statistics(root)
    assert path.read_bytes() == first_content


def test_v3_episode_statistics_are_flattened_into_episode_parquet(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "dataset"
    _write_v3(root)
    _add_shared_v3_video(root)
    metadata_path = root / "meta/episodes/chunk-000/file-000.parquet"
    before_metadata = pq.read_table(metadata_path)
    immutable_before = {
        name: digest
        for name, digest in _hashes(root).items()
        if not name.startswith("meta/episodes/")
    }
    original_open = av.open
    video_opens = 0

    def tracked_open(file, *args, **kwargs):
        nonlocal video_opens
        if Path(file) == root / "videos/observation.images.top/chunk-000/file-000.mp4":
            video_opens += 1
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(av, "open", tracked_open)

    result = write_episode_statistics(root)

    after = pq.read_table(metadata_path)
    assert result == {
        "engine": EPISODE_STATISTICS_ENGINE,
        "version": "v3.0",
        "episodes": 2,
        "format": "parquet-columns",
        "paths": ["meta/episodes/chunk-000/file-000.parquet"],
    }
    for name in before_metadata.column_names:
        assert before_metadata.column(name).equals(after.column(name))
    required = {
        "stats/action/min",
        "stats/action/std",
        "stats/action/count",
        "stats/action/q50",
        "stats/observation.images.top/min",
        "stats/observation.images.top/count",
        "stats/observation.images.top/q50",
    }
    assert required <= set(after.column_names)
    assert after.column("stats/action/count").to_pylist() == [[4], [4]]
    assert after.column("stats/action/min").to_pylist() == [[0.0], [4.0]]
    assert after.column("stats/observation.images.top/count").to_pylist() == [
        [4],
        [4],
    ]
    assert immutable_before == {
        name: digest
        for name, digest in _hashes(root).items()
        if not name.startswith("meta/episodes/")
    }
    assert video_opens == 1

    first_values = {
        name: after.column(name).to_pylist()
        for name in after.column_names
        if name.startswith("stats/")
    }
    write_episode_statistics(root)
    repeated = pq.read_table(metadata_path)
    assert len(repeated.column_names) == len(set(repeated.column_names))
    assert {
        name: repeated.column(name).to_pylist() for name in first_values
    } == first_values


def test_batched_visual_statistics_match_independent_episode_scans_and_keep_global_union(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "dataset"
    _write_v3(root)
    video_path = _add_shared_v3_video(root)
    metadata_path = root / "meta/episodes/chunk-000/file-000.parquet"
    metadata = pd.read_parquet(metadata_path)
    metadata["videos/observation.images.top/from_timestamp"] = [0.0, 0.0]
    metadata["videos/observation.images.top/to_timestamp"] = [0.4, 0.4]
    metadata.to_parquet(metadata_path, index=False)
    baseline = {
        episode: recompute_visual_statistics(root, episode_indices=[episode])
        for episode in (0, 1)
    }
    original_open = av.open
    video_opens = 0

    def tracked_open(file, *args, **kwargs):
        nonlocal video_opens
        if Path(file) == video_path:
            video_opens += 1
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(av, "open", tracked_open)
    global_stats, episode_stats = recompute_visual_statistics_with_episodes(root)

    assert episode_stats == baseline
    assert global_stats["observation.images.top"]["count"] == [4]
    assert episode_stats[0]["observation.images.top"]["count"] == [4]
    assert episode_stats[1]["observation.images.top"]["count"] == [4]
    assert video_opens == 1


def test_batched_visual_statistics_reject_missing_expected_frame(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_v3(root)
    video_path = _add_shared_v3_video(root)
    _rewrite_test_video(video_path, range(7))

    with pytest.raises(CurationTransformError, match="missing expected"):
        recompute_visual_statistics_with_episodes(root)


def test_episode_writer_rejects_incomplete_precomputed_visual_statistics(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    _write_v3(root)
    _add_shared_v3_video(root)

    with pytest.raises(CurationTransformError, match="features differ"):
        write_episode_statistics(root, visual_stats_by_episode={0: {}, 1: {}})


def test_batched_visual_statistics_checks_cancellation_during_decode_gap(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    _write_v3(root)
    video_path = _add_shared_v3_video(root)
    _rewrite_test_video(video_path, range(200))
    metadata_path = root / "meta/episodes/chunk-000/file-000.parquet"
    metadata = pd.read_parquet(metadata_path)
    metadata["videos/observation.images.top/from_timestamp"] = [0.0, 19.6]
    metadata["videos/observation.images.top/to_timestamp"] = [0.4, 20.0]
    metadata.to_parquet(metadata_path, index=False)
    reached_first_episode_end = False

    def cancel_in_gap(event: dict) -> None:
        nonlocal reached_first_episode_end
        if event["completed"] == 4:
            if reached_first_episode_end:
                raise RuntimeError("cancelled during decode gap")
            reached_first_episode_end = True

    with pytest.raises(RuntimeError, match="cancelled during decode gap"):
        recompute_visual_statistics_with_episodes(root, on_progress=cancel_in_gap)


def _rewrite_test_video(path: Path, values) -> None:
    path.unlink()
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width = 32
        stream.height = 24
        stream.pix_fmt = "yuv420p"
        for value in values:
            frame = av.VideoFrame.from_ndarray(
                np.full((24, 32, 3), int(value) % 256, dtype=np.uint8),
                format="rgb24",
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_episode_metadata_count_mismatch_does_not_write_partial_stats(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    _write_v21(root)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["total_episodes"] = 3
    info_path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(CurationTransformError, match="count differs"):
        write_episode_statistics(root)

    assert not (root / "meta/episodes_stats.jsonl").exists()
    assert list((root / "meta").glob(".episodes_stats.jsonl.episode-stats-*.tmp")) == []
