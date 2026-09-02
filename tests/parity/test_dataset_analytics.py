from __future__ import annotations

import json
import math
from pathlib import Path

import lerobot_dataset_editor.parity.insights as insights_module
from lerobot_dataset_editor.parity.common import (episode_records, evenly_sample,
                                                  evenly_sample_indices,
                                                  read_sampled_episodes)
from lerobot_dataset_editor.parity.dataset import dataset_episode, dataset_summary
from lerobot_dataset_editor.parity.insights import dataset_analytics


def test_episode_and_summary_are_frame_complete_and_json_finite(v30_fixture: Path) -> None:
    summary = dataset_summary(v30_fixture)
    episode = dataset_episode(v30_fixture, 0)
    assert summary["total_episodes"] == 3
    assert summary["total_tasks"] == 2
    assert episode["frame_count"] == 10
    assert episode["timestamps"][0] == 0
    assert len(episode["action"]) == 4
    assert len(episode["state"]) == 4
    assert episode["episode_count"] == 3
    assert len(episode["media"]) == 2
    assert "NaN" not in json.dumps(episode, allow_nan=False)


def test_quality_and_six_action_insights_follow_sampling_contract(v30_fixture: Path) -> None:
    result = dataset_analytics(v30_fixture)
    assert result["quality"]["sampling"] == {"max_episodes": 120, "max_frames_per_episode": 2500, "time_bins": 50}
    assert set(result["quality"]["action_insights"]) == {
        "variance", "autocorrelation", "distribution", "jerk",
        "speed_consistency", "temporal_alignment",
    }
    assert len(result["variance"]) == 50
    assert len(result["velocity"][0]["bins"]) == 30
    assert len(result["alignment"]) == 4
    assert result["quality"]["low_movement"] == sorted(result["quality"]["low_movement"], key=lambda item: item["score"])
    assert all(math.isfinite(item["score"]) for item in result["quality"]["low_movement"])
    assert all(item["name"].startswith("Episode ") for item in result["jerk"])
    assert {"inactive", "discrete", "std", "maxAbs", "lo", "hi", "motorRange"} <= set(result["velocity"][0])
    assert result["velocity"][0]["min"] == result["velocity"][0]["lo"]
    assert result["velocity"][0]["max"] == result["velocity"][0]["hi"]
    assert result["speed_cv"]["lo"] <= result["speed_cv"]["hi"]
    assert result["episode_lengths"]["episode_length_histogram"] == [{"bin_label": "1.0s", "count": 3}]
    encoded = json.dumps(result, allow_nan=False)
    assert "data:image" not in encoded


def test_analytics_reads_only_numeric_signal_columns(v30_fixture: Path, monkeypatch) -> None:
    requested: list[str] = []
    original = insights_module.read_sampled_episodes

    def recording_read(source: Path, records, columns, maximum_frames, **kwargs):
        requested.extend(list(columns or []))
        return original(source, records, columns, maximum_frames, **kwargs)

    monkeypatch.setattr(insights_module, "read_sampled_episodes", recording_read)
    dataset_analytics(v30_fixture)
    assert "action" in requested and "observation.state" in requested
    assert not any("images" in column for column in requested)


def test_bounded_reader_materializes_at_most_sampling_contract_rows(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    episode_count = 121
    frames_per_episode = 2501
    total_rows = episode_count * frames_per_episode
    info = {
        "codebase_version": "v3.0", "fps": 10, "total_episodes": episode_count,
        "total_frames": total_rows,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "features": {
            "episode_index": {"dtype": "int64", "shape": [1]},
            "action": {"dtype": "float32", "shape": [1], "names": ["joint"]},
        },
    }
    (tmp_path / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (tmp_path / "data" / "chunk-000").mkdir(parents=True)
    (tmp_path / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    metadata = pa.table({
        "episode_index": range(episode_count),
        "length": [frames_per_episode] * episode_count,
        "data/chunk_index": [0] * episode_count,
        "data/file_index": [0] * episode_count,
        "dataset_from_index": [episode * frames_per_episode for episode in range(episode_count)],
        "dataset_to_index": [(episode + 1) * frames_per_episode for episode in range(episode_count)],
    })
    pq.write_table(metadata, tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    episode_indices = pa.array([episode for episode in range(episode_count)
                                for _ in range(frames_per_episode)], type=pa.int64())
    action = pa.FixedSizeListArray.from_arrays(pa.array(range(total_rows), type=pa.float32()), 1)
    pq.write_table(pa.table({"episode_index": episode_indices, "action": action}),
                   tmp_path / "data" / "chunk-000" / "file-000.parquet",
                   row_group_size=frames_per_episode)

    selected = evenly_sample(episode_records(tmp_path), insights_module.MAX_EPISODES)
    metrics: dict[str, int] = {}
    sampled = read_sampled_episodes(tmp_path, selected, ["episode_index", "action"],
                                    insights_module.MAX_FRAMES_PER_EPISODE,
                                    metrics=metrics)
    materialized = sampled.to_pylist()

    assert len(selected) == 120
    assert selected[0]["episode_index"] == 0 and selected[-1]["episode_index"] == 120
    assert len(materialized) == 120 * 2500
    assert len(materialized) <= (insights_module.MAX_EPISODES
                                 * insights_module.MAX_FRAMES_PER_EPISODE)
    assert metrics == {"retained_rows": 120 * 2500, "peak_batch_rows": 2501}
    assert materialized[0]["action"] == [0.0]
    assert materialized[2499]["action"] == [2500.0]


def test_episode_records_falls_back_to_v2_file_metadata(v21_fixture: Path, tmp_path: Path) -> None:
    import shutil

    dataset = tmp_path / "v21"
    shutil.copytree(v21_fixture, dataset)
    episodes_path = dataset / "meta" / "episodes.jsonl"
    episodes_path.unlink()

    records = episode_records(dataset)

    assert records == [
        {"episode_index": 0, "length": 10},
        {"episode_index": 1, "length": 10},
    ]


def test_frame_index_sampling_is_bounded_for_huge_logical_episode() -> None:
    indices = evenly_sample_indices(10**12, 2500)

    assert len(indices) == 2500
    assert indices[0] == 0
    assert indices[-1] == 10**12 - 1
    assert indices == sorted(set(indices))


def test_analytics_never_full_reads_media_columns(v30_fixture: Path, monkeypatch) -> None:
    import pyarrow.parquet as pq

    calls: list[tuple[str, ...] | None] = []
    original = pq.read_table

    def guarded_read_table(where, columns=None, *args, **kwargs):
        calls.append(tuple(columns) if columns is not None else None)
        if columns is None or any("image" in column or "video" in column for column in columns):
            raise AssertionError("analytics attempted a full/media parquet read")
        return original(where, columns=columns, *args, **kwargs)

    monkeypatch.setattr(pq, "read_table", guarded_read_table)
    result = dataset_analytics(v30_fixture)

    assert result["doctor"]["status"] == "ok"
    assert all(columns is not None for columns in calls)


def test_image_gallery_visits_only_selected_episode_directories(tmp_path: Path) -> None:
    feature = "observation.images.front"
    selected = [0, 3000]
    for episode in (*selected, 1):
        directory = tmp_path / "images" / feature / f"episode_{episode:06d}"
        directory.mkdir(parents=True)
        for frame in range(4):
            (directory / f"frame_{frame:06d}.png").write_bytes(b"png")
    metrics: dict[str, int] = {}

    paths = insights_module._external_image_paths(tmp_path, feature, selected,
                                                   metrics=metrics)

    assert set(paths) == set(selected)
    assert metrics["visited_files"] == len(selected) * 4
    assert all("episode_000001" not in relative
               for pair in paths.values() for relative in pair)


def test_analytics_reads_each_episode_metadata_file_once(v30_fixture: Path, monkeypatch) -> None:
    import lerobot_dataset_editor.parity.common as common_module

    original = common_module.pq.ParquetFile
    metadata_opens: list[Path] = []

    def recording_parquet_file(where, *args, **kwargs):
        path = Path(where)
        if "meta/episodes" in path.as_posix():
            metadata_opens.append(path)
        return original(where, *args, **kwargs)

    monkeypatch.setattr(common_module.pq, "ParquetFile", recording_parquet_file)
    dataset_analytics(v30_fixture)

    metadata_files = sorted((v30_fixture / "meta" / "episodes").rglob("*.parquet"))
    assert sorted(metadata_opens) == metadata_files


def test_episode_reads_only_target_file_and_projected_columns(v21_fixture: Path, tmp_path: Path,
                                                               monkeypatch) -> None:
    import shutil
    import lerobot_dataset_editor.parity.common as common_module

    dataset = tmp_path / "viewer"
    shutil.copytree(v21_fixture, dataset)
    unrelated = dataset / "data" / "chunk-999" / "episode_999999.parquet"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_bytes(b"must not be opened")
    original = common_module.pq.ParquetFile
    opened: list[Path] = []
    projected: list[tuple[str, ...] | None] = []

    class RecordingFile:
        def __init__(self, where, *args, **kwargs):
            opened.append(Path(where))
            self._inner = original(where, *args, **kwargs)
        def __getattr__(self, name):
            return getattr(self._inner, name)
        def iter_batches(self, *args, **kwargs):
            columns = kwargs.get("columns")
            projected.append(tuple(columns) if columns is not None else None)
            return self._inner.iter_batches(*args, **kwargs)

    monkeypatch.setattr(common_module.pq, "ParquetFile", RecordingFile)
    episode = dataset_episode(dataset, 0)

    assert episode["episode_index"] == 0
    assert unrelated not in opened
    assert projected and all(columns is not None for columns in projected)
    assert all("task_index" not in columns for columns in projected if columns)
