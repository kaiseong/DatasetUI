from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import datasetui.dataset_io.source as dataset_source
from test_transforms import _settings, _write_v21

from datasetui.conversion import convert_dataset_to_v21
from datasetui.database import Database
from datasetui.datasets import inspect_dataset
from datasetui.validation import validate_dataset_root


def test_validation_levels_report_structural_failures_and_nonblocking_warnings(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    _write_v21(root)

    quick = validate_dataset_root(root, mode="quick")
    full = validate_dataset_root(root, mode="full")

    assert quick["passed"] is True
    assert full["passed"] is False
    assert any(item["code"] == "stats_load_failed" for item in full["issues"])

    path = root / "data/chunk-000/episode_000000.parquet"
    frame = pd.read_parquet(path)
    frame.loc[3, "timestamp"] = -1
    frame.to_parquet(path, index=False)
    broken = validate_dataset_root(root, mode="quick")
    assert broken["passed"] is False
    assert any(item["code"] == "timestamp_regression" for item in broken["issues"])


def _write_v3(root: Path) -> None:
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "rby1",
        "total_episodes": 2,
        "total_frames": 8,
        "total_tasks": 1,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 10,
        "splits": {"train": "0:2"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [1], "names": ["joint_0"]},
            "observation.state": {
                "dtype": "float32",
                "shape": [1],
                "names": ["joint_0"],
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    pd.DataFrame([{"task_index": 0, "task": "pick"}]).to_parquet(
        root / "meta/tasks.parquet", index=False
    )
    pd.DataFrame(
        [
            {
                "episode_index": index,
                "tasks": ["pick"],
                "length": 4,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": index * 4,
                "dataset_to_index": index * 4 + 4,
            }
            for index in range(2)
        ]
    ).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet", index=False)
    data = pd.DataFrame(
        {
            "action": [np.asarray([value], dtype=np.float32) for value in range(8)],
            "observation.state": [
                np.asarray([value], dtype=np.float32) for value in range(8)
            ],
            "timestamp": np.tile(np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float32), 2),
            "frame_index": [0, 1, 2, 3] * 2,
            "episode_index": [0] * 4 + [1] * 4,
            "index": list(range(8)),
            "task_index": [0] * 8,
        }
    )
    data.to_parquet(root / "data/chunk-000/file-000.parquet", index=False)
    stats = {
        name: _exact_stats(
            np.stack(
                [
                    np.asarray(value, dtype=np.float64).reshape(-1)
                    for value in data[name]
                ]
            )
        )
        for name in data.columns
    }
    (root / "meta/stats.json").write_text(json.dumps(stats), encoding="utf-8")


def _exact_stats(values: np.ndarray, *, count: int | None = None) -> dict:
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "count": [len(values) if count is None else count],
    }


def _add_shared_v3_video(root: Path) -> Path:
    import av

    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    video_key = "observation.images.top"
    info["features"][video_key] = {
        "dtype": "video",
        "shape": [24, 32, 3],
        "names": ["height", "width", "channels"],
    }
    info_path.write_text(json.dumps(info), encoding="utf-8")

    metadata_path = root / "meta/episodes/chunk-000/file-000.parquet"
    metadata = pd.read_parquet(metadata_path)
    metadata[f"videos/{video_key}/chunk_index"] = 0
    metadata[f"videos/{video_key}/file_index"] = 0
    metadata[f"videos/{video_key}/from_timestamp"] = [0.0, 0.4]
    metadata[f"videos/{video_key}/to_timestamp"] = [0.4, 0.8]
    metadata.to_parquet(metadata_path, index=False)

    video_path = root / f"videos/{video_key}/chunk-000/file-000.mp4"
    video_path.parent.mkdir(parents=True)
    container = av.open(str(video_path), mode="w")
    stream = container.add_stream("libx264", rate=10)
    stream.width = 32
    stream.height = 24
    stream.pix_fmt = "yuv420p"
    for value in range(8):
        frame = av.VideoFrame.from_ndarray(
            np.full((24, 32, 3), value * 10, dtype=np.uint8), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    decoded_container = av.open(str(video_path))
    decoded_pixels = np.concatenate(
        [
            frame.to_ndarray(format="rgb24").reshape(-1, 3)
            for frame in decoded_container.decode(video=0)
        ]
    ).astype(np.float64)
    decoded_container.close()
    stats_path = root / "meta/stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    visual_stats = _exact_stats(decoded_pixels / 255.0, count=8)
    stats[video_key] = {
        name: (
            value if name == "count" else np.asarray(value).reshape(3, 1, 1).tolist()
        )
        for name, value in visual_stats.items()
    }
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    return video_path


def test_v3_full_validation_decodes_a_shared_video_shard_once(
    tmp_path: Path, monkeypatch
) -> None:
    import av

    root = tmp_path / "dataset"
    _write_v3(root)
    video_path = _add_shared_v3_video(root)
    original_open = av.open
    video_opens = 0

    def tracked_open(file, *args, **kwargs):
        nonlocal video_opens
        if Path(file) == video_path:
            video_opens += 1
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(av, "open", tracked_open)
    result = validate_dataset_root(root, mode="full")

    assert result["passed"] is True
    assert video_opens == 1


def test_v3_validation_reads_a_shared_data_shard_once(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "dataset"
    _write_v3(root)
    original = dataset_source.read_parquet
    data_reads = 0

    def tracked(path: Path):
        nonlocal data_reads
        if path == root / "data/chunk-000/file-000.parquet":
            data_reads += 1
        return original(path)

    monkeypatch.setattr(dataset_source, "read_parquet", tracked)
    validate_dataset_root(root, mode="full")

    assert data_reads == 1


def test_v3_to_v21_conversion_rebuilds_layout_and_passes_export_gate(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    source = settings.nas_root / "raw/lab/v3"
    _write_v3(source)
    database = Database(settings.database_path)
    database.initialize()
    candidate = inspect_dataset(
        area_root=settings.nas_root / "raw", storage_area="raw", relative_path="lab/v3"
    )
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw", records=[candidate.as_record()], scan_generation=generation
    )
    dataset = database.list_datasets()[0]
    profile = database.create_profile("Converter")
    payload = {
        "dataset_id": dataset["id"],
        "fingerprint": dataset["fingerprint"],
        "storage_area": dataset["storage_area"],
        "relative_path": dataset["relative_path"],
        "output_name": "converted-v21",
    }
    job, _ = database.create_job(
        kind="datasets.convert_v21",
        queue_name="converter-v21",
        profile_id=profile["id"],
        payload=payload,
        idempotency_key="convert-1",
    )
    database.claim_job(job["id"], worker_id="converter", lease_seconds=120)

    result = convert_dataset_to_v21(
        database=database,
        settings=settings,
        payload=payload,
        job_id=job["id"],
        worker_id="converter",
    )

    output = settings.nas_root / "derived/converted-v21"
    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    assert info["codebase_version"] == "v2.1"
    assert not (output / "meta/tasks.parquet").exists()
    assert (output / "meta/tasks.jsonl").is_file()
    assert len(list((output / "data").rglob("episode_*.parquet"))) == 2
    assert result["validation"]["passed"] is True


def test_validation_reports_malformed_numeric_parquet_as_structured_failure(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    _write_v21(root)
    path = root / "data/chunk-000/episode_000000.parquet"
    frame = pd.read_parquet(path)
    frame["timestamp"] = ["bad"] * len(frame)
    frame.to_parquet(path, index=False)

    result = validate_dataset_root(root, mode="quick")

    assert result["passed"] is False
    assert any(item["code"] == "feature_dtype_mismatch" for item in result["issues"])


def test_v21_conversion_rejects_source_changed_during_job(
    tmp_path: Path, monkeypatch
) -> None:
    import pytest

    import datasetui.conversion as conversion
    from datasetui.database import RecipeRevisionMismatchError

    settings = _settings(tmp_path)
    source = settings.nas_root / "raw/lab/v3"
    _write_v3(source)
    database = Database(settings.database_path)
    database.initialize()
    candidate = inspect_dataset(
        area_root=settings.nas_root / "raw", storage_area="raw", relative_path="lab/v3"
    )
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw", records=[candidate.as_record()], scan_generation=generation
    )
    dataset = database.list_datasets()[0]
    profile = database.create_profile("Converter")
    payload = {
        "dataset_id": dataset["id"],
        "fingerprint": dataset["fingerprint"],
        "storage_area": dataset["storage_area"],
        "relative_path": dataset["relative_path"],
        "output_name": "converted-changed",
    }
    job, _ = database.create_job(
        kind="datasets.convert_v21",
        queue_name="converter-v21",
        profile_id=profile["id"],
        payload=payload,
        idempotency_key="convert-changed",
    )
    database.claim_job(job["id"], worker_id="converter", lease_seconds=120)
    original_inspect = conversion.inspect_dataset

    def inspect_then_mutate(**kwargs):
        # meta/info.json stays identical; only a data file is rewritten.
        data_file = next((source / "data").rglob("*.parquet"))
        data_file.write_bytes(data_file.read_bytes() + b"\0")
        return original_inspect(**kwargs)

    monkeypatch.setattr(conversion, "inspect_dataset", inspect_then_mutate)

    with pytest.raises(RecipeRevisionMismatchError):
        convert_dataset_to_v21(
            database=database,
            settings=settings,
            payload=payload,
            job_id=job["id"],
            worker_id="converter",
        )
    assert not (settings.nas_root / "derived/converted-changed").exists()
