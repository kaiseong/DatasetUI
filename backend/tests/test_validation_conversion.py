from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from datasetui.conversion import convert_dataset_to_v21
from datasetui.database import Database
from datasetui.datasets import inspect_dataset
from datasetui.validation import validate_dataset_root
from test_transforms import _settings, _write_v21


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
            "observation.state": {"dtype": "float32", "shape": [1], "names": ["joint_0"]},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/stats.json").write_text(json.dumps({"action": {"count": [8]}}), encoding="utf-8")
    pd.DataFrame([{"task_index": 0, "task": "pick"}]).to_parquet(root / "meta/tasks.parquet", index=False)
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
    pd.DataFrame(
        {
            "action": [[float(value)] for value in range(8)],
            "observation.state": [[float(value)] for value in range(8)],
            "timestamp": [0.0, 0.1, 0.2, 0.3] * 2,
            "frame_index": [0, 1, 2, 3] * 2,
            "episode_index": [0] * 4 + [1] * 4,
            "index": list(range(8)),
            "task_index": [0] * 8,
        }
    ).to_parquet(root / "data/chunk-000/file-000.parquet", index=False)


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
