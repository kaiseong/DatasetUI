from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from datasetui.database import Database
from datasetui.datasets import inspect_dataset
from datasetui.merge import MergeCompatibilityError, merge_datasets
from test_transforms import _settings, _write_v21


def _register(database: Database, settings, relative: str) -> dict:
    candidate = inspect_dataset(
        area_root=settings.nas_root / "raw",
        storage_area="raw",
        relative_path=relative,
    )
    generation = database.begin_dataset_scan("raw")
    existing = [item.as_record() for item in []]
    database.synchronize_datasets(
        storage_area="raw",
        records=existing + [candidate.as_record()],
        scan_generation=generation,
    )
    return next(item for item in database.list_datasets() if item["relative_path"] == relative)


def test_merge_reindexes_episodes_and_preserves_source_lineage(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_v21(settings.nas_root / "raw/lab/first")
    _write_v21(settings.nas_root / "raw/lab/second")
    database = Database(settings.database_path)
    database.initialize()
    candidates = [
        inspect_dataset(
            area_root=settings.nas_root / "raw", storage_area="raw", relative_path=path
        ).as_record()
        for path in ("lab/first", "lab/second")
    ]
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw", records=candidates, scan_generation=generation
    )
    datasets = database.list_datasets()
    profile = database.create_profile("Merge operator")
    payload = {
        "sources": [
            {"id": item["id"], "fingerprint": item["fingerprint"]}
            for item in datasets
        ],
        "output_name": "merged-pick",
        "robot_type": "rby1-combined",
    }
    job, _ = database.create_job(
        kind="datasets.merge",
        queue_name="cpu",
        profile_id=profile["id"],
        payload=payload,
        idempotency_key="merge-1",
    )
    database.claim_job(job["id"], worker_id="merge-worker", lease_seconds=120)

    result = merge_datasets(
        database=database,
        settings=settings,
        payload=payload,
        job_id=job["id"],
        worker_id="merge-worker",
    )

    output = settings.nas_root / "derived/merged-pick"
    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    data = pd.concat(
        [pd.read_parquet(path) for path in sorted((output / "data").rglob("*.parquet"))]
    )
    assert info["total_episodes"] == 4
    assert info["total_frames"] == 40
    assert info["robot_type"] == "rby1-combined"
    assert sorted(data["episode_index"].unique().tolist()) == [0, 1, 2, 3]
    assert data["index"].tolist() == list(range(40))
    assert [item["source_dataset_id"] for item in result["output"]["lineage"]] == [
        datasets[0]["id"],
        datasets[0]["id"],
        datasets[1]["id"],
        datasets[1]["id"],
    ]


def test_merge_rejects_feature_schema_difference(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    first = settings.nas_root / "raw/lab/first"
    second = settings.nas_root / "raw/lab/second"
    _write_v21(first)
    _write_v21(second)
    info_path = second / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["action"]["names"] = ["other_joint"]
    info_path.write_text(json.dumps(info), encoding="utf-8")
    database = Database(settings.database_path)
    database.initialize()
    candidates = [
        inspect_dataset(
            area_root=settings.nas_root / "raw", storage_area="raw", relative_path=path
        ).as_record()
        for path in ("lab/first", "lab/second")
    ]
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw", records=candidates, scan_generation=generation
    )
    datasets = database.list_datasets()
    profile = database.create_profile("Merge guard")
    payload = {
        "sources": [
            {"id": item["id"], "fingerprint": item["fingerprint"]}
            for item in datasets
        ],
        "output_name": "bad-merge",
        "robot_type": "rby1",
    }
    job, _ = database.create_job(
        kind="datasets.merge",
        queue_name="cpu",
        profile_id=profile["id"],
        payload=payload,
        idempotency_key="merge-bad",
    )
    database.claim_job(job["id"], worker_id="merge-worker", lease_seconds=120)
    with pytest.raises(MergeCompatibilityError, match="identical feature"):
        merge_datasets(
            database=database,
            settings=settings,
            payload=payload,
            job_id=job["id"],
            worker_id="merge-worker",
        )
    assert not (settings.nas_root / "derived/bad-merge").exists()
