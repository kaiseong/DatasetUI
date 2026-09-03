from __future__ import annotations

from fastapi.testclient import TestClient

from datasetui.database import Database
from datasetui.queueing import RecordingDispatcher


def _record(path: str, fingerprint: str) -> dict:
    return {
        "storage_area": "raw",
        "relative_path": path,
        "name": path.rsplit("/", 1)[-1],
        "codebase_version": "v3.0",
        "readiness": "ready",
        "robot_type": "rby1",
        "total_episodes": 2,
        "total_frames": 20,
        "total_tasks": 1,
        "fps": 30,
        "fingerprint": fingerprint,
        "info_mtime_ns": 1,
        "info_size": 100,
        "scan_error": None,
    }


def test_merge_endpoint_snapshots_server_side_fingerprints_and_is_idempotent(
    client: TestClient, database: Database, dispatcher: RecordingDispatcher
) -> None:
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw",
        records=[_record("lab/a", "a" * 64), _record("lab/b", "b" * 64)],
        scan_generation=generation,
    )
    datasets = database.list_datasets()
    profile = client.post("/api/v1/profiles", json={"name": "Merger"}).json()
    request = {
        "profile_id": profile["id"],
        "dataset_ids": [item["id"] for item in datasets],
        "output_name": "combined",
        "robot_type": "rby1-combined",
        "idempotency_key": "merge-once",
    }

    created = client.post("/api/v1/datasets/merge", json=request)
    repeated = client.post("/api/v1/datasets/merge", json=request)

    assert created.status_code == 202
    assert repeated.status_code == 200
    assert repeated.json()["id"] == created.json()["id"]
    assert created.json()["payload"]["sources"] == [
        {"id": item["id"], "fingerprint": item["fingerprint"]}
        for item in datasets
    ]
    assert len(dispatcher.enqueued) == 1


def test_merge_endpoint_rejects_duplicates_and_secret_fields(
    client: TestClient, database: Database
) -> None:
    profile = client.post("/api/v1/profiles", json={"name": "Strict merger"}).json()
    body = {
        "profile_id": profile["id"],
        "dataset_ids": ["same", "same"],
        "output_name": "combined",
        "robot_type": "rby1",
        "idempotency_key": "strict",
        "password": "never-store",
    }
    response = client.post("/api/v1/datasets/merge", json=body)
    assert response.status_code == 422
