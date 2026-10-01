from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from datasetui.api import create_router
from datasetui.config import Settings
from datasetui.database import Database
from datasetui.datasets import scan_storage_area
from datasetui.queueing import RecordingDispatcher


def _write_dataset(root: Path, relative_path: str, *, ready: bool = True) -> Path:
    dataset = root / "raw" / relative_path
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "robot_type": "rby1",
                "total_episodes": 1,
                "total_frames": 3,
                "total_tasks": 1,
                "fps": 30,
                "features": {"action": {"dtype": "float32", "shape": [2]}},
            }
        ),
        encoding="utf-8",
    )
    if ready:
        (dataset / "data" / "chunk-000").mkdir(parents=True)
        (dataset / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"0123456789")
    return dataset


def _client_for_nas(
    *, database: Database, dispatcher: RecordingDispatcher, nas_root: Path
) -> TestClient:
    settings = replace(Settings.from_env(), nas_root=nas_root)
    app = FastAPI()
    app.include_router(create_router(database, dispatcher, settings=settings))
    return TestClient(app)


def _register(database: Database, nas_root: Path) -> dict:
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw",
        records=scan_storage_area(nas_root, "raw", max_depth=6),
        scan_generation=generation,
    )
    return database.list_datasets(include_missing=True)[0]


def test_registered_dataset_file_supports_get_head_and_ranges(
    database: Database, dispatcher: RecordingDispatcher, tmp_path: Path
) -> None:
    nas_root = tmp_path / "nas"
    (nas_root / "raw").mkdir(parents=True)
    _write_dataset(nas_root, "lab/pick-cup")
    dataset = _register(database, nas_root)
    client = _client_for_nas(
        database=database, dispatcher=dispatcher, nas_root=nas_root
    )
    url = f"/api/v1/datasets/{dataset['id']}/files/data/chunk-000/file-000.parquet"

    complete = client.get(url)
    assert complete.status_code == 200
    assert complete.content == b"0123456789"
    assert complete.headers["accept-ranges"] == "bytes"
    assert complete.headers["content-length"] == "10"
    assert complete.headers["content-type"].startswith("application/")
    assert complete.headers["cache-control"] == "private, no-store"

    partial = client.get(url, headers={"Range": "bytes=2-5"})
    assert partial.status_code == 206
    assert partial.content == b"2345"
    assert partial.headers["content-range"] == "bytes 2-5/10"
    assert partial.headers["content-length"] == "4"

    open_ended = client.get(url, headers={"Range": "bytes=7-"})
    assert open_ended.status_code == 206
    assert open_ended.content == b"789"

    suffix = client.get(url, headers={"Range": "bytes=-3"})
    assert suffix.status_code == 206
    assert suffix.content == b"789"

    head = client.head(url, headers={"Range": "bytes=0-1"})
    assert head.status_code == 206
    assert head.content == b""
    assert head.headers["content-range"] == "bytes 0-1/10"
    assert head.headers["content-length"] == "2"


def test_dataset_file_rejects_invalid_ranges_and_path_escape(
    database: Database, dispatcher: RecordingDispatcher, tmp_path: Path
) -> None:
    nas_root = tmp_path / "nas"
    (nas_root / "raw").mkdir(parents=True)
    _write_dataset(nas_root, "lab/pick-cup")
    dataset = _register(database, nas_root)
    client = _client_for_nas(
        database=database, dispatcher=dispatcher, nas_root=nas_root
    )
    base = f"/api/v1/datasets/{dataset['id']}/files"
    file_url = f"{base}/data/chunk-000/file-000.parquet"

    for range_value in ("items=0-1", "bytes=20-30", "bytes=0-1,4-5", "bytes=-0"):
        response = client.get(file_url, headers={"Range": range_value})
        assert response.status_code == 416
        assert response.headers["content-range"] == "bytes */10"

    traversal = client.get(f"{base}/%2e%2e/meta/info.json")
    assert traversal.status_code in {400, 404}
    backslash = client.get(f"{base}/data%5C..%5Cmeta%5Cinfo.json")
    assert backslash.status_code == 400
    assert str(nas_root) not in traversal.text + backslash.text


def test_dataset_file_does_not_follow_file_or_directory_symlinks(
    database: Database, dispatcher: RecordingDispatcher, tmp_path: Path
) -> None:
    nas_root = tmp_path / "nas"
    (nas_root / "raw").mkdir(parents=True)
    dataset_root = _write_dataset(nas_root, "lab/pick-cup")
    dataset = _register(database, nas_root)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"private")
    (dataset_root / "linked.bin").symlink_to(outside)
    (dataset_root / "linked-dir").symlink_to(tmp_path, target_is_directory=True)
    client = _client_for_nas(
        database=database, dispatcher=dispatcher, nas_root=nas_root
    )
    base = f"/api/v1/datasets/{dataset['id']}/files"

    assert client.get(f"{base}/linked.bin").status_code == 404
    assert client.get(f"{base}/linked-dir/outside.bin").status_code == 404


def test_dataset_file_requires_available_ready_registry_entry(
    database: Database, dispatcher: RecordingDispatcher, tmp_path: Path
) -> None:
    nas_root = tmp_path / "nas"
    (nas_root / "raw").mkdir(parents=True)
    incomplete_root = _write_dataset(nas_root, "lab/incomplete", ready=False)
    incomplete = _register(database, nas_root)
    client = _client_for_nas(
        database=database, dispatcher=dispatcher, nas_root=nas_root
    )

    not_ready = client.get(f"/api/v1/datasets/{incomplete['id']}/files/meta/info.json")
    assert not_ready.status_code == 409
    assert str(nas_root) not in not_ready.text

    for child in (incomplete_root / "meta").iterdir():
        child.unlink()
    (incomplete_root / "meta").rmdir()
    incomplete_root.rmdir()
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw", records=[], scan_generation=generation
    )
    missing = client.get(f"/api/v1/datasets/{incomplete['id']}/files/meta/info.json")
    assert missing.status_code == 404
    assert (
        client.get("/api/v1/datasets/not-a-real-id/files/meta/info.json").status_code
        == 404
    )
