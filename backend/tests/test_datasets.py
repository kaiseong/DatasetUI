from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
from pathlib import Path

from fastapi.testclient import TestClient

from datasetui.database import Database
from datasetui.datasets import (
    MAX_INFO_BYTES,
    DatasetRootUnavailableError,
    _read_info_safely,
    scan_storage_area,
)


def _write_dataset(
    root: Path,
    relative_path: str,
    *,
    version: str = "v3.0",
    with_data: bool = True,
) -> Path:
    dataset_root = root / relative_path
    (dataset_root / "meta").mkdir(parents=True, exist_ok=True)
    (dataset_root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": version,
                "robot_type": "rby1",
                "total_episodes": 12,
                "total_frames": 360,
                "total_tasks": 2,
                "fps": 30,
                "features": {"action": {"dtype": "float32", "shape": [6]}},
            }
        ),
        encoding="utf-8",
    )
    if with_data:
        (dataset_root / "data").mkdir(exist_ok=True)
    return dataset_root


def test_scanner_classifies_supported_incomplete_and_unsupported_datasets(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_dataset(raw, "team/ready")
    _write_dataset(raw, "team/incomplete", with_data=False)
    _write_dataset(raw, "legacy", version="v1.6")

    records = scan_storage_area(tmp_path, "raw", max_depth=6)
    by_name = {record["name"]: record for record in records}

    assert by_name["ready"]["readiness"] == "ready"
    assert by_name["ready"]["total_frames"] == 360
    assert by_name["incomplete"]["readiness"] == "incomplete"
    assert by_name["legacy"]["readiness"] == "unsupported"
    assert all("absolute_path" not in record for record in records)


def test_scanner_reports_bad_metadata_without_following_symlinks(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    broken_meta = raw / "broken" / "meta"
    broken_meta.mkdir(parents=True)
    (broken_meta / "info.json").write_text("not json", encoding="utf-8")

    outside = tmp_path / "outside"
    _write_dataset(outside, "secret")
    (raw / "linked").symlink_to(outside / "secret", target_is_directory=True)

    records = scan_storage_area(tmp_path, "raw", max_depth=6)
    assert len(records) == 1
    assert records[0]["name"] == "broken"
    assert records[0]["readiness"] == "invalid"
    assert records[0]["scan_error"].startswith("Invalid meta/info.json")


def test_scanner_hides_incoming_imports_and_names_managed_revisions(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    base = "hf/rainbowrobotics/pick-cup/revisions"
    _write_dataset(raw, f"{base}/{'a' * 40}")
    _write_dataset(raw, f"{base}/.incoming-{'b' * 40}-staging")

    records = scan_storage_area(tmp_path, "raw", max_depth=8)
    assert len(records) == 1
    assert records[0]["name"] == "pick-cup"
    assert records[0]["relative_path"].endswith("a" * 40)


def test_scanner_rejects_symlinked_storage_area(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    nas_root = tmp_path / "nas"
    nas_root.mkdir()
    (nas_root / "raw").symlink_to(outside, target_is_directory=True)

    try:
        scan_storage_area(nas_root, "raw", max_depth=6)
    except DatasetRootUnavailableError as exc:
        assert str(exc) == "Dataset storage area is unavailable: raw"
    else:
        raise AssertionError("a symlinked storage area must not be scanned")


def test_info_symlink_error_does_not_leak_its_absolute_target(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    meta = raw / "linked-info" / "meta"
    meta.mkdir(parents=True)
    (raw / "linked-info" / "data").mkdir()
    outside = tmp_path / "private-info.json"
    outside.write_text('{"codebase_version":"v3.0","features":{}}', encoding="utf-8")
    (meta / "info.json").symlink_to(outside)

    records = scan_storage_area(tmp_path, "raw", max_depth=6)
    assert len(records) == 1
    assert records[0]["readiness"] == "invalid"
    assert records[0]["scan_error"] == "Unable to read meta/info.json safely"
    assert str(outside) not in records[0]["scan_error"]


def test_oversized_info_file_is_rejected(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    meta = raw / "oversized" / "meta"
    meta.mkdir(parents=True)
    (raw / "oversized" / "data").mkdir()
    (meta / "info.json").write_bytes(b"x" * (MAX_INFO_BYTES + 1))

    records = scan_storage_area(tmp_path, "raw", max_depth=6)
    assert len(records) == 1
    assert records[0]["readiness"] == "invalid"
    assert records[0]["scan_error"] == "Unable to read meta/info.json safely"


def test_growing_info_read_is_bounded_to_limit_plus_one(
    tmp_path: Path, monkeypatch
) -> None:
    raw = tmp_path / "raw"
    meta = raw / "growing" / "meta"
    meta.mkdir(parents=True)
    info_path = meta / "info.json"
    info_path.write_bytes(b"x" * (MAX_INFO_BYTES + 4096))

    original_fstat = os.fstat
    original_read = os.read
    requested_bytes = 0

    def undersized_fstat(file_descriptor):
        result = original_fstat(file_descriptor)

        class StatView:
            st_mode = result.st_mode
            st_size = MAX_INFO_BYTES
            st_mtime_ns = result.st_mtime_ns

        return StatView()

    def tracked_read(file_descriptor, size):
        nonlocal requested_bytes
        requested_bytes += size
        return original_read(file_descriptor, size)

    monkeypatch.setattr(os, "fstat", undersized_fstat)
    monkeypatch.setattr(os, "read", tracked_read)
    try:
        _read_info_safely(raw, "growing")
    except OSError:
        pass
    else:
        raise AssertionError("metadata that grows past the limit must be rejected")
    assert requested_bytes == MAX_INFO_BYTES + 1


def test_scanner_fails_instead_of_returning_a_partial_walk(
    tmp_path: Path, monkeypatch
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()

    def failed_walk(*args, onerror, **kwargs):
        onerror(PermissionError("unreadable subtree"))
        return iter(())

    monkeypatch.setattr(os, "walk", failed_walk)
    try:
        scan_storage_area(tmp_path, "raw", max_depth=6)
    except DatasetRootUnavailableError as exc:
        assert str(exc) == "Dataset storage area could not be read completely: raw"
    else:
        raise AssertionError("a partial storage scan must fail")


def test_registry_preserves_identity_and_marks_disappeared_datasets(
    database: Database, tmp_path: Path
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    dataset_root = _write_dataset(raw, "stable")
    first_records = scan_storage_area(tmp_path, "raw", max_depth=6)
    first_generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw",
        records=first_records,
        scan_generation=first_generation,
    )
    first = database.list_datasets()[0]

    (dataset_root / "meta" / "info.json").unlink()
    missing_generation = database.begin_dataset_scan("raw")
    result = database.synchronize_datasets(
        storage_area="raw",
        records=[],
        scan_generation=missing_generation,
    )
    assert result == {"discovered": 0, "missing": 1}
    assert database.list_datasets() == []
    missing = database.list_datasets(include_missing=True)[0]
    assert missing["id"] == first["id"]
    assert missing["available"] is False

    _write_dataset(raw, "stable")
    rediscovered_generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw",
        records=scan_storage_area(tmp_path, "raw", max_depth=6),
        scan_generation=rediscovered_generation,
    )
    rediscovered = database.list_datasets()[0]
    assert rediscovered["id"] == first["id"]
    assert rediscovered["available"] is True


def test_dataset_library_api_uses_registry_ids_not_absolute_paths(
    client: TestClient, database: Database, tmp_path: Path
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_dataset(raw, "lab/pick-cup")
    scan_generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw",
        records=scan_storage_area(tmp_path, "raw", max_depth=6),
        scan_generation=scan_generation,
    )

    response = client.get("/api/v1/datasets?storage_area=raw&readiness=ready")
    assert response.status_code == 200
    datasets = response.json()
    assert len(datasets) == 1
    assert datasets[0]["relative_path"] == "lab/pick-cup"
    assert str(tmp_path) not in response.text

    detail = client.get(f"/api/v1/datasets/{datasets[0]['id']}")
    assert detail.status_code == 200
    assert detail.json() == datasets[0]
    assert client.get("/api/v1/datasets/missing").status_code == 404


def test_dataset_display_name_survives_rescan_without_changing_source_identity(
    client: TestClient, database: Database, tmp_path: Path
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_dataset(raw, "lab/pick-cup")
    generation = database.begin_dataset_scan("raw")
    records = scan_storage_area(tmp_path, "raw", max_depth=6)
    database.synchronize_datasets(
        storage_area="raw", records=records, scan_generation=generation
    )
    original = database.list_datasets()[0]

    response = client.patch(
        f"/api/v1/datasets/{original['id']}",
        json={"name": "  왼쪽 로봇 집기  ", "expected_name": original["name"]},
    )

    assert response.status_code == 200
    renamed = response.json()
    assert renamed["name"] == "왼쪽 로봇 집기"
    assert renamed["id"] == original["id"]
    assert renamed["relative_path"] == original["relative_path"]
    assert renamed["fingerprint"] == original["fingerprint"]
    assert database.list_datasets()[0]["name"] == "왼쪽 로봇 집기"

    next_generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw", records=records, scan_generation=next_generation
    )

    rescanned = client.get(f"/api/v1/datasets/{original['id']}").json()
    assert rescanned["name"] == "왼쪽 로봇 집기"
    assert rescanned["id"] == original["id"]
    assert rescanned["fingerprint"] == original["fingerprint"]
    with database.connect() as connection:
        stored = connection.execute(
            "SELECT name, display_name FROM datasets WHERE id = ?", (original["id"],)
        ).fetchone()
    assert stored["name"] == "pick-cup"
    assert stored["display_name"] == "왼쪽 로봇 집기"


def test_dataset_display_name_rejects_invalid_and_stale_updates(
    client: TestClient, database: Database, tmp_path: Path
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_dataset(raw, "source-name")
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw",
        records=scan_storage_area(tmp_path, "raw", max_depth=6),
        scan_generation=generation,
    )
    dataset = database.list_datasets()[0]
    url = f"/api/v1/datasets/{dataset['id']}"

    for invalid_name in ("   ", "x" * 121, "bad\nname", "bad\u0085name"):
        response = client.patch(
            url, json={"name": invalid_name, "expected_name": dataset["name"]}
        )
        assert response.status_code == 422

    first = client.patch(
        url, json={"name": "first edit", "expected_name": dataset["name"]}
    )
    stale = client.patch(
        url, json={"name": "stale edit", "expected_name": dataset["name"]}
    )

    assert first.status_code == 200
    assert stale.status_code == 409
    assert "다른 곳에서 변경" in stale.json()["detail"]
    assert client.get(url).json()["name"] == "first edit"


def test_dataset_display_name_update_requires_expected_name_and_existing_dataset(
    client: TestClient,
) -> None:
    missing_url = "/api/v1/datasets/00000000-0000-4000-8000-000000000000"
    assert client.patch(missing_url, json={"name": "new name"}).status_code == 422
    response = client.patch(
        missing_url,
        json={"name": "new name", "expected_name": "old name"},
    )
    assert response.status_code == 404


def test_dataset_display_name_allows_long_legacy_name_as_expected_value(
    client: TestClient, database: Database, tmp_path: Path
) -> None:
    legacy_name = "l" * 121
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_dataset(raw, legacy_name)
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw",
        records=scan_storage_area(tmp_path, "raw", max_depth=6),
        scan_generation=generation,
    )
    dataset = database.list_datasets()[0]

    response = client.patch(
        f"/api/v1/datasets/{dataset['id']}",
        json={"name": "readable name", "expected_name": legacy_name},
    )

    assert response.status_code == 200
    assert response.json()["name"] == "readable name"


def test_stale_scan_cannot_resurrect_a_dataset_removed_by_a_newer_scan(
    database: Database, tmp_path: Path
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_dataset(raw, "race")
    record = scan_storage_area(tmp_path, "raw", max_depth=6)

    initial_generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw",
        records=record,
        scan_generation=initial_generation,
    )
    stale_generation = database.begin_dataset_scan("raw")
    newer_generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw",
        records=[],
        scan_generation=newer_generation,
    )

    stale = database.synchronize_datasets(
        storage_area="raw",
        records=record,
        scan_generation=stale_generation,
    )
    assert stale == {"discovered": 0, "missing": 0, "stale": 1}
    assert database.list_datasets() == []
    assert database.list_datasets(include_missing=True)[0]["available"] is False


def test_scan_generation_check_and_registry_write_share_one_write_lock(
    database: Database, tmp_path: Path, monkeypatch
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_dataset(raw, "serialized")
    record = scan_storage_area(tmp_path, "raw", max_depth=6)
    generation = database.begin_dataset_scan("raw")

    generation_checked = threading.Event()
    release_stale_scan = threading.Event()
    original_connect = database.connect

    class ObservedConnection:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, sql, parameters=()):
            cursor = self.connection.execute(sql, parameters)
            if (
                threading.current_thread().name == "stale-sync"
                and "SELECT generation FROM dataset_scan_generations" in sql
            ):
                assert self.connection.in_transaction is True
                generation_checked.set()
                assert release_stale_scan.wait(timeout=2)
            return cursor

        def __getattr__(self, name):
            return getattr(self.connection, name)

    @contextmanager
    def observed_connect():
        with original_connect() as connection:
            yield ObservedConnection(connection)

    monkeypatch.setattr(database, "connect", observed_connect)

    def run_stale_sync():
        threading.current_thread().name = "stale-sync"
        return database.synchronize_datasets(
            storage_area="raw",
            records=record,
            scan_generation=generation,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        stale_future = executor.submit(run_stale_sync)
        if not generation_checked.wait(timeout=2):
            release_stale_scan.set()
            raise AssertionError("stale scan did not reach the generation check")
        newer_future = executor.submit(database.begin_dataset_scan, "raw")
        try:
            newer_future.result(timeout=0.1)
        except TimeoutError:
            pass
        else:
            raise AssertionError("new generation bypassed the registry write lock")
        finally:
            release_stale_scan.set()

        assert stale_future.result(timeout=2)["discovered"] == 1
        newer_generation = newer_future.result(timeout=2)

    database.synchronize_datasets(
        storage_area="raw",
        records=[],
        scan_generation=newer_generation,
    )
    assert database.list_datasets() == []
