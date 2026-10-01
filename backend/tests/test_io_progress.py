from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import datasetui.delivery as delivery
from datasetui.delivery import copy_to_pc_with_key, export_to_nas, upload_to_huggingface
from datasetui.huggingface import import_huggingface_dataset
from test_delivery import _pass_gate, _registered
from test_huggingface import FakeGateway, SHA, _settings


def _record_job_progress(database, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    events: list[dict] = []
    original = database.update_job_progress

    def record(job_id: str, *, worker_id: str, progress: dict) -> None:
        events.append(dict(progress))
        original(job_id, worker_id=worker_id, progress=progress)

    monkeypatch.setattr(database, "update_job_progress", record)
    return events


def test_huggingface_upload_reports_real_copy_and_opaque_upload_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, database, dataset, profile, _ = _registered(tmp_path)
    _pass_gate(database, dataset, profile)
    settings = replace(settings, hf_write_token="test-token")
    payload = {
        "dataset_id": dataset["id"],
        "fingerprint": dataset["fingerprint"],
        "storage_area": dataset["storage_area"],
        "relative_path": dataset["relative_path"],
        "repo_name": "progress-export",
        "visibility": "private",
    }
    job, _ = database.create_job(
        kind="datasets.upload_hf",
        queue_name="io",
        profile_id=profile["id"],
        payload=payload,
        idempotency_key="progress-upload",
    )
    database.claim_job(job["id"], worker_id="worker", lease_seconds=120)
    events = _record_job_progress(database, monkeypatch)

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def create_repo(self, **kwargs):
            return None

        def upload_folder(self, **kwargs):
            return SimpleNamespace(commit_url="https://example.invalid/commit")

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)

    upload_to_huggingface(
        database=database,
        settings=settings,
        payload=payload,
        job_id=job["id"],
        worker_id="worker",
    )

    stages = [event["stage"] for event in events]
    assert stages[0] == "preparing"
    assert {"copy", "verify", "register", "upload", "complete"} <= set(stages)
    copy_events = [event for event in events if event["stage"] == "copy"]
    assert copy_events[-1]["completed"] == copy_events[-1]["total"]
    upload = next(event for event in events if event["stage"] == "upload")
    assert upload["total"] == 0
    assert database.get_job(job["id"])["progress"]["stage"] == "complete"


def test_nas_export_reports_copy_verify_publish_and_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, database, dataset, profile, _ = _registered(tmp_path)
    _pass_gate(database, dataset, profile)
    payload = {
        "dataset_id": dataset["id"],
        "fingerprint": dataset["fingerprint"],
        "storage_area": dataset["storage_area"],
        "relative_path": dataset["relative_path"],
        "output_name": "progress-export",
    }
    job, _ = database.create_job(
        kind="datasets.export_nas",
        queue_name="io",
        profile_id=profile["id"],
        payload=payload,
        idempotency_key="progress-nas",
    )
    database.claim_job(job["id"], worker_id="worker", lease_seconds=120)
    events = _record_job_progress(database, monkeypatch)

    export_to_nas(
        database=database,
        settings=settings,
        payload=payload,
        job_id=job["id"],
        worker_id="worker",
    )

    stages = [event["stage"] for event in events]
    assert stages[0] == "preparing"
    assert {"copy", "verify", "publish", "complete"} <= set(stages)
    copy_events = [event for event in events if event["stage"] == "copy"]
    assert copy_events[-1]["completed"] == copy_events[-1]["total"]


def test_pc_key_delivery_passes_reporter_through_verified_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, database, dataset, profile, _ = _registered(tmp_path)
    _pass_gate(database, dataset, profile)
    payload = {
        "dataset_id": dataset["id"],
        "fingerprint": dataset["fingerprint"],
        "storage_area": dataset["storage_area"],
        "relative_path": dataset["relative_path"],
        "host": "192.0.2.10",
        "port": 22,
        "username": "researcher",
        "destination": "~/pick",
    }
    job, _ = database.create_job(
        kind="datasets.copy_pc_key",
        queue_name="io",
        profile_id=profile["id"],
        payload=payload,
        idempotency_key="progress-pc",
    )
    database.claim_job(job["id"], worker_id="worker", lease_seconds=120)
    events = _record_job_progress(database, monkeypatch)

    def fake_copy(*, on_progress, **kwargs):
        on_progress(
            {
                "stage": "connect",
                "completed": 0,
                "total": 0,
                "unit": "items",
                "current_item": "test host",
                "_force": True,
            }
        )
        on_progress(
            {
                "stage": "complete",
                "completed": 1,
                "total": 1,
                "unit": "items",
                "current_item": "done",
                "_force": True,
            }
        )
        return {"ok": True, "files": 1, "bytes": 1, "destination": "~/pick"}

    monkeypatch.setattr(delivery, "_copy_verified_to_pc", fake_copy)
    copy_to_pc_with_key(
        database=database,
        settings=settings,
        payload=payload,
        job_id=job["id"],
        worker_id="worker",
    )

    assert [event["stage"] for event in events] == [
        "preparing",
        "connect",
        "complete",
    ]


def test_huggingface_import_reports_download_copy_register_and_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    from datasetui.database import Database

    database = Database(settings.database_path)
    database.initialize()
    profile = database.create_profile("Researcher")
    job, _ = database.create_hf_import_job(
        profile_id=profile["id"],
        repo_id="rainbowrobotics/pick-cup",
        dataset_name="pick-cup",
        requested_revision="main",
        commit_sha=SHA,
        expected_file_count=2,
        expected_total_bytes=512,
        idempotency_key="progress-import",
    )
    database.claim_job(job["id"], worker_id="worker", lease_seconds=120)
    events = _record_job_progress(database, monkeypatch)

    import_huggingface_dataset(
        database=database,
        settings=settings,
        payload=database.get_job(job["id"])["payload"],
        job_id=job["id"],
        worker_id="worker",
        gateway=FakeGateway(),
    )

    stages = [event["stage"] for event in events]
    assert stages[0] == "preparing"
    assert {"download", "verify", "copy", "register", "complete"} <= set(stages)
    download = next(event for event in events if event["stage"] == "download")
    assert download["total"] == 0
    copy_events = [event for event in events if event["stage"] == "copy"]
    assert copy_events[-1]["completed"] == copy_events[-1]["total"]


def test_sftp_progress_counts_bytes_after_each_successful_put(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "first.bin").write_bytes(b"123")
    (source / "second.bin").write_bytes(b"45678")
    events: list[dict] = []

    class Sftp:
        def mkdir(self, path: str) -> None:
            return None

        def put(self, source_path: str, target_path: str) -> None:
            return None

    files, total = delivery._sftp_tree(
        Sftp(), source, "/incoming", on_progress=events.append, total_bytes=8
    )

    assert (files, total) == (2, 8)
    assert [event["completed"] for event in events] == [3, 8]
    assert all(event["total"] == 8 for event in events)


def test_copy_progress_callback_failure_propagates(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.txt").write_text("one", encoding="utf-8")

    def fail(_event: dict) -> None:
        raise RuntimeError("lease lost")

    with pytest.raises(RuntimeError, match="lease lost"):
        delivery._copytree_with_progress(
            source,
            tmp_path / "destination",
            total_files=1,
            on_progress=fail,
        )


def test_copy_progress_oserror_stops_before_later_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "first.txt").write_text("first", encoding="utf-8")
    (source / "second.txt").write_text("second", encoding="utf-8")
    destination = tmp_path / "destination"
    copied: list[str] = []
    original_copy = delivery.shutil.copy2

    def record_copy(source_path: str, destination_path: str) -> str:
        copied.append(Path(source_path).name)
        return original_copy(source_path, destination_path)

    monkeypatch.setattr(delivery.shutil, "copy2", record_copy)

    def fail_after_first(event: dict) -> None:
        if event["completed"] == 1:
            raise OSError("progress storage unavailable")

    with pytest.raises(OSError, match="progress storage unavailable"):
        delivery._copytree_with_progress(
            source,
            destination,
            total_files=2,
            on_progress=fail_after_first,
        )

    assert len(copied) == 1
