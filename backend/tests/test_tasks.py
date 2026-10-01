from __future__ import annotations

from pathlib import Path
import json

import pytest

from datasetui.database import Database
from datasetui.delivery import (
    HuggingFaceCleanupRequiredError,
    HuggingFaceExternalOperationAmbiguousError,
)
from datasetui.tasks import _public_failure, run_job


def test_worker_claims_and_completes_job(tmp_path: Path, monkeypatch) -> None:
    database_path = tmp_path / "datasetui.sqlite3"
    monkeypatch.setenv("DATASETUI_DB_PATH", str(database_path))
    database = Database(database_path)
    database.initialize()
    profile = database.create_profile("Researcher")
    job, _ = database.create_job(
        kind="phase2.smoke",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={},
        idempotency_key="worker-completion",
    )

    result = run_job(job["id"])
    assert result == {"job_id": job["id"], "status": "succeeded", "claimed": True}
    stored = database.get_job(job["id"])
    assert stored["result"] == {"ok": True}
    assert [event["event_type"] for event in database.list_job_events(job["id"])] == [
        "queued",
        "running",
        "succeeded",
    ]


def test_worker_scans_nas_dataset_into_registry(tmp_path: Path, monkeypatch) -> None:
    database_path = tmp_path / "datasetui.sqlite3"
    nas_root = tmp_path / "nas"
    for area in ("raw", "derived"):
        (nas_root / area).mkdir(parents=True)
    dataset_root = nas_root / "raw" / "lab" / "demo"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "data").mkdir()
    (dataset_root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "features": {},
                "total_episodes": 1,
                "total_frames": 10,
                "total_tasks": 1,
                "fps": 10,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DATASETUI_DB_PATH", str(database_path))
    monkeypatch.setenv("DATASETUI_NAS_ROOT", str(nas_root))

    database = Database(database_path)
    database.initialize()
    profile = database.create_profile("Researcher")
    job, _ = database.create_job(
        kind="datasets.scan",
        queue_name="io",
        profile_id=profile["id"],
        payload={"storage_areas": ["raw"]},
        idempotency_key="worker-scan",
    )

    result = run_job(job["id"])
    assert result["status"] == "succeeded"
    stored = database.get_job(job["id"])
    assert stored["result"] == {
        "storage_areas": {"raw": {"discovered": 1, "missing": 0}}
    }
    datasets = database.list_datasets()
    assert [(item["storage_area"], item["relative_path"]) for item in datasets] == [
        ("raw", "lab/demo")
    ]


def test_worker_does_not_persist_internal_exception_details(
    tmp_path: Path, monkeypatch
) -> None:
    database_path = tmp_path / "datasetui.sqlite3"
    monkeypatch.setenv("DATASETUI_DB_PATH", str(database_path))
    database = Database(database_path)
    database.initialize()
    profile = database.create_profile("Researcher")
    job, _ = database.create_job(
        kind="phase2.smoke",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={},
        idempotency_key="worker-safe-error",
    )
    private_path = "/mnt/datasetui-nas/private/location"

    def fail_with_private_detail(kind: str, payload: dict[str, object]) -> dict:
        raise RuntimeError(f"unable to read {private_path}")

    monkeypatch.setattr("datasetui.tasks.run_registered_job", fail_with_private_detail)

    with pytest.raises(RuntimeError, match=private_path):
        run_job(job["id"])

    stored = database.get_job(job["id"])
    assert stored["status"] == "failed"
    assert stored["error_code"] == "job_failed"
    assert stored["error_message"] == "The job could not be completed"
    assert private_path not in stored["error_message"]


def test_worker_persists_safe_hf_orphan_cleanup_state(
    tmp_path: Path, monkeypatch
) -> None:
    database_path = tmp_path / "datasetui.sqlite3"
    monkeypatch.setenv("DATASETUI_DB_PATH", str(database_path))
    database = Database(database_path)
    database.initialize()
    profile = database.create_profile("Publisher")
    job, _ = database.create_job(
        kind="datasets.upload_hf",
        queue_name="io",
        profile_id=profile["id"],
        payload={
            "dataset_id": "11111111-1111-4111-8111-111111111111",
            "fingerprint": "a" * 64,
            "storage_area": "derived",
            "relative_path": "safe/output",
            "repo_name": "pick-export",
            "visibility": "private",
        },
        idempotency_key="orphan-cleanup",
    )

    def fail_external(*args, **kwargs):
        raise HuggingFaceCleanupRequiredError("rainbowrobotics/pick-export")

    monkeypatch.setattr("datasetui.tasks.run_registered_job", fail_external)
    with pytest.raises(HuggingFaceCleanupRequiredError):
        run_job(job["id"])

    stored = database.get_job(job["id"])
    assert stored["error_code"] == "hf_cleanup_required"
    assert stored["error_message"] == (
        "Manual cleanup is required for Hugging Face repository "
        "rainbowrobotics/pick-export"
    )


def test_worker_reports_hf_external_phase_ambiguity() -> None:
    failure = HuggingFaceExternalOperationAmbiguousError(
        "rainbowrobotics/pick-delayed", "upload"
    )
    assert _public_failure(failure) == (
        "external_outcome_uncertain",
        "Hugging Face upload outcome is uncertain for repository "
        "rainbowrobotics/pick-delayed",
    )
