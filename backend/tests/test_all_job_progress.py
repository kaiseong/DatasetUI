from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from datasetui.api import create_router
from datasetui.config import Settings
from datasetui.database import Database, JobLeaseLostError
from datasetui.jobs import run_registered_job
from datasetui.queueing import RecordingDispatcher
from datasetui.tasks import run_job
from test_curation_api import _register
from test_transforms import _settings


BASE_ASYNC_KINDS = (
    "phase2.smoke",
    "datasets.scan",
    "hf.import",
    "datasets.validate",
    "curation.materialize",
    "datasets.merge",
    "datasets.convert_v21",
    "datasets.export_nas",
    "datasets.upload_hf",
    "datasets.copy_pc_key",
)


def test_every_async_kind_records_running_and_completion_progress(
    tmp_path: Path, monkeypatch
) -> None:
    database_path = tmp_path / "datasetui.sqlite3"
    monkeypatch.setenv("DATASETUI_DB_PATH", str(database_path))
    database = Database(database_path)
    database.initialize()
    profile = database.create_profile("All progress")
    kinds = list(BASE_ASYNC_KINDS)
    if importlib.util.find_spec("datasetui.segmentation") is not None:
        kinds.extend(("segmentation.preview", "segmentation.export"))

    calls: dict[str, list[dict]] = {}
    original_update = Database.update_job_progress

    def capture_progress(self, job_id, *, worker_id, progress):
        calls.setdefault(job_id, []).append(progress)
        return original_update(self, job_id, worker_id=worker_id, progress=progress)

    def fake_registered_job(kind, payload, **kwargs):
        return {"kind": kind, "ok": True}

    monkeypatch.setattr(Database, "update_job_progress", capture_progress)
    monkeypatch.setattr("datasetui.tasks.run_registered_job", fake_registered_job)

    for index, kind in enumerate(kinds):
        job, _ = database.create_job(
            kind=kind,
            queue_name="cpu",
            profile_id=profile["id"],
            payload={},
            idempotency_key=f"all-progress-{index}",
        )
        assert run_job(job["id"])["status"] == "succeeded"
        stored = database.get_job(job["id"])
        assert stored["progress"]["stage"] == "complete"
        assert calls[job["id"]][0]["stage"] == "preparing"
        assert calls[job["id"]][-1]["stage"] == "complete"


def test_final_progress_write_failure_is_recorded_as_job_failure(
    tmp_path: Path, monkeypatch
) -> None:
    database_path = tmp_path / "datasetui.sqlite3"
    monkeypatch.setenv("DATASETUI_DB_PATH", str(database_path))
    database = Database(database_path)
    database.initialize()
    profile = database.create_profile("Progress failure")
    job, _ = database.create_job(
        kind="phase2.smoke",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={},
        idempotency_key="progress-write-failure",
    )
    original_update = Database.update_job_progress

    def fail_final_progress(self, job_id, *, worker_id, progress):
        if progress["stage"] == "complete":
            raise RuntimeError("progress storage unavailable")
        return original_update(self, job_id, worker_id=worker_id, progress=progress)

    monkeypatch.setattr(Database, "update_job_progress", fail_final_progress)

    with pytest.raises(RuntimeError, match="progress storage unavailable"):
        run_job(job["id"])

    stored = database.get_job(job["id"])
    assert stored["status"] == "failed"
    assert stored["error_code"] == "job_failed"


def test_validation_progress_updates_validation_and_generic_job_snapshots(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: settings))
    database = Database(settings.database_path)
    database.initialize()
    dataset = _register(database)
    profile = database.create_profile("Validation bridge")
    payload = {
        "dataset_id": dataset["id"],
        "fingerprint": "a" * 64,
        "storage_area": dataset["storage_area"],
        "relative_path": dataset["relative_path"],
        "mode": "full",
    }
    job, _ = database.create_job(
        kind="datasets.validate",
        queue_name="cpu",
        profile_id=profile["id"],
        payload=payload,
        idempotency_key="validation-bridge",
    )
    database.record_validation_run(
        job_id=job["id"],
        dataset_id=dataset["id"],
        dataset_fingerprint=dataset["fingerprint"],
        mode="full",
    )
    database.claim_job(job["id"], worker_id="validation-owner")
    snapshot = {
        "stage": "video",
        "completed": 2,
        "total": 5,
        "decoded_frames": 20,
        "elapsed_seconds": 1.0,
        "checks": [],
    }

    def fake_validation(**kwargs):
        kwargs["on_progress"](snapshot)
        return {"checks": []}

    monkeypatch.setattr(
        "datasetui.validation.validate_registered_dataset", fake_validation
    )
    run_registered_job(
        "datasets.validate",
        payload,
        job_id=job["id"],
        worker_id="validation-owner",
    )

    assert database.list_validation_runs(dataset["id"])[0]["progress"] == snapshot
    generic = database.get_job(job["id"])["progress"]
    assert generic["stage"] == "video"
    assert generic["completed"] == 2
    assert generic["total"] == 5
    assert generic["unit"] == "episodes"

    with pytest.raises(JobLeaseLostError):
        run_registered_job(
            "datasets.validate",
            payload,
            job_id=job["id"],
            worker_id="stale-worker",
        )


def test_scan_progress_reports_real_area_and_dataset_counts(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: settings))
    root = settings.nas_root / "raw" / "lab" / "sample"
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "features": {}}), encoding="utf-8"
    )
    database = Database(settings.database_path)
    database.initialize()
    profile = database.create_profile("Scan progress")
    job, _ = database.create_job(
        kind="datasets.scan",
        queue_name="io",
        profile_id=profile["id"],
        payload={"storage_areas": ["raw"]},
        idempotency_key="scan-progress",
    )
    database.claim_job(job["id"], worker_id="scan-owner")

    run_registered_job(
        "datasets.scan",
        {"storage_areas": ["raw"]},
        job_id=job["id"],
        worker_id="scan-owner",
    )

    progress = database.get_job(job["id"])["progress"]
    assert progress["completed"] == progress["total"] == 1
    assert progress["stage"] == "scan"
    assert progress["current_item"] == "raw: 데이터셋 1개 발견"


@pytest.mark.parametrize(
    ("write_token", "configured_flag", "expected"),
    [(None, False, False), ("super-secret-token", False, True), (None, True, True)],
)
def test_delivery_capabilities_reports_configuration_without_exposing_token(
    tmp_path: Path,
    database: Database,
    write_token: str | None,
    configured_flag: bool,
    expected: bool,
) -> None:
    settings = replace(
        _settings(tmp_path),
        database_path=database.path,
        hf_write_token=write_token,
        hf_upload_configured=configured_flag,
    )
    app = FastAPI()
    app.include_router(
        create_router(database, RecordingDispatcher(), settings=settings)
    )

    response = TestClient(app).get("/api/v1/delivery/capabilities")

    assert response.status_code == 200
    assert response.json() == {
        "hf_upload_configured": expected,
        "hf_delete_configured": expected,
        "pc_password_configured": False,
        "hf_namespace": "rainbowrobotics",
    }
    assert "super-secret-token" not in response.text


def test_hf_upload_configured_flag_does_not_require_api_token(monkeypatch) -> None:
    monkeypatch.delenv("HF_WRITE_TOKEN", raising=False)
    monkeypatch.setenv("DATASETUI_HF_UPLOAD_CONFIGURED", "true")

    settings = Settings.from_env()

    assert settings.hf_upload_configured is True
    assert settings.hf_write_token is None
