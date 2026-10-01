import pytest
from datasetui.database import JobLeaseLostError
from test_curation_api import _register


def test_progress_persists_through_polling_and_rejects_stale_worker(client, database):
    dataset = _register(database)
    p = database.create_profile("Progress")
    j, _ = database.create_job(
        kind="datasets.validate",
        queue_name="cpu",
        profile_id=p["id"],
        payload={},
        idempotency_key="progress",
    )
    database.record_validation_run(
        job_id=j["id"],
        dataset_id=dataset["id"],
        dataset_fingerprint=dataset["fingerprint"],
        mode="full",
    )
    assert database.list_validation_runs(dataset["id"])[0]["progress"] is None
    database.claim_job(j["id"], worker_id="owner", lease_seconds=120)
    progress = {
        "stage": "video",
        "completed": 1,
        "total": 5,
        "decoded_frames": 30,
        "elapsed_seconds": 2,
        "checks": [],
    }
    with pytest.raises(JobLeaseLostError):
        database.update_validation_progress(
            j["id"], worker_id="other", progress=progress
        )
    database.update_validation_progress(j["id"], worker_id="owner", progress=progress)
    result = client.get(f"/api/v1/datasets/{dataset['id']}/validations")
    assert result.status_code == 200
    assert result.json()[0]["progress"] == progress
    assert result.json()[0]["started_at"]
    database.fail_job(j["id"], "job_timeout", "timeout", worker_id="owner")
    with pytest.raises(JobLeaseLostError):
        database.update_validation_progress(j["id"], worker_id="owner", progress={})
    assert database.list_validation_runs(dataset["id"])[0]["progress"] == progress


def test_owned_worker_persists_real_validation_report(tmp_path, monkeypatch):
    from datasetui.config import Settings
    from datasetui.database import Database
    from datasetui.datasets import inspect_dataset
    from datasetui.tasks import run_job
    from test_transforms import _settings, _write_v21

    settings = _settings(tmp_path)
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: settings))
    root = settings.nas_root / "raw" / "sample"
    _write_v21(root)
    db = Database(settings.database_path)
    db.initialize()
    candidate = inspect_dataset(
        area_root=settings.nas_root / "raw", storage_area="raw", relative_path="sample"
    )
    db.synchronize_datasets(
        storage_area="raw",
        records=[candidate.as_record()],
        scan_generation=db.begin_dataset_scan("raw"),
    )
    dataset = db.list_datasets()[0]
    profile = db.create_profile("Progress integration")
    payload = {
        key: dataset[key] for key in ("fingerprint", "storage_area", "relative_path")
    }
    payload.update(dataset_id=dataset["id"], mode="quick")
    job, _ = db.create_job(
        kind="datasets.validate",
        queue_name="cpu",
        profile_id=profile["id"],
        payload=payload,
        idempotency_key="real-progress",
    )
    db.record_validation_run(
        job_id=job["id"],
        dataset_id=dataset["id"],
        dataset_fingerprint=dataset["fingerprint"],
        mode="quick",
    )
    assert run_job(job["id"])["status"] == "succeeded"
    run = db.list_validation_runs(dataset["id"])[0]
    assert run["progress"]["stage"] == "complete"
    assert run["progress"]["completed"] == run["progress"]["total"] == 2
    assert run["result"]["checks"] == run["progress"]["checks"]
