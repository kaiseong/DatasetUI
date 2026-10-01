from __future__ import annotations

from pathlib import Path

from datasetui.conversion import convert_dataset_to_v21
from datasetui.database import Database, JobLeaseLostError
from datasetui.datasets import inspect_dataset
from datasetui.transforms import materialize_curation_recipe
from test_transforms import _settings, _write_v21
from test_validation_conversion import _write_v3


def _capture_progress(database: Database, monkeypatch) -> list[dict]:
    events: list[dict] = []
    original = database.update_job_progress

    def capture(job_id, *, worker_id, progress):
        events.append(progress)
        original(job_id, worker_id=worker_id, progress=progress)

    monkeypatch.setattr(database, "update_job_progress", capture)
    return events


def _register(database: Database, settings, relative_path: str) -> dict:
    candidate = inspect_dataset(
        area_root=settings.nas_root / "raw",
        storage_area="raw",
        relative_path=relative_path,
    )
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw",
        records=[candidate.as_record()],
        scan_generation=generation,
    )
    return database.list_datasets()[0]


def test_curation_materialization_persists_measured_progress(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path)
    _write_v21(settings.nas_root / "raw/lab/source")
    database = Database(settings.database_path)
    database.initialize()
    dataset = _register(database, settings, "lab/source")
    profile = database.create_profile("Processing progress")
    recipe = database.create_curation_recipe(
        dataset_id=dataset["id"],
        profile_id=profile["id"],
        name="Progress recipe",
        selection_mode="all",
    )
    snapshot = database.snapshot_curation_recipe(recipe["id"], profile_id=profile["id"])
    payload = {"snapshot_id": snapshot["id"], "output_name": "progress-output"}
    job, _ = database.create_job(
        kind="curation.materialize",
        queue_name="cpu",
        profile_id=profile["id"],
        payload=payload,
        idempotency_key="progress-curation",
    )
    database.claim_job(job["id"], worker_id="worker", lease_seconds=120)
    events = _capture_progress(database, monkeypatch)

    materialize_curation_recipe(
        database=database,
        settings=settings,
        payload=payload,
        job_id=job["id"],
        worker_id="worker",
    )

    stages = {event["stage"] for event in events}
    assert {
        "preparing",
        "read",
        "write",
        "statistics",
        "validate",
        "publish",
        "register",
        "complete",
    } <= stages
    assert any(
        event["stage"] == "read" and event["completed"] == event["total"] == 2
        for event in events
    )
    assert database.get_job(job["id"])["progress"]["stage"] == "complete"


def test_conversion_persists_validation_and_completion_progress(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path)
    _write_v3(settings.nas_root / "raw/lab/v3")
    database = Database(settings.database_path)
    database.initialize()
    dataset = _register(database, settings, "lab/v3")
    profile = database.create_profile("Conversion progress")
    payload = {
        "dataset_id": dataset["id"],
        "fingerprint": dataset["fingerprint"],
        "storage_area": dataset["storage_area"],
        "relative_path": dataset["relative_path"],
        "output_name": "converted-progress",
    }
    job, _ = database.create_job(
        kind="datasets.convert_v21",
        queue_name="converter-v21",
        profile_id=profile["id"],
        payload=payload,
        idempotency_key="progress-conversion",
    )
    database.claim_job(job["id"], worker_id="converter", lease_seconds=120)
    events = _capture_progress(database, monkeypatch)

    convert_dataset_to_v21(
        database=database,
        settings=settings,
        payload=payload,
        job_id=job["id"],
        worker_id="converter",
    )

    assert any(
        event["stage"] == "validate" and event["completed"] == event["total"] == 3
        for event in events
    )
    assert events[-1]["stage"] == "complete"
    assert events[-1]["output_name"] == "converted-progress"


def test_processing_progress_lease_failure_stops_before_output(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path)
    _write_v21(settings.nas_root / "raw/lab/source")
    database = Database(settings.database_path)
    database.initialize()
    dataset = _register(database, settings, "lab/source")
    profile = database.create_profile("Lost lease")
    recipe = database.create_curation_recipe(
        dataset_id=dataset["id"],
        profile_id=profile["id"],
        name="Lease recipe",
        selection_mode="all",
    )
    snapshot = database.snapshot_curation_recipe(recipe["id"], profile_id=profile["id"])
    payload = {"snapshot_id": snapshot["id"], "output_name": "must-not-publish"}
    job, _ = database.create_job(
        kind="curation.materialize",
        queue_name="cpu",
        profile_id=profile["id"],
        payload=payload,
        idempotency_key="lost-lease-progress",
    )
    database.claim_job(job["id"], worker_id="worker", lease_seconds=120)

    def lose_lease(*args, **kwargs):
        raise JobLeaseLostError(job["id"])

    monkeypatch.setattr(database, "update_job_progress", lose_lease)

    try:
        materialize_curation_recipe(
            database=database,
            settings=settings,
            payload=payload,
            job_id=job["id"],
            worker_id="worker",
        )
    except JobLeaseLostError:
        pass
    else:
        raise AssertionError("progress lease failure must propagate")
    assert not (settings.nas_root / "derived/must-not-publish").exists()
