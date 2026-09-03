from __future__ import annotations

from pathlib import Path

import pytest

from datasetui.database import Database
from datasetui.datasets import inspect_dataset
import datasetui.delivery as delivery_module
from datasetui.delivery import (
    ExportGateRequiredError,
    copy_to_pc_with_password,
    export_to_nas,
)
from datasetui.jobs import validate_job_payload
from test_transforms import _settings, _write_v21


def _registered(tmp_path: Path) -> tuple:
    settings = _settings(tmp_path)
    source = settings.nas_root / "raw/lab/pick"
    _write_v21(source)
    database = Database(settings.database_path)
    database.initialize()
    candidate = inspect_dataset(
        area_root=settings.nas_root / "raw", storage_area="raw", relative_path="lab/pick"
    )
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw", records=[candidate.as_record()], scan_generation=generation
    )
    dataset = database.list_datasets()[0]
    profile = database.create_profile("Exporter")
    return settings, database, dataset, profile, source


def _pass_gate(database: Database, dataset: dict, profile: dict) -> None:
    payload = {
        "dataset_id": dataset["id"],
        "fingerprint": dataset["fingerprint"],
        "storage_area": dataset["storage_area"],
        "relative_path": dataset["relative_path"],
        "mode": "export_gate",
    }
    job, _ = database.create_job(
        kind="datasets.validate",
        queue_name="cpu",
        profile_id=profile["id"],
        payload=payload,
        idempotency_key="gate",
    )
    database.record_validation_run(
        job_id=job["id"],
        dataset_id=dataset["id"],
        dataset_fingerprint=dataset["fingerprint"],
        mode="export_gate",
    )
    database.claim_job(job["id"], worker_id="validator", lease_seconds=120)
    database.succeed_job(
        job["id"],
        {"mode": "export_gate", "passed": True},
        worker_id="validator",
    )


def test_nas_delivery_requires_gate_and_publishes_verified_copy(tmp_path: Path) -> None:
    settings, database, dataset, profile, source = _registered(tmp_path)
    payload = {
        "dataset_id": dataset["id"],
        "fingerprint": dataset["fingerprint"],
        "storage_area": dataset["storage_area"],
        "relative_path": dataset["relative_path"],
        "output_name": "pick-export",
    }
    job, _ = database.create_job(
        kind="datasets.export_nas",
        queue_name="io",
        profile_id=profile["id"],
        payload=payload,
        idempotency_key="export",
    )
    database.claim_job(job["id"], worker_id="exporter", lease_seconds=120)
    with pytest.raises(ExportGateRequiredError):
        export_to_nas(
            database=database,
            settings=settings,
            payload=payload,
            job_id=job["id"],
            worker_id="exporter",
        )

    _pass_gate(database, dataset, profile)
    result = export_to_nas(
        database=database,
        settings=settings,
        payload=payload,
        job_id=job["id"],
        worker_id="exporter",
    )
    output = settings.nas_root / "exports/pick-export"
    assert result["manifest_sha256"]
    assert output.is_dir()
    assert (output / "meta/info.json").read_bytes() == (source / "meta/info.json").read_bytes()
    assert (settings.nas_root / "manifests/delivery" / f"{job['id']}.json").is_file()


def test_export_gate_is_bound_to_the_exact_fingerprint(tmp_path: Path) -> None:
    _, database, dataset, profile, _ = _registered(tmp_path)
    _pass_gate(database, dataset, profile)
    assert database.has_successful_export_gate(
        dataset_id=dataset["id"], dataset_fingerprint=dataset["fingerprint"]
    )
    assert not database.has_successful_export_gate(
        dataset_id=dataset["id"], dataset_fingerprint="f" * 64
    )


def test_one_time_pc_password_is_not_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, database, dataset, profile, _ = _registered(tmp_path)
    _pass_gate(database, dataset, profile)
    captured: dict = {}

    def fake_copy(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "files": 1, "bytes": 10, "destination": kwargs["destination"]}

    monkeypatch.setattr(delivery_module, "_copy_to_pc", fake_copy)
    result = copy_to_pc_with_password(
        database=database,
        settings=settings,
        dataset_id=dataset["id"],
        profile_id=profile["id"],
        host="192.168.0.51",
        port=22,
        username="researcher",
        password="one-time-secret",
        destination="~/pick",
    )
    assert result["ok"] is True
    assert captured["password"] == "one-time-secret"
    assert "one-time-secret" not in str(database.list_jobs(limit=100))


def test_delivery_job_payload_rejects_any_secret_field() -> None:
    payload = {
        "dataset_id": "11111111-1111-4111-8111-111111111111",
        "fingerprint": "a" * 64,
        "storage_area": "derived",
        "relative_path": "safe/output",
        "host": "192.168.0.51",
        "port": 22,
        "username": "researcher",
        "destination": "~/pick",
        "password": "must-not-persist",
    }
    with pytest.raises(ValueError, match="delivery payload"):
        validate_job_payload("datasets.copy_pc_key", payload)
