from __future__ import annotations

import uuid
import pytest

import datasetui.delivery as delivery
from datasetui.validation_integrity import VALIDATOR_POLICY, validation_content_manifest
from test_delivery import _registered, _pass_gate


def _payload(dataset):
    return {
        "dataset_id": dataset["id"],
        **{k: dataset[k] for k in ("fingerprint", "storage_area", "relative_path")},
    }


@pytest.mark.parametrize("status", ["queued", "running", "failed", "succeeded"])
def test_latest_nonpassing_gate_revokes_prior_pass(tmp_path, status):
    settings, db, dataset, profile, root = _registered(tmp_path)
    _pass_gate(db, dataset, profile)
    job, _ = db.create_job(
        kind="datasets.validate",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={},
        idempotency_key=str(uuid.uuid4()),
    )
    db.record_validation_run(
        job_id=job["id"],
        dataset_id=dataset["id"],
        dataset_fingerprint=dataset["fingerprint"],
        mode="export_gate",
    )
    if status != "queued":
        db.claim_job(job["id"], worker_id="validator", lease_seconds=120)
    if status == "failed":
        db.fail_job(job["id"], "validation_error", "failure", worker_id="validator")
    if status == "succeeded":
        db.succeed_job(
            job["id"],
            {
                "passed": False,
                "validator_policy": VALIDATOR_POLICY,
                "content_manifest": validation_content_manifest(root),
            },
            worker_id="validator",
        )
    assert not db.has_successful_export_gate(
        dataset_id=dataset["id"], dataset_fingerprint=dataset["fingerprint"]
    )
    with pytest.raises(delivery.ExportGateRequiredError):
        delivery._source(db, settings, _payload(dataset))


def test_legacy_gate_is_not_accepted(tmp_path):
    settings, db, dataset, profile, root = _registered(tmp_path)
    _pass_gate(db, dataset, profile)
    with db.connect() as connection:
        connection.execute(
            "UPDATE jobs SET result_json = ? WHERE kind = 'datasets.validate'",
            ('{"passed":true}',),
        )
    with pytest.raises(delivery.ExportGateRequiredError):
        delivery._source(db, settings, _payload(dataset))


def test_data_mutation_without_info_change_invalidates_gate(tmp_path):
    settings, db, dataset, profile, root = _registered(tmp_path)
    _pass_gate(db, dataset, profile)
    (root / "data/chunk-000/episode_000000.parquet").write_bytes(b"changed")
    with pytest.raises(
        (delivery.ExportGateRequiredError, delivery.RecipeRevisionMismatchError)
    ):
        delivery._source(db, settings, _payload(dataset))


@pytest.mark.parametrize("kind", ["nas", "pc"])
def test_mutation_during_copy_cannot_publish(tmp_path, monkeypatch, kind):
    settings, db, dataset, profile, root = _registered(tmp_path)
    _pass_gate(db, dataset, profile)
    original = delivery.shutil.copytree

    def corrupt_copy(source, destination, *args, **kwargs):
        result = original(source, destination, *args, **kwargs)
        from pathlib import Path

        (Path(destination) / "unexpected.txt").write_text("changed after validation")
        return result

    monkeypatch.setattr(delivery.shutil, "copytree", corrupt_copy)
    monkeypatch.setattr(
        delivery,
        "_copy_to_pc",
        lambda **kwargs: pytest.fail("Unverified bytes reached SSH"),
    )
    if kind == "nas":
        payload = {**_payload(dataset), "output_name": "blocked"}
        job, _ = db.create_job(
            kind="datasets.export_nas",
            queue_name="io",
            profile_id=profile["id"],
            payload=payload,
            idempotency_key="export",
        )
        db.claim_job(job["id"], worker_id="exporter", lease_seconds=120)
        with pytest.raises(delivery.CurationTransformError):
            delivery.export_to_nas(
                database=db,
                settings=settings,
                payload=payload,
                job_id=job["id"],
                worker_id="exporter",
            )
        assert not (settings.nas_root / "exports/blocked").exists()
    else:
        with pytest.raises(delivery.CurationTransformError):
            delivery.copy_to_pc_with_password(
                database=db,
                settings=settings,
                dataset_id=dataset["id"],
                profile_id=profile["id"],
                host="192.168.0.51",
                port=22,
                username="test",
                password="test",
                destination="~/test",
            )
