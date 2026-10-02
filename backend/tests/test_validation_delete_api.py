from __future__ import annotations

from test_curation_api import _record, _register

from datasetui.validation.integrity import VALIDATOR_POLICY


def _validation_run(
    database,
    dataset,
    profile,
    *,
    mode: str,
    key: str,
    status: str = "succeeded",
):
    job, _ = database.create_job(
        kind="datasets.validate",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={"dataset_id": dataset["id"], "mode": mode},
        idempotency_key=key,
    )
    database.record_validation_run(
        job_id=job["id"],
        dataset_id=dataset["id"],
        dataset_fingerprint=dataset["fingerprint"],
        mode=mode,
    )
    if status == "queued":
        return job
    database.claim_job(job["id"], worker_id="validator", lease_seconds=120)
    if status == "running":
        return job
    if status == "succeeded":
        database.succeed_job(
            job["id"],
            {
                "passed": True,
                "validator_policy": VALIDATOR_POLICY,
                "content_manifest": {"tree_sha256": "a" * 64},
            },
            worker_id="validator",
        )
    else:
        database.fail_job(
            job["id"], "validation_failed", "failed", worker_id="validator"
        )
    return job


def test_terminal_validation_delete_is_idempotent_and_preserves_job(
    client, database
) -> None:
    dataset = _register(database)
    profile = database.create_profile("Validation deletion")
    job = _validation_run(
        database, dataset, profile, mode="full", key="delete-terminal"
    )

    first = client.delete(f"/api/v1/datasets/{dataset['id']}/validations/{job['id']}")
    second = client.delete(f"/api/v1/datasets/{dataset['id']}/validations/{job['id']}")

    assert first.status_code == second.status_code == 204
    assert client.get(f"/api/v1/datasets/{dataset['id']}/validations").json() == []
    assert database.get_job(job["id"])["status"] == "succeeded"


def test_active_validation_delete_is_blocked(client, database) -> None:
    dataset = _register(database)
    profile = database.create_profile("Active validation")
    queued = _validation_run(
        database, dataset, profile, mode="quick", key="delete-queued", status="queued"
    )
    running = _validation_run(
        database,
        dataset,
        profile,
        mode="full",
        key="delete-running",
        status="running",
    )

    for job in (queued, running):
        response = client.delete(
            f"/api/v1/datasets/{dataset['id']}/validations/{job['id']}"
        )
        assert response.status_code == 409
    assert len(database.list_validation_runs(dataset["id"])) == 2


def test_validation_delete_rejects_missing_or_wrong_dataset(client, database) -> None:
    dataset = _register(database)
    other_record = _record(fingerprint="other-revision")
    other_record.update(relative_path="lab/other", name="other", info_mtime_ns=2)
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw",
        records=[_record(), other_record],
        scan_generation=generation,
    )
    other = next(
        item
        for item in database.list_datasets()
        if item["relative_path"] == "lab/other"
    )
    profile = database.create_profile("Wrong dataset")
    job = _validation_run(database, dataset, profile, mode="quick", key="wrong-dataset")

    wrong = client.delete(f"/api/v1/datasets/{other['id']}/validations/{job['id']}")
    missing = client.delete(
        f"/api/v1/datasets/{dataset['id']}/validations/00000000-0000-4000-8000-000000000000"
    )

    assert wrong.status_code == missing.status_code == 404
    assert len(database.list_validation_runs(dataset["id"])) == 1


def test_deleted_latest_export_gate_never_resurrects_older_pass(
    client, database
) -> None:
    dataset = _register(database)
    profile = database.create_profile("Gate deletion")
    older = _validation_run(
        database, dataset, profile, mode="export_gate", key="gate-older"
    )
    unrelated = _validation_run(
        database, dataset, profile, mode="quick", key="gate-unrelated"
    )
    newer = _validation_run(
        database, dataset, profile, mode="export_gate", key="gate-newer"
    )
    gate_args = {
        "dataset_id": dataset["id"],
        "dataset_fingerprint": dataset["fingerprint"],
    }
    assert database.has_successful_export_gate(**gate_args)

    assert (
        client.delete(
            f"/api/v1/datasets/{dataset['id']}/validations/{unrelated['id']}"
        ).status_code
        == 204
    )
    assert database.has_successful_export_gate(**gate_args)
    assert (
        client.delete(
            f"/api/v1/datasets/{dataset['id']}/validations/{newer['id']}"
        ).status_code
        == 204
    )

    assert database.has_successful_export_gate(**gate_args) is False
    assert database.get_job(older["id"])["status"] == "succeeded"


def test_deleted_latest_failed_export_gate_never_resurrects_older_pass(
    client, database
) -> None:
    dataset = _register(database)
    profile = database.create_profile("Failed gate deletion")
    _validation_run(database, dataset, profile, mode="export_gate", key="passed-gate")
    failed = _validation_run(
        database,
        dataset,
        profile,
        mode="export_gate",
        key="failed-gate",
        status="failed",
    )
    gate_args = {
        "dataset_id": dataset["id"],
        "dataset_fingerprint": dataset["fingerprint"],
    }
    assert database.has_successful_export_gate(**gate_args) is False

    response = client.delete(
        f"/api/v1/datasets/{dataset['id']}/validations/{failed['id']}"
    )

    assert response.status_code == 204
    assert database.has_successful_export_gate(**gate_args) is False


def test_deleting_older_export_gate_preserves_newer_pass(client, database) -> None:
    dataset = _register(database)
    profile = database.create_profile("Older gate deletion")
    older = _validation_run(
        database, dataset, profile, mode="export_gate", key="older-passed-gate"
    )
    _validation_run(
        database, dataset, profile, mode="export_gate", key="newer-passed-gate"
    )
    gate_args = {
        "dataset_id": dataset["id"],
        "dataset_fingerprint": dataset["fingerprint"],
    }

    response = client.delete(
        f"/api/v1/datasets/{dataset['id']}/validations/{older['id']}"
    )

    assert response.status_code == 204
    assert database.has_successful_export_gate(**gate_args)
