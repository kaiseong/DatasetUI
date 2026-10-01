from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from datasetui.database import Database, JobLeaseLostError


def _job(database: Database, *, kind: str = "datasets.merge", key: str = "merge"):
    profile = database.create_profile(f"Profile {key}")
    job, _ = database.create_job(
        kind=kind,
        queue_name="cpu",
        profile_id=profile["id"],
        payload={},
        idempotency_key=key,
    )
    return profile, job


def test_progress_persists_in_get_and_list(database: Database) -> None:
    profile, job = _job(database)
    database.claim_job(job["id"], worker_id="worker-1")
    progress = {
        "stage": "video",
        "completed": 3,
        "total": 8,
        "unit": "episodes",
        "elapsed_seconds": 1.25,
        "current_item": "episode 3",
        "output_name": "merged-v1",
    }

    database.update_job_progress(job["id"], worker_id="worker-1", progress=progress)

    assert database.get_job(job["id"])["progress"] == progress
    assert database.list_jobs(profile_id=profile["id"])[0]["progress"] == progress


def test_progress_requires_current_running_lease(database: Database) -> None:
    _, job = _job(database)
    database.claim_job(job["id"], worker_id="worker-1")
    progress = {
        "stage": "read",
        "completed": 0,
        "total": 1,
        "unit": "items",
        "elapsed_seconds": 0.0,
    }

    with pytest.raises(JobLeaseLostError):
        database.update_job_progress(job["id"], worker_id="worker-2", progress=progress)

    expired = (
        (datetime.now(timezone.utc) - timedelta(seconds=1))
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
    with database.connect() as connection:
        connection.execute(
            "UPDATE jobs SET lease_expires_at = ? WHERE id = ?", (expired, job["id"])
        )
    with pytest.raises(JobLeaseLostError):
        database.update_job_progress(job["id"], worker_id="worker-1", progress=progress)


def test_progress_clears_when_job_is_requeued_and_reclaimed(database: Database) -> None:
    _, job = _job(database)
    database.claim_job(job["id"], worker_id="worker-1")
    database.update_job_progress(
        job["id"],
        worker_id="worker-1",
        progress={
            "stage": "write",
            "completed": 1,
            "total": 2,
            "unit": "files",
            "elapsed_seconds": 0.5,
        },
    )
    expired = (
        (datetime.now(timezone.utc) - timedelta(seconds=1))
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
    with database.connect() as connection:
        connection.execute(
            "UPDATE jobs SET lease_expires_at = ? WHERE id = ?", (expired, job["id"])
        )

    assert database.requeue_expired_jobs() == [job["id"]]
    assert database.get_job(job["id"])["progress"] is None
    assert database.claim_job(job["id"], worker_id="worker-2")["progress"] is None


def test_jobs_api_filters_kind_before_limit(client, database: Database) -> None:
    profile = database.create_profile("Filter profile")
    merge, _ = database.create_job(
        kind="datasets.merge",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={},
        idempotency_key="merge-filter",
    )
    database.create_job(
        kind="phase2.smoke",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={},
        idempotency_key="newer-other-kind",
    )
    running_merge, _ = database.create_job(
        kind="datasets.merge",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={},
        idempotency_key="newer-running-merge",
    )
    database.claim_job(running_merge["id"], worker_id="filter-worker")

    response = client.get(
        "/api/v1/jobs",
        params={
            "profile_id": profile["id"],
            "status": "queued",
            "kind": "datasets.merge",
            "limit": 1,
        },
    )

    assert response.status_code == 200
    assert [item["id"] for item in response.json()] == [merge["id"]]
