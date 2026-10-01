from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import datasetui.database as database_module
from datasetui.database import (
    Database,
    JobCancellationConflictError,
    JobOwnershipError,
)


def test_initialize_is_idempotent(database: Database) -> None:
    database.initialize()
    assert database.schema_versions() == list(range(1, 17))


def test_api_database_reopen_does_not_interrupt_running_job(database: Database) -> None:
    profile = database.create_profile("Researcher")
    job, _ = database.create_job(
        kind="phase2.smoke",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={},
        idempotency_key="restart-safe",
    )
    assert database.claim_job(job["id"])["status"] == "running"

    reopened = Database(database.path)
    reopened.initialize()
    assert reopened.get_job(job["id"])["status"] == "running"
    assert reopened.succeed_job(job["id"], {"ok": True})["status"] == "succeeded"


def test_cancel_queued_job_is_owned_and_idempotent(database: Database) -> None:
    owner = database.create_profile("Owner")
    other = database.create_profile("Other")
    job, _ = database.create_job(
        kind="phase2.smoke",
        queue_name="cpu",
        profile_id=owner["id"],
        payload={},
        idempotency_key="cancel-owned",
    )
    database.mark_enqueued(job["id"], f"datasetui-{job['id']}")

    with pytest.raises(JobOwnershipError):
        database.cancel_queued_job(job["id"], profile_id=other["id"])

    cancelled = database.cancel_queued_job(job["id"], profile_id=owner["id"])
    repeated = database.cancel_queued_job(job["id"], profile_id=owner["id"])

    assert cancelled["status"] == repeated["status"] == "cancelled"
    assert cancelled["finished_at"] is not None
    assert [event["event_type"] for event in database.list_job_events(job["id"])] == [
        "queued",
        "dispatched",
        "cancelled",
    ]
    assert database.claim_job(job["id"], worker_id="late-worker") is None


def test_cancel_running_job_records_cooperative_request(database: Database) -> None:
    profile = database.create_profile("Runner")
    job, _ = database.create_job(
        kind="phase2.smoke",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={},
        idempotency_key="cancel-running",
    )
    database.claim_job(job["id"], worker_id="active-worker")

    requested = database.cancel_queued_job(job["id"], profile_id=profile["id"])

    assert requested["status"] == "running"
    assert requested["cancellation_requested"] is True
    assert [event["event_type"] for event in database.list_job_events(job["id"])] == [
        "queued",
        "running",
        "cancellation_requested",
    ]


def test_claim_and_cancel_race_has_exactly_one_winner(database: Database) -> None:
    profile = database.create_profile("Racer")
    job, _ = database.create_job(
        kind="phase2.smoke",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={},
        idempotency_key="claim-cancel-race",
    )
    barrier = threading.Barrier(2)

    def claim() -> str:
        barrier.wait()
        return "claimed" if database.claim_job(job["id"], worker_id="racer") else "lost"

    def cancel() -> str:
        barrier.wait()
        result = database.cancel_queued_job(job["id"], profile_id=profile["id"])
        return result["status"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        claim_future = executor.submit(claim)
        cancel_future = executor.submit(cancel)
        results = {claim_future.result(), cancel_future.result()}

    assert results in ({"claimed", "running"}, {"lost", "cancelled"})
    final_status = database.get_job(job["id"])["status"]
    assert final_status in {"running", "cancelled"}
    if final_status == "running":
        assert database.get_job(job["id"])["cancellation_requested"] is True
    else:
        assert database.list_job_events(job["id"])[-1]["event_type"] == "cancelled"


def test_concurrent_initialization_applies_each_migration_once(tmp_path: Path) -> None:
    path = tmp_path / "concurrent.sqlite3"
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: Database(path).initialize(), range(4)))
    assert Database(path).schema_versions() == list(range(1, 17))


def test_existing_schema_10_database_adds_validation_tombstone(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "schema-10.sqlite3"
    migrations = database_module.MIGRATIONS
    monkeypatch.setattr(database_module, "MIGRATIONS", migrations[:10])
    Database(path).initialize()
    monkeypatch.setattr(database_module, "MIGRATIONS", migrations)

    Database(path).initialize()

    assert Database(path).schema_versions() == list(range(1, 17))
    with sqlite3.connect(path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(validation_runs)")
        }
    assert "deleted_at" in columns


def test_existing_schema_11_database_adds_dataset_display_name(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "schema-11.sqlite3"
    migrations = database_module.MIGRATIONS
    monkeypatch.setattr(database_module, "MIGRATIONS", migrations[:11])
    Database(path).initialize()
    monkeypatch.setattr(database_module, "MIGRATIONS", migrations)

    Database(path).initialize()

    assert Database(path).schema_versions() == list(range(1, 17))


def test_existing_schema_12_database_adds_job_progress(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "schema-12.sqlite3"
    migrations = database_module.MIGRATIONS
    monkeypatch.setattr(database_module, "MIGRATIONS", migrations[:12])
    Database(path).initialize()
    monkeypatch.setattr(database_module, "MIGRATIONS", migrations)

    Database(path).initialize()

    assert Database(path).schema_versions() == list(range(1, 17))
    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
    assert "progress_json" in columns
    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(datasets)")}
    assert "display_name" in columns


def test_failed_migration_is_rolled_back(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "broken.sqlite3"
    monkeypatch.setattr(
        database_module,
        "MIGRATIONS",
        ((1, ("CREATE TABLE partial_table(id INTEGER)", "INVALID SQL")),),
    )
    with pytest.raises(sqlite3.OperationalError):
        Database(path).initialize()

    with sqlite3.connect(path) as connection:
        names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert "partial_table" not in names


def test_expired_worker_is_requeued_without_allowing_stale_completion(
    database: Database,
) -> None:
    profile = database.create_profile("Researcher")
    job, _ = database.create_job(
        kind="phase2.smoke",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={},
        idempotency_key="expired-worker",
    )
    database.claim_job(job["id"], worker_id="worker-old", lease_seconds=120)
    expired = (
        (datetime.now(timezone.utc) - timedelta(seconds=1))
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
    with database.connect() as connection:
        connection.execute(
            "UPDATE jobs SET lease_expires_at = ? WHERE id = ?",
            (expired, job["id"]),
        )

    assert database.requeue_expired_jobs() == [job["id"]]
    assert database.get_job(job["id"])["status"] == "queued"
    assert database.rq_job_id_for(job["id"]).endswith("-1")
    assert (
        database.claim_job(job["id"], worker_id="worker-new", lease_seconds=120)[
            "status"
        ]
        == "running"
    )

    with pytest.raises(database_module.JobLeaseLostError):
        database.succeed_job(job["id"], {"stale": True}, worker_id="worker-old")
    assert (
        database.succeed_job(job["id"], {"ok": True}, worker_id="worker-new")["status"]
        == "succeeded"
    )
