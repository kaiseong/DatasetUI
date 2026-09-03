from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import datasetui.database as database_module
from datasetui.database import Database


def test_initialize_is_idempotent(database: Database) -> None:
    database.initialize()
    assert database.schema_versions() == [1, 2, 3, 4, 5, 6, 7, 8]


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


def test_concurrent_initialization_applies_each_migration_once(tmp_path: Path) -> None:
    path = tmp_path / "concurrent.sqlite3"
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: Database(path).initialize(), range(4)))
    assert Database(path).schema_versions() == [1, 2, 3, 4, 5, 6, 7, 8]


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
