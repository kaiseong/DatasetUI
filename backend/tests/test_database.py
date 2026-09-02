from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import datasetui.database as database_module
from datasetui.database import Database


def test_initialize_is_idempotent(database: Database) -> None:
    database.initialize()
    assert database.schema_versions() == [1, 2]


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
    assert Database(path).schema_versions() == [1, 2]


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
