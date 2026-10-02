"""Connection handling, migrations and shared row decoding."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from datasetui.database.schema import json_dump, MIGRATIONS, utc_now


class DatabaseCore:
    """Connection handling, migrations and shared row decoding."""

    def __init__(self, path: Path):
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )
            applied = {
                row["version"]
                for row in connection.execute("SELECT version FROM schema_migrations")
            }
            for version, statements in MIGRATIONS:
                if version in applied:
                    continue
                for statement in statements:
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (version, utc_now()),
                )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        deadline = time.monotonic() + 5.0
        while True:
            try:
                connection.execute("PRAGMA journal_mode = WAL")
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or time.monotonic() >= deadline:
                    connection.close()
                    raise
                time.sleep(0.025)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def ping(self) -> bool:
        with self.connect() as connection:
            return connection.execute("SELECT 1").fetchone()[0] == 1

    def schema_versions(self) -> list[int]:
        with self.connect() as connection:
            return [
                row["version"]
                for row in connection.execute(
                    "SELECT version FROM schema_migrations ORDER BY version"
                )
            ]

    @staticmethod
    def _append_event(
        connection: sqlite3.Connection,
        job_id: str,
        event_type: str,
        payload: dict[str, Any],
        *,
        now: str,
    ) -> None:
        sequence = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM job_events WHERE job_id = ?",
            (job_id,),
        ).fetchone()[0]
        connection.execute(
            """
            INSERT INTO job_events(job_id, sequence, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (job_id, sequence, event_type, json_dump(payload), now),
        )

    @staticmethod
    def _decode_job(row: sqlite3.Row) -> dict[str, Any]:
        job = dict(row)
        ready = job.pop("validation_ready", 1)
        job["wait_reason"] = (
            ("credentials" if ready == -1 else "validation")
            if ready != 1 and job["status"] == "queued"
            else None
        )
        job["queue_position"] = None
        job["cancellation_requested"] = job["cancellation_requested_at"] is not None
        job["payload"] = json.loads(job.pop("payload_json"))
        result_json = job.pop("result_json")
        job["result"] = json.loads(result_json) if result_json else None
        progress_json = job.pop("progress_json")
        job["progress"] = json.loads(progress_json) if progress_json else None
        return job

    @staticmethod
    def _decode_dataset(row: sqlite3.Row) -> dict[str, Any]:
        dataset = dict(row)
        display_name = dataset.pop("display_name")
        if display_name is not None:
            dataset["name"] = display_name
        dataset["available"] = dataset.pop("missing_since") is None
        return dataset

    @staticmethod
    def _decode_recipe(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["trim_config"] = json.loads(result.pop("trim_config_json"))
        result["include_annotations"] = bool(result["include_annotations"])
        result["relative_action"] = json.loads(result.pop("relative_action_json"))
        result["split_config"] = json.loads(result.pop("split_config_json"))
        return result
