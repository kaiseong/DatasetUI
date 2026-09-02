from __future__ import annotations

import json
import sqlite3
import unicodedata
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DuplicateProfileNameError(ValueError):
    pass


class ProfileNotFoundError(LookupError):
    pass


class JobNotFoundError(LookupError):
    pass


class IdempotencyConflictError(ValueError):
    pass


MIGRATIONS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (
        1,
        (
            """
            CREATE TABLE profiles (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                name_key TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                archived_at TEXT
            )
            """,
            """
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                queue_name TEXT NOT NULL,
                status TEXT NOT NULL CHECK (
                    status IN (
                        'queued', 'running', 'succeeded', 'failed',
                        'cancelled', 'interrupted'
                    )
                ),
                profile_id TEXT NOT NULL REFERENCES profiles(id),
                payload_json TEXT NOT NULL,
                result_json TEXT,
                error_code TEXT,
                error_message TEXT,
                idempotency_key TEXT NOT NULL,
                rq_job_id TEXT UNIQUE,
                created_at TEXT NOT NULL,
                enqueued_at TEXT,
                started_at TEXT,
                finished_at TEXT,
                UNIQUE (profile_id, idempotency_key)
            )
            """,
            """
            CREATE TABLE job_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                sequence INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (job_id, sequence)
            )
            """,
            "CREATE INDEX jobs_status_created_idx ON jobs(status, created_at DESC)",
            "CREATE INDEX jobs_profile_created_idx ON jobs(profile_id, created_at DESC)",
            "CREATE INDEX job_events_job_idx ON job_events(job_id, sequence)",
        ),
    ),
)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _profile_name_key(name: str) -> str:
    return unicodedata.normalize("NFKC", name).casefold()


class Database:
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
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
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

    def create_profile(self, name: str) -> dict[str, Any]:
        profile_id = str(uuid.uuid4())
        now = utc_now()
        try:
            with self.connect() as connection:
                connection.execute(
                    """
                    INSERT INTO profiles(id, name, name_key, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (profile_id, name, _profile_name_key(name), now, now),
                )
        except sqlite3.IntegrityError as exc:
            if "profiles.name_key" in str(exc):
                raise DuplicateProfileNameError(name) from exc
            raise
        return self.get_profile(profile_id, include_archived=True)

    def list_profiles(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        where = "" if include_archived else "WHERE archived_at IS NULL"
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT id, name, created_at, updated_at, archived_at
                FROM profiles
                {where}
                ORDER BY name_key, created_at
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_profile(
        self, profile_id: str, *, include_archived: bool = False
    ) -> dict[str, Any]:
        archived_clause = "" if include_archived else "AND archived_at IS NULL"
        with self.connect() as connection:
            row = connection.execute(
                f"""
                SELECT id, name, created_at, updated_at, archived_at
                FROM profiles
                WHERE id = ? {archived_clause}
                """,
                (profile_id,),
            ).fetchone()
        if row is None:
            raise ProfileNotFoundError(profile_id)
        return dict(row)

    def update_profile(
        self,
        profile_id: str,
        *,
        name: str | None = None,
        archived: bool | None = None,
    ) -> dict[str, Any]:
        current = self.get_profile(profile_id, include_archived=True)
        next_name = name if name is not None else current["name"]
        if archived is True:
            archived_at = current["archived_at"] or utc_now()
        elif archived is False:
            archived_at = None
        else:
            archived_at = current["archived_at"]
        try:
            with self.connect() as connection:
                connection.execute(
                    """
                    UPDATE profiles
                    SET name = ?, name_key = ?, archived_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        next_name,
                        _profile_name_key(next_name),
                        archived_at,
                        utc_now(),
                        profile_id,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            if "profiles.name_key" in str(exc):
                raise DuplicateProfileNameError(next_name) from exc
            raise
        return self.get_profile(profile_id, include_archived=True)

    def create_job(
        self,
        *,
        kind: str,
        queue_name: str,
        profile_id: str,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> tuple[dict[str, Any], bool]:
        self.get_profile(profile_id)
        existing = self._job_for_idempotency(profile_id, idempotency_key)
        if existing is not None:
            return self._resolve_idempotent_job(
                existing,
                kind=kind,
                queue_name=queue_name,
                payload=payload,
            )

        job_id = str(uuid.uuid4())
        now = utc_now()
        try:
            with self.connect() as connection:
                connection.execute(
                    """
                    INSERT INTO jobs(
                        id, kind, queue_name, status, profile_id, payload_json,
                        idempotency_key, created_at
                    ) VALUES (?, ?, ?, 'queued', ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        kind,
                        queue_name,
                        profile_id,
                        _json_dump(payload),
                        idempotency_key,
                        now,
                    ),
                )
                self._append_event(
                    connection,
                    job_id,
                    "queued",
                    {"kind": kind, "queue": queue_name},
                    now=now,
                )
        except sqlite3.IntegrityError as exc:
            if "jobs.profile_id, jobs.idempotency_key" in str(exc):
                existing = self._job_for_idempotency(profile_id, idempotency_key)
                if existing is not None:
                    return self._resolve_idempotent_job(
                        existing,
                        kind=kind,
                        queue_name=queue_name,
                        payload=payload,
                    )
            raise
        return self.get_job(job_id), True

    def mark_enqueued(self, job_id: str, rq_job_id: str) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as connection:
            updated = connection.execute(
                """
                UPDATE jobs SET rq_job_id = ?, enqueued_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (rq_job_id, now, job_id),
            ).rowcount
            if updated != 1:
                raise JobNotFoundError(job_id)
            self._append_event(
                connection,
                job_id,
                "dispatched",
                {"rq_job_id": rq_job_id},
                now=now,
            )
        return self.get_job(job_id)

    def record_dispatch_error(self, job_id: str, message: str) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as connection:
            updated = connection.execute(
                """
                UPDATE jobs SET error_code = 'queue_unavailable', error_message = ?
                WHERE id = ? AND status = 'queued'
                """,
                (message, job_id),
            ).rowcount
            if updated == 1:
                self._append_event(
                    connection,
                    job_id,
                    "dispatch_uncertain",
                    {"error_code": "queue_unavailable"},
                    now=now,
                )
        return self.get_job(job_id)

    def clear_dispatch_error(self, job_id: str) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as connection:
            updated = connection.execute(
                """
                UPDATE jobs SET error_code = NULL, error_message = NULL
                WHERE id = ? AND status = 'queued' AND error_code = 'queue_unavailable'
                """,
                (job_id,),
            ).rowcount
            if updated == 1:
                self._append_event(
                    connection,
                    job_id,
                    "dispatch_recovered",
                    {},
                    now=now,
                )
        return self.get_job(job_id)

    def claim_job(self, job_id: str) -> dict[str, Any] | None:
        now = utc_now()
        with self.connect() as connection:
            updated = connection.execute(
                """
                UPDATE jobs
                SET status = 'running', started_at = ?,
                    error_code = NULL, error_message = NULL
                WHERE id = ? AND status = 'queued'
                """,
                (now, job_id),
            ).rowcount
            if updated != 1:
                return None
            self._append_event(connection, job_id, "running", {}, now=now)
        return self.get_job(job_id)

    def succeed_job(self, job_id: str, result: dict[str, Any]) -> dict[str, Any]:
        return self._finish_job(
            job_id,
            from_statuses=("running",),
            status="succeeded",
            result=result,
        )

    def fail_job(self, job_id: str, code: str, message: str) -> dict[str, Any]:
        return self._finish_job(
            job_id,
            from_statuses=("running",),
            status="failed",
            error_code=code,
            error_message=message,
        )

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT id, kind, queue_name, status, profile_id, payload_json,
                       result_json, error_code, error_message, idempotency_key,
                       rq_job_id, created_at, enqueued_at, started_at, finished_at
                FROM jobs WHERE id = ?
                """,
                (job_id,),
            ).fetchone()
        if row is None:
            raise JobNotFoundError(job_id)
        return self._decode_job(row)

    def list_jobs(
        self,
        *,
        profile_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if profile_id:
            clauses.append("profile_id = ?")
            parameters.append(profile_id)
        if status:
            clauses.append("status = ?")
            parameters.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(limit)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT id, kind, queue_name, status, profile_id, payload_json,
                       result_json, error_code, error_message, idempotency_key,
                       rq_job_id, created_at, enqueued_at, started_at, finished_at
                FROM jobs
                {where}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [self._decode_job(row) for row in rows]

    def list_job_events(self, job_id: str) -> list[dict[str, Any]]:
        self.get_job(job_id)
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT sequence, event_type, payload_json, created_at
                FROM job_events WHERE job_id = ? ORDER BY sequence
                """,
                (job_id,),
            ).fetchall()
        return [
            {
                "sequence": row["sequence"],
                "event_type": row["event_type"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def _job_for_idempotency(
        self, profile_id: str, idempotency_key: str
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT id FROM jobs
                WHERE profile_id = ? AND idempotency_key = ?
                """,
                (profile_id, idempotency_key),
            ).fetchone()
        return self.get_job(row["id"]) if row else None

    @staticmethod
    def _resolve_idempotent_job(
        existing: dict[str, Any],
        *,
        kind: str,
        queue_name: str,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        if (
            existing["kind"] != kind
            or existing["queue_name"] != queue_name
            or existing["payload"] != payload
        ):
            raise IdempotencyConflictError(
                "idempotency key is already bound to a different request"
            )
        return existing, False

    def _finish_job(
        self,
        job_id: str,
        *,
        from_statuses: tuple[str, ...],
        status: str,
        result: dict[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        placeholders = ",".join("?" for _ in from_statuses)
        with self.connect() as connection:
            updated = connection.execute(
                f"""
                UPDATE jobs
                SET status = ?, result_json = ?, error_code = ?, error_message = ?,
                    finished_at = ?
                WHERE id = ? AND status IN ({placeholders})
                """,
                (
                    status,
                    _json_dump(result) if result is not None else None,
                    error_code,
                    error_message,
                    now,
                    job_id,
                    *from_statuses,
                ),
            ).rowcount
            if updated != 1:
                raise JobNotFoundError(job_id)
            event_payload: dict[str, Any] = {}
            if error_code:
                event_payload["error_code"] = error_code
            self._append_event(connection, job_id, status, event_payload, now=now)
        return self.get_job(job_id)

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
            (job_id, sequence, event_type, _json_dump(payload), now),
        )

    @staticmethod
    def _decode_job(row: sqlite3.Row) -> dict[str, Any]:
        job = dict(row)
        job["payload"] = json.loads(job.pop("payload_json"))
        result_json = job.pop("result_json")
        job["result"] = json.loads(result_json) if result_json else None
        return job
