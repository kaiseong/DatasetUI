from __future__ import annotations

import json
import sqlite3
import time
import unicodedata
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class DuplicateProfileNameError(ValueError):
    pass


class ProfileNotFoundError(LookupError):
    pass


class JobNotFoundError(LookupError):
    pass


class JobLeaseLostError(RuntimeError):
    pass


class DatasetNotFoundError(LookupError):
    pass


class DatasetNotReadyError(RuntimeError):
    pass


class FlagRevisionConflictError(RuntimeError):
    pass


class DuplicateRecipeNameError(ValueError):
    pass


class RecipeNotFoundError(LookupError):
    pass


class RecipeRevisionMismatchError(RuntimeError):
    pass


class IdempotencyConflictError(ValueError):
    pass


class ImmutableRevisionConflictError(RuntimeError):
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
    (
        2,
        (
            """
            CREATE TABLE datasets (
                id TEXT PRIMARY KEY,
                storage_area TEXT NOT NULL CHECK (storage_area IN ('raw', 'derived')),
                relative_path TEXT NOT NULL,
                name TEXT NOT NULL,
                codebase_version TEXT,
                readiness TEXT NOT NULL CHECK (
                    readiness IN ('ready', 'incomplete', 'unsupported', 'invalid')
                ),
                robot_type TEXT,
                total_episodes INTEGER,
                total_frames INTEGER,
                total_tasks INTEGER,
                fps REAL,
                fingerprint TEXT NOT NULL,
                info_mtime_ns INTEGER NOT NULL,
                info_size INTEGER NOT NULL,
                scan_error TEXT,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                last_scan_generation INTEGER NOT NULL,
                missing_since TEXT,
                UNIQUE (storage_area, relative_path)
            )
            """,
            """
            CREATE TABLE dataset_scan_generations (
                storage_area TEXT PRIMARY KEY CHECK (storage_area IN ('raw', 'derived')),
                generation INTEGER NOT NULL,
                started_at TEXT NOT NULL
            )
            """,
            "CREATE INDEX datasets_library_idx ON datasets(missing_since, readiness, name)",
            "CREATE INDEX datasets_storage_idx ON datasets(storage_area, relative_path)",
        ),
    ),
    (
        3,
        (
            "ALTER TABLE jobs ADD COLUMN worker_id TEXT",
            "ALTER TABLE jobs ADD COLUMN heartbeat_at TEXT",
            "ALTER TABLE jobs ADD COLUMN lease_expires_at TEXT",
            "ALTER TABLE jobs ADD COLUMN attempt INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE jobs ADD COLUMN dispatch_generation INTEGER NOT NULL DEFAULT 0",
            "CREATE INDEX jobs_lease_idx ON jobs(status, lease_expires_at)",
            """
            CREATE TABLE hf_sources (
                repo_id TEXT PRIMARY KEY,
                desired_commit_sha TEXT NOT NULL,
                generation INTEGER NOT NULL,
                current_commit_sha TEXT,
                current_generation INTEGER,
                pointer_confirmed INTEGER NOT NULL DEFAULT 0 CHECK (
                    pointer_confirmed IN (0, 1)
                ),
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE hf_import_requests (
                job_id TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
                repo_id TEXT NOT NULL,
                requested_revision TEXT NOT NULL,
                commit_sha TEXT NOT NULL,
                generation INTEGER NOT NULL
            )
            """,
            """
            CREATE TABLE hf_revisions (
                repo_id TEXT NOT NULL,
                commit_sha TEXT NOT NULL,
                requested_revision TEXT NOT NULL,
                relative_path TEXT NOT NULL UNIQUE,
                manifest_sha256 TEXT NOT NULL,
                file_count INTEGER NOT NULL,
                total_bytes INTEGER NOT NULL,
                imported_at TEXT NOT NULL,
                PRIMARY KEY (repo_id, commit_sha)
            )
            """,
            "CREATE INDEX hf_import_requests_repo_idx ON hf_import_requests(repo_id, generation)",
            "CREATE INDEX hf_revisions_repo_idx ON hf_revisions(repo_id, imported_at DESC)",
        ),
    ),
    (
        4,
        (
            """
            CREATE TABLE flag_sets (
                dataset_id TEXT NOT NULL REFERENCES datasets(id),
                dataset_fingerprint TEXT NOT NULL,
                profile_id TEXT NOT NULL REFERENCES profiles(id),
                revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
                updated_at TEXT NOT NULL,
                PRIMARY KEY (dataset_id, dataset_fingerprint, profile_id)
            )
            """,
            """
            CREATE TABLE episode_flags (
                dataset_id TEXT NOT NULL,
                dataset_fingerprint TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                episode_index INTEGER NOT NULL CHECK (episode_index >= 0),
                created_at TEXT NOT NULL,
                PRIMARY KEY (
                    dataset_id, dataset_fingerprint, profile_id, episode_index
                ),
                FOREIGN KEY (dataset_id, dataset_fingerprint, profile_id)
                    REFERENCES flag_sets(
                        dataset_id, dataset_fingerprint, profile_id
                    ) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE curation_recipes (
                id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL REFERENCES datasets(id),
                dataset_fingerprint TEXT NOT NULL,
                profile_id TEXT NOT NULL REFERENCES profiles(id),
                name TEXT NOT NULL,
                name_key TEXT NOT NULL,
                selection_mode TEXT NOT NULL CHECK (
                    selection_mode IN ('all', 'flagged', 'unflagged')
                ),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                archived_at TEXT,
                UNIQUE (
                    dataset_id, dataset_fingerprint, profile_id, name_key
                )
            )
            """,
            """
            CREATE TABLE curation_recipe_snapshots (
                id TEXT PRIMARY KEY,
                recipe_id TEXT NOT NULL REFERENCES curation_recipes(id),
                dataset_id TEXT NOT NULL REFERENCES datasets(id),
                dataset_fingerprint TEXT NOT NULL,
                profile_id TEXT NOT NULL REFERENCES profiles(id),
                recipe_name TEXT NOT NULL,
                selection_mode TEXT NOT NULL CHECK (
                    selection_mode IN ('all', 'flagged', 'unflagged')
                ),
                flag_revision INTEGER NOT NULL CHECK (flag_revision >= 0),
                flagged_episode_indices_json TEXT NOT NULL,
                selected_episode_indices_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            "CREATE INDEX episode_flags_lookup_idx ON episode_flags(dataset_id, dataset_fingerprint, profile_id, episode_index)",
            "CREATE INDEX curation_recipes_lookup_idx ON curation_recipes(dataset_id, dataset_fingerprint, profile_id, archived_at, updated_at DESC)",
            "CREATE INDEX curation_recipe_snapshots_recipe_idx ON curation_recipe_snapshots(recipe_id, created_at DESC)",
        ),
    ),
)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _utc_after(seconds: int) -> str:
    return (
        (datetime.now(timezone.utc) + timedelta(seconds=seconds))
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _profile_name_key(name: str) -> str:
    return unicodedata.normalize("NFKC", name).casefold()


def _recipe_name_key(name: str) -> str:
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

    def create_hf_import_job(
        self,
        *,
        profile_id: str,
        repo_id: str,
        dataset_name: str,
        requested_revision: str,
        commit_sha: str,
        expected_file_count: int,
        expected_total_bytes: int | None,
        idempotency_key: str,
    ) -> tuple[dict[str, Any], bool]:
        self.get_profile(profile_id)
        existing = self._job_for_idempotency(profile_id, idempotency_key)
        if existing is not None:
            return self._resolve_hf_idempotent_job(
                existing,
                repo_id=repo_id,
                dataset_name=dataset_name,
                requested_revision=requested_revision,
                commit_sha=commit_sha,
                expected_file_count=expected_file_count,
                expected_total_bytes=expected_total_bytes,
            )

        job_id = str(uuid.uuid4())
        now = utc_now()
        try:
            with self.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                source = connection.execute(
                    "SELECT generation FROM hf_sources WHERE repo_id = ?",
                    (repo_id,),
                ).fetchone()
                generation = (source["generation"] if source else 0) + 1
                connection.execute(
                    """
                    INSERT INTO hf_sources(
                        repo_id, desired_commit_sha, generation, updated_at
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(repo_id) DO UPDATE SET
                        desired_commit_sha = excluded.desired_commit_sha,
                        generation = excluded.generation,
                        updated_at = excluded.updated_at
                    """,
                    (repo_id, commit_sha, generation, now),
                )
                payload = {
                    "repo_id": repo_id,
                    "dataset_name": dataset_name,
                    "requested_revision": requested_revision,
                    "commit_sha": commit_sha,
                    "generation": generation,
                    "expected_file_count": expected_file_count,
                    "expected_total_bytes": expected_total_bytes,
                }
                connection.execute(
                    """
                    INSERT INTO jobs(
                        id, kind, queue_name, status, profile_id, payload_json,
                        idempotency_key, created_at
                    ) VALUES (?, 'hf.import', 'io', 'queued', ?, ?, ?, ?)
                    """,
                    (job_id, profile_id, _json_dump(payload), idempotency_key, now),
                )
                connection.execute(
                    """
                    INSERT INTO hf_import_requests(
                        job_id, repo_id, requested_revision, commit_sha, generation
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (job_id, repo_id, requested_revision, commit_sha, generation),
                )
                self._append_event(
                    connection,
                    job_id,
                    "queued",
                    {"kind": "hf.import", "queue": "io"},
                    now=now,
                )
        except sqlite3.IntegrityError as exc:
            if "jobs.profile_id, jobs.idempotency_key" in str(exc):
                existing = self._job_for_idempotency(profile_id, idempotency_key)
                if existing is not None:
                    return self._resolve_hf_idempotent_job(
                        existing,
                        repo_id=repo_id,
                        dataset_name=dataset_name,
                        requested_revision=requested_revision,
                        commit_sha=commit_sha,
                        expected_file_count=expected_file_count,
                        expected_total_bytes=expected_total_bytes,
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

    def claim_job(
        self,
        job_id: str,
        *,
        worker_id: str = "manual",
        lease_seconds: int = 120,
    ) -> dict[str, Any] | None:
        now = utc_now()
        lease_expires_at = _utc_after(lease_seconds)
        with self.connect() as connection:
            updated = connection.execute(
                """
                UPDATE jobs
                SET status = 'running', started_at = ?,
                    error_code = NULL, error_message = NULL,
                    worker_id = ?, heartbeat_at = ?, lease_expires_at = ?,
                    attempt = attempt + 1
                WHERE id = ? AND status = 'queued'
                """,
                (now, worker_id, now, lease_expires_at, job_id),
            ).rowcount
            if updated != 1:
                return None
            self._append_event(connection, job_id, "running", {}, now=now)
        return self.get_job(job_id)

    def heartbeat_job(self, job_id: str, *, worker_id: str, lease_seconds: int) -> bool:
        now = utc_now()
        with self.connect() as connection:
            return (
                connection.execute(
                    """
                    UPDATE jobs SET heartbeat_at = ?, lease_expires_at = ?
                    WHERE id = ? AND status = 'running' AND worker_id = ?
                    """,
                    (now, _utc_after(lease_seconds), job_id, worker_id),
                ).rowcount
                == 1
            )

    def requeue_expired_jobs(self) -> list[str]:
        now = utc_now()
        requeued: list[str] = []
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT id FROM jobs
                WHERE status = 'running'
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at < ?
                """,
                (now,),
            ).fetchall()
            for row in rows:
                updated = connection.execute(
                    """
                    UPDATE jobs
                    SET status = 'queued', error_code = 'worker_lost',
                        error_message = 'The worker stopped responding', finished_at = NULL,
                        started_at = NULL, rq_job_id = NULL,
                        worker_id = NULL, heartbeat_at = NULL, lease_expires_at = NULL,
                        dispatch_generation = dispatch_generation + 1
                    WHERE id = ? AND status = 'running' AND lease_expires_at < ?
                    """,
                    (row["id"], now),
                ).rowcount
                if updated == 1:
                    self._append_event(
                        connection,
                        row["id"],
                        "interrupted",
                        {"error_code": "worker_lost"},
                        now=now,
                    )
                    self._append_event(
                        connection,
                        row["id"],
                        "requeued",
                        {},
                        now=now,
                    )
                    requeued.append(row["id"])
        return requeued

    def rq_job_id_for(self, job_id: str) -> str:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT dispatch_generation FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise JobNotFoundError(job_id)
        generation = row["dispatch_generation"]
        return (
            f"datasetui-{job_id}"
            if generation == 0
            else f"datasetui-{job_id}-{generation}"
        )

    def assert_job_lease(self, job_id: str, *, worker_id: str) -> None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM jobs
                WHERE id = ? AND status = 'running' AND worker_id = ?
                  AND lease_expires_at >= ?
                """,
                (job_id, worker_id, utc_now()),
            ).fetchone()
        if row is None:
            raise JobLeaseLostError(job_id)

    def record_hf_revision(
        self,
        *,
        repo_id: str,
        requested_revision: str,
        commit_sha: str,
        relative_path: str,
        manifest_sha256: str,
        file_count: int,
        total_bytes: int,
    ) -> None:
        record = {
            "relative_path": relative_path,
            "manifest_sha256": manifest_sha256,
            "file_count": file_count,
            "total_bytes": total_bytes,
        }
        with self.connect() as connection:
            inserted = connection.execute(
                """
                INSERT INTO hf_revisions(
                    repo_id, commit_sha, requested_revision, relative_path,
                    manifest_sha256, file_count, total_bytes, imported_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo_id, commit_sha) DO NOTHING
                """,
                (
                    repo_id,
                    commit_sha,
                    requested_revision,
                    relative_path,
                    manifest_sha256,
                    file_count,
                    total_bytes,
                    utc_now(),
                ),
            ).rowcount
            if inserted == 0:
                existing = connection.execute(
                    """
                    SELECT relative_path, manifest_sha256, file_count, total_bytes
                    FROM hf_revisions
                    WHERE repo_id = ? AND commit_sha = ?
                    """,
                    (repo_id, commit_sha),
                ).fetchone()
                if existing is None or any(
                    existing[key] != value for key, value in record.items()
                ):
                    raise ImmutableRevisionConflictError(
                        "immutable Hugging Face revision metadata conflicts"
                    )

    def claim_hf_current(
        self,
        *,
        repo_id: str,
        commit_sha: str,
        generation: int,
        job_id: str,
        worker_id: str,
    ) -> bool:
        with self.connect() as connection:
            updated = connection.execute(
                """
                UPDATE hf_sources
                SET current_commit_sha = ?, current_generation = ?,
                    pointer_confirmed = 0, updated_at = ?
                WHERE repo_id = ? AND desired_commit_sha = ? AND generation = ?
                  AND EXISTS (
                    SELECT 1 FROM jobs
                    WHERE id = ? AND status = 'running' AND worker_id = ?
                      AND lease_expires_at >= ?
                  )
                """,
                (
                    commit_sha,
                    generation,
                    utc_now(),
                    repo_id,
                    commit_sha,
                    generation,
                    job_id,
                    worker_id,
                    utc_now(),
                ),
            ).rowcount
        return updated == 1

    def confirm_hf_current(
        self,
        *,
        repo_id: str,
        commit_sha: str,
        generation: int,
        job_id: str,
        worker_id: str,
    ) -> bool:
        with self.connect() as connection:
            updated = connection.execute(
                """
                UPDATE hf_sources SET pointer_confirmed = 1, updated_at = ?
                WHERE repo_id = ? AND current_commit_sha = ?
                  AND current_generation = ?
                  AND desired_commit_sha = ? AND generation = ?
                  AND EXISTS (
                    SELECT 1 FROM jobs
                    WHERE id = ? AND status = 'running' AND worker_id = ?
                      AND lease_expires_at >= ?
                  )
                """,
                (
                    utc_now(),
                    repo_id,
                    commit_sha,
                    generation,
                    commit_sha,
                    generation,
                    job_id,
                    worker_id,
                    utc_now(),
                ),
            ).rowcount
        return updated == 1

    def list_hf_sources(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT s.repo_id, s.desired_commit_sha, s.generation,
                       s.current_commit_sha, s.current_generation,
                       s.pointer_confirmed, s.updated_at,
                       r.relative_path, r.manifest_sha256, r.file_count,
                       r.total_bytes, r.imported_at
                FROM hf_sources AS s
                LEFT JOIN hf_revisions AS r
                  ON r.repo_id = s.repo_id
                 AND r.commit_sha = s.current_commit_sha
                ORDER BY s.repo_id
                """
            ).fetchall()
        return [
            {**dict(row), "pointer_confirmed": bool(row["pointer_confirmed"])}
            for row in rows
        ]

    def succeed_job(
        self,
        job_id: str,
        result: dict[str, Any],
        *,
        worker_id: str | None = None,
    ) -> dict[str, Any]:
        return self._finish_job(
            job_id,
            from_statuses=("running",),
            status="succeeded",
            result=result,
            worker_id=worker_id,
        )

    def fail_job(
        self,
        job_id: str,
        code: str,
        message: str,
        *,
        worker_id: str | None = None,
    ) -> dict[str, Any]:
        return self._finish_job(
            job_id,
            from_statuses=("running",),
            status="failed",
            error_code=code,
            error_message=message,
            worker_id=worker_id,
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

    def begin_dataset_scan(self, storage_area: str) -> int:
        if storage_area not in {"raw", "derived"}:
            raise ValueError("unsupported dataset storage area")
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO dataset_scan_generations(storage_area, generation, started_at)
                VALUES (?, 1, ?)
                ON CONFLICT(storage_area) DO UPDATE SET
                    generation = dataset_scan_generations.generation + 1,
                    started_at = excluded.started_at
                """,
                (storage_area, now),
            )
            generation = connection.execute(
                """
                SELECT generation FROM dataset_scan_generations
                WHERE storage_area = ?
                """,
                (storage_area,),
            ).fetchone()[0]
        return generation

    def synchronize_datasets(
        self,
        *,
        storage_area: str,
        records: list[dict[str, Any]],
        scan_generation: int,
    ) -> dict[str, int]:
        """Atomically merge one complete storage-area scan into the registry."""

        discovered = 0
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current_generation_row = connection.execute(
                """
                SELECT generation FROM dataset_scan_generations
                WHERE storage_area = ?
                """,
                (storage_area,),
            ).fetchone()
            if (
                current_generation_row is None
                or current_generation_row["generation"] != scan_generation
            ):
                return {"discovered": 0, "missing": 0, "stale": 1}
            for record in records:
                if record["storage_area"] != storage_area:
                    raise ValueError("dataset record belongs to another storage area")
                dataset_id = str(uuid.uuid4())
                connection.execute(
                    """
                    INSERT INTO datasets(
                        id, storage_area, relative_path, name, codebase_version,
                        readiness, robot_type, total_episodes, total_frames,
                        total_tasks, fps, fingerprint, info_mtime_ns, info_size,
                        scan_error, first_seen_at, last_seen_at, missing_since,
                        last_scan_generation
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                    ON CONFLICT(storage_area, relative_path) DO UPDATE SET
                        name = excluded.name,
                        codebase_version = excluded.codebase_version,
                        readiness = excluded.readiness,
                        robot_type = excluded.robot_type,
                        total_episodes = excluded.total_episodes,
                        total_frames = excluded.total_frames,
                        total_tasks = excluded.total_tasks,
                        fps = excluded.fps,
                        fingerprint = excluded.fingerprint,
                        info_mtime_ns = excluded.info_mtime_ns,
                        info_size = excluded.info_size,
                        scan_error = excluded.scan_error,
                        last_seen_at = excluded.last_seen_at,
                        last_scan_generation = excluded.last_scan_generation,
                        missing_since = NULL
                    WHERE excluded.last_scan_generation >= datasets.last_scan_generation
                    """,
                    (
                        dataset_id,
                        storage_area,
                        record["relative_path"],
                        record["name"],
                        record["codebase_version"],
                        record["readiness"],
                        record["robot_type"],
                        record["total_episodes"],
                        record["total_frames"],
                        record["total_tasks"],
                        record["fps"],
                        record["fingerprint"],
                        record["info_mtime_ns"],
                        record["info_size"],
                        record["scan_error"],
                        now,
                        now,
                        scan_generation,
                    ),
                )
                discovered += 1

            missing = connection.execute(
                """
                UPDATE datasets
                SET missing_since = ?, last_scan_generation = ?
                WHERE storage_area = ?
                  AND missing_since IS NULL
                  AND last_scan_generation < ?
                """,
                (now, scan_generation, storage_area, scan_generation),
            ).rowcount

        return {"discovered": discovered, "missing": missing}

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT id, storage_area, relative_path, name, codebase_version,
                       readiness, robot_type, total_episodes, total_frames,
                       total_tasks, fps, fingerprint, scan_error, first_seen_at,
                       last_seen_at, missing_since
                FROM datasets WHERE id = ?
                """,
                (dataset_id,),
            ).fetchone()
        if row is None:
            raise DatasetNotFoundError(dataset_id)
        return self._decode_dataset(row)

    def list_datasets(
        self,
        *,
        storage_area: str | None = None,
        readiness: str | None = None,
        include_missing: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if storage_area:
            clauses.append("storage_area = ?")
            parameters.append(storage_area)
        if readiness:
            clauses.append("readiness = ?")
            parameters.append(readiness)
        if not include_missing:
            clauses.append("missing_since IS NULL")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(limit)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT id, storage_area, relative_path, name, codebase_version,
                       readiness, robot_type, total_episodes, total_frames,
                       total_tasks, fps, fingerprint, scan_error, first_seen_at,
                       last_seen_at, missing_since
                FROM datasets
                {where}
                ORDER BY name COLLATE NOCASE, storage_area, relative_path
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [self._decode_dataset(row) for row in rows]

    @staticmethod
    def _curation_source(
        connection: sqlite3.Connection,
        *,
        dataset_id: str,
        profile_id: str,
    ) -> sqlite3.Row:
        profile = connection.execute(
            "SELECT id FROM profiles WHERE id = ? AND archived_at IS NULL",
            (profile_id,),
        ).fetchone()
        if profile is None:
            raise ProfileNotFoundError(profile_id)

        dataset = connection.execute(
            """
            SELECT id, fingerprint, total_episodes, readiness, missing_since
            FROM datasets WHERE id = ?
            """,
            (dataset_id,),
        ).fetchone()
        if dataset is None:
            raise DatasetNotFoundError(dataset_id)
        if (
            dataset["missing_since"] is not None
            or dataset["readiness"] != "ready"
            or dataset["total_episodes"] is None
        ):
            raise DatasetNotReadyError(dataset_id)
        return dataset

    @staticmethod
    def _episode_flags_record(
        connection: sqlite3.Connection,
        *,
        dataset_id: str,
        dataset_fingerprint: str,
        profile_id: str,
    ) -> dict[str, Any]:
        flag_set = connection.execute(
            """
            SELECT revision, updated_at FROM flag_sets
            WHERE dataset_id = ? AND dataset_fingerprint = ? AND profile_id = ?
            """,
            (dataset_id, dataset_fingerprint, profile_id),
        ).fetchone()
        indices = [
            row["episode_index"]
            for row in connection.execute(
                """
                SELECT episode_index FROM episode_flags
                WHERE dataset_id = ? AND dataset_fingerprint = ? AND profile_id = ?
                ORDER BY episode_index
                """,
                (dataset_id, dataset_fingerprint, profile_id),
            )
        ]
        return {
            "dataset_id": dataset_id,
            "dataset_fingerprint": dataset_fingerprint,
            "profile_id": profile_id,
            "revision": flag_set["revision"] if flag_set else 0,
            "episode_indices": indices,
            "updated_at": flag_set["updated_at"] if flag_set else None,
        }

    def get_episode_flags(self, *, dataset_id: str, profile_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            connection.execute("BEGIN")
            dataset = self._curation_source(
                connection, dataset_id=dataset_id, profile_id=profile_id
            )
            return self._episode_flags_record(
                connection,
                dataset_id=dataset_id,
                dataset_fingerprint=dataset["fingerprint"],
                profile_id=profile_id,
            )

    def update_episode_flags(
        self,
        *,
        dataset_id: str,
        profile_id: str,
        expected_revision: int,
        changes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            dataset = self._curation_source(
                connection, dataset_id=dataset_id, profile_id=profile_id
            )
            total_episodes = dataset["total_episodes"]
            for change in changes:
                if change["episode_index"] >= total_episodes:
                    raise ValueError("episode index is outside the dataset")

            fingerprint = dataset["fingerprint"]
            flag_set = connection.execute(
                """
                SELECT revision FROM flag_sets
                WHERE dataset_id = ? AND dataset_fingerprint = ? AND profile_id = ?
                """,
                (dataset_id, fingerprint, profile_id),
            ).fetchone()
            current_revision = flag_set["revision"] if flag_set else 0
            if current_revision != expected_revision:
                raise FlagRevisionConflictError(str(current_revision))

            existing = {
                row["episode_index"]
                for row in connection.execute(
                    """
                    SELECT episode_index FROM episode_flags
                    WHERE dataset_id = ? AND dataset_fingerprint = ? AND profile_id = ?
                    """,
                    (dataset_id, fingerprint, profile_id),
                )
            }
            effective = [
                change
                for change in changes
                if (change["episode_index"] in existing) != change["flagged"]
            ]
            if effective:
                connection.execute(
                    """
                    INSERT INTO flag_sets(
                        dataset_id, dataset_fingerprint, profile_id,
                        revision, updated_at
                    ) VALUES (?, ?, ?, 0, ?)
                    ON CONFLICT(dataset_id, dataset_fingerprint, profile_id)
                    DO NOTHING
                    """,
                    (dataset_id, fingerprint, profile_id, now),
                )
                for change in effective:
                    key = (
                        dataset_id,
                        fingerprint,
                        profile_id,
                        change["episode_index"],
                    )
                    if change["flagged"]:
                        connection.execute(
                            """
                            INSERT INTO episode_flags(
                                dataset_id, dataset_fingerprint, profile_id,
                                episode_index, created_at
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            (*key, now),
                        )
                    else:
                        connection.execute(
                            """
                            DELETE FROM episode_flags
                            WHERE dataset_id = ? AND dataset_fingerprint = ?
                              AND profile_id = ? AND episode_index = ?
                            """,
                            key,
                        )
                connection.execute(
                    """
                    UPDATE flag_sets SET revision = revision + 1, updated_at = ?
                    WHERE dataset_id = ? AND dataset_fingerprint = ?
                      AND profile_id = ?
                    """,
                    (now, dataset_id, fingerprint, profile_id),
                )

            return self._episode_flags_record(
                connection,
                dataset_id=dataset_id,
                dataset_fingerprint=fingerprint,
                profile_id=profile_id,
            )

    def create_curation_recipe(
        self,
        *,
        dataset_id: str,
        profile_id: str,
        name: str,
        selection_mode: str,
    ) -> dict[str, Any]:
        recipe_id = str(uuid.uuid4())
        now = utc_now()
        try:
            with self.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                dataset = self._curation_source(
                    connection, dataset_id=dataset_id, profile_id=profile_id
                )
                connection.execute(
                    """
                    INSERT INTO curation_recipes(
                        id, dataset_id, dataset_fingerprint, profile_id,
                        name, name_key, selection_mode, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        recipe_id,
                        dataset_id,
                        dataset["fingerprint"],
                        profile_id,
                        name,
                        _recipe_name_key(name),
                        selection_mode,
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            if "curation_recipes.dataset_id" in str(exc):
                raise DuplicateRecipeNameError(name) from exc
            raise
        return self.get_curation_recipe(recipe_id, profile_id=profile_id)

    def list_curation_recipes(
        self,
        *,
        dataset_id: str,
        profile_id: str,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        with self.connect() as connection:
            connection.execute("BEGIN")
            dataset = self._curation_source(
                connection, dataset_id=dataset_id, profile_id=profile_id
            )
            archived_clause = "" if include_archived else "AND archived_at IS NULL"
            rows = connection.execute(
                f"""
                SELECT id, dataset_id, dataset_fingerprint, profile_id, name,
                       selection_mode, created_at, updated_at, archived_at
                FROM curation_recipes
                WHERE dataset_id = ? AND dataset_fingerprint = ?
                  AND profile_id = ? {archived_clause}
                ORDER BY updated_at DESC, name_key
                """,
                (dataset_id, dataset["fingerprint"], profile_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_curation_recipe(self, recipe_id: str, *, profile_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT id, dataset_id, dataset_fingerprint, profile_id, name,
                       selection_mode, created_at, updated_at, archived_at
                FROM curation_recipes WHERE id = ? AND profile_id = ?
                """,
                (recipe_id, profile_id),
            ).fetchone()
        if row is None:
            raise RecipeNotFoundError(recipe_id)
        return dict(row)

    def update_curation_recipe(
        self,
        recipe_id: str,
        *,
        profile_id: str,
        name: str | None,
        selection_mode: str | None,
        archived: bool | None,
    ) -> dict[str, Any]:
        now = utc_now()
        try:
            with self.connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                recipe = connection.execute(
                    "SELECT * FROM curation_recipes WHERE id = ? AND profile_id = ?",
                    (recipe_id, profile_id),
                ).fetchone()
                if recipe is None:
                    raise RecipeNotFoundError(recipe_id)
                dataset = self._curation_source(
                    connection,
                    dataset_id=recipe["dataset_id"],
                    profile_id=profile_id,
                )
                if dataset["fingerprint"] != recipe["dataset_fingerprint"]:
                    raise RecipeRevisionMismatchError(recipe_id)
                next_name = name if name is not None else recipe["name"]
                next_selection = (
                    selection_mode
                    if selection_mode is not None
                    else recipe["selection_mode"]
                )
                next_archived_at = recipe["archived_at"]
                if archived is True:
                    next_archived_at = now
                elif archived is False:
                    next_archived_at = None
                connection.execute(
                    """
                    UPDATE curation_recipes
                    SET name = ?, name_key = ?, selection_mode = ?,
                        archived_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        next_name,
                        _recipe_name_key(next_name),
                        next_selection,
                        next_archived_at,
                        now,
                        recipe_id,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            if "curation_recipes.dataset_id" in str(exc):
                raise DuplicateRecipeNameError(name or "") from exc
            raise
        return self.get_curation_recipe(recipe_id, profile_id=profile_id)

    def snapshot_curation_recipe(
        self, recipe_id: str, *, profile_id: str
    ) -> dict[str, Any]:
        snapshot_id = str(uuid.uuid4())
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            recipe = connection.execute(
                """
                SELECT * FROM curation_recipes
                WHERE id = ? AND profile_id = ? AND archived_at IS NULL
                """,
                (recipe_id, profile_id),
            ).fetchone()
            if recipe is None:
                raise RecipeNotFoundError(recipe_id)
            dataset = self._curation_source(
                connection,
                dataset_id=recipe["dataset_id"],
                profile_id=profile_id,
            )
            if dataset["fingerprint"] != recipe["dataset_fingerprint"]:
                raise RecipeRevisionMismatchError(recipe_id)
            flags = self._episode_flags_record(
                connection,
                dataset_id=recipe["dataset_id"],
                dataset_fingerprint=recipe["dataset_fingerprint"],
                profile_id=profile_id,
            )
            flagged = flags["episode_indices"]
            flagged_set = set(flagged)
            if recipe["selection_mode"] == "all":
                selected = list(range(dataset["total_episodes"]))
            elif recipe["selection_mode"] == "flagged":
                selected = flagged
            else:
                selected = [
                    index
                    for index in range(dataset["total_episodes"])
                    if index not in flagged_set
                ]
            connection.execute(
                """
                INSERT INTO curation_recipe_snapshots(
                    id, recipe_id, dataset_id, dataset_fingerprint, profile_id,
                    recipe_name, selection_mode, flag_revision,
                    flagged_episode_indices_json, selected_episode_indices_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    recipe_id,
                    recipe["dataset_id"],
                    recipe["dataset_fingerprint"],
                    profile_id,
                    recipe["name"],
                    recipe["selection_mode"],
                    flags["revision"],
                    _json_dump(flagged),
                    _json_dump(selected),
                    now,
                ),
            )
        return {
            "id": snapshot_id,
            "recipe_id": recipe_id,
            "dataset_id": recipe["dataset_id"],
            "dataset_fingerprint": recipe["dataset_fingerprint"],
            "profile_id": profile_id,
            "recipe_name": recipe["name"],
            "selection_mode": recipe["selection_mode"],
            "flag_revision": flags["revision"],
            "flagged_episode_indices": flagged,
            "selected_episode_indices": selected,
            "created_at": now,
        }

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

    @staticmethod
    def _resolve_hf_idempotent_job(
        existing: dict[str, Any],
        *,
        repo_id: str,
        dataset_name: str,
        requested_revision: str,
        commit_sha: str,
        expected_file_count: int,
        expected_total_bytes: int | None,
    ) -> tuple[dict[str, Any], bool]:
        expected = {
            "repo_id": repo_id,
            "dataset_name": dataset_name,
            "requested_revision": requested_revision,
            "commit_sha": commit_sha,
            "expected_file_count": expected_file_count,
            "expected_total_bytes": expected_total_bytes,
        }
        if existing["kind"] != "hf.import" or any(
            existing["payload"].get(key) != value for key, value in expected.items()
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
        worker_id: str | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        placeholders = ",".join("?" for _ in from_statuses)
        worker_clause = " AND worker_id = ?" if worker_id is not None else ""
        with self.connect() as connection:
            updated = connection.execute(
                f"""
                UPDATE jobs
                SET status = ?, result_json = ?, error_code = ?, error_message = ?,
                    finished_at = ?
                WHERE id = ? AND status IN ({placeholders}){worker_clause}
                """,
                (
                    status,
                    _json_dump(result) if result is not None else None,
                    error_code,
                    error_message,
                    now,
                    job_id,
                    *from_statuses,
                    *((worker_id,) if worker_id is not None else ()),
                ),
            ).rowcount
            if updated != 1:
                if worker_id is not None:
                    raise JobLeaseLostError(job_id)
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

    @staticmethod
    def _decode_dataset(row: sqlite3.Row) -> dict[str, Any]:
        dataset = dict(row)
        dataset["available"] = dataset.pop("missing_since") is None
        return dataset
