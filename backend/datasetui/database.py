from __future__ import annotations

import json
import math
import random
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


class JobOwnershipError(PermissionError):
    pass


class JobCancellationConflictError(RuntimeError):
    def __init__(
        self, job_id: str, job_status: str, *, reason: str = "terminal"
    ) -> None:
        super().__init__(job_id)
        self.job_status = job_status
        self.reason = reason


class ValidationRunNotFoundError(LookupError):
    pass


class ValidationRunActiveError(RuntimeError):
    pass


class DatasetNotFoundError(LookupError):
    pass


class DatasetNameConflictError(RuntimeError):
    pass


class DatasetTrashConflictError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class DatasetNotReadyError(RuntimeError):
    pass


class FlagRevisionConflictError(RuntimeError):
    pass


class AnnotationRevisionConflictError(RuntimeError):
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
    (
        5,
        (
            "ALTER TABLE curation_recipes ADD COLUMN operation TEXT NOT NULL DEFAULT 'subset'",
            'ALTER TABLE curation_recipes ADD COLUMN trim_config_json TEXT NOT NULL DEFAULT \'{"enabled":false,"threshold":0.02,"hold_time_s":0.5,"margin_s":1.0,"dimensions":[],"episode_overrides":{}}\'',
            "ALTER TABLE curation_recipe_snapshots ADD COLUMN operation TEXT NOT NULL DEFAULT 'subset'",
            'ALTER TABLE curation_recipe_snapshots ADD COLUMN trim_config_json TEXT NOT NULL DEFAULT \'{"enabled":false,"threshold":0.02,"hold_time_s":0.5,"margin_s":1.0,"dimensions":[],"episode_overrides":{}}\'',
            """
            CREATE TABLE curation_runs (
                job_id TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
                snapshot_id TEXT NOT NULL REFERENCES curation_recipe_snapshots(id),
                output_name TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            "CREATE INDEX curation_runs_snapshot_idx ON curation_runs(snapshot_id, created_at DESC)",
        ),
    ),
    (
        6,
        (
            "ALTER TABLE curation_recipes ADD COLUMN include_annotations INTEGER NOT NULL DEFAULT 0 CHECK (include_annotations IN (0, 1))",
            "ALTER TABLE curation_recipe_snapshots ADD COLUMN include_annotations INTEGER NOT NULL DEFAULT 0 CHECK (include_annotations IN (0, 1))",
            """
            CREATE TABLE episode_annotations (
                dataset_id TEXT NOT NULL REFERENCES datasets(id),
                dataset_fingerprint TEXT NOT NULL,
                profile_id TEXT NOT NULL REFERENCES profiles(id),
                episode_index INTEGER NOT NULL CHECK (episode_index >= 0),
                revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
                task_override TEXT,
                atoms_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (
                    dataset_id, dataset_fingerprint, profile_id, episode_index
                )
            )
            """,
            """
            CREATE TABLE curation_annotation_snapshots (
                snapshot_id TEXT NOT NULL REFERENCES curation_recipe_snapshots(id)
                    ON DELETE CASCADE,
                episode_index INTEGER NOT NULL CHECK (episode_index >= 0),
                annotation_revision INTEGER NOT NULL CHECK (annotation_revision >= 0),
                task_override TEXT,
                atoms_json TEXT NOT NULL,
                PRIMARY KEY (snapshot_id, episode_index)
            )
            """,
            "CREATE INDEX episode_annotations_lookup_idx ON episode_annotations(dataset_id, dataset_fingerprint, profile_id, updated_at DESC)",
        ),
    ),
    (
        7,
        (
            'ALTER TABLE curation_recipes ADD COLUMN relative_action_json TEXT NOT NULL DEFAULT \'{"enabled":false,"dimensions":[]}\'',
            'ALTER TABLE curation_recipe_snapshots ADD COLUMN relative_action_json TEXT NOT NULL DEFAULT \'{"enabled":false,"dimensions":[]}\'',
        ),
    ),
    (
        8,
        (
            """
            CREATE TABLE validation_runs (
                job_id TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
                dataset_id TEXT NOT NULL REFERENCES datasets(id),
                dataset_fingerprint TEXT NOT NULL,
                mode TEXT NOT NULL CHECK (mode IN ('quick', 'full', 'export_gate')),
                created_at TEXT NOT NULL
            )
            """,
            "CREATE INDEX validation_runs_dataset_idx ON validation_runs(dataset_id, created_at DESC)",
        ),
    ),
    (
        9,
        ("ALTER TABLE validation_runs ADD COLUMN progress_json TEXT",),
    ),
    (
        10,
        (
            'ALTER TABLE curation_recipes ADD COLUMN split_config_json TEXT NOT NULL DEFAULT \'{"method":"flagged","eval_percent":20.0,"seed":0}\'',
            'ALTER TABLE curation_recipe_snapshots ADD COLUMN split_config_json TEXT NOT NULL DEFAULT \'{"method":"flagged","eval_percent":20.0,"seed":0}\'',
            "ALTER TABLE curation_recipe_snapshots ADD COLUMN eval_episode_indices_json TEXT NOT NULL DEFAULT '[]'",
        ),
    ),
    (
        11,
        ("ALTER TABLE validation_runs ADD COLUMN deleted_at TEXT",),
    ),
    (
        12,
        ("ALTER TABLE datasets ADD COLUMN display_name TEXT",),
    ),
    (
        13,
        ("ALTER TABLE jobs ADD COLUMN progress_json TEXT",),
    ),
    (
        14,
        (
            "ALTER TABLE jobs ADD COLUMN cancellation_requested_at TEXT",
            "ALTER TABLE jobs ADD COLUMN cancellation_guarded_at TEXT",
        ),
    ),
    (
        15,
        (
            """
            CREATE TABLE dataset_trash (
                dataset_id TEXT PRIMARY KEY REFERENCES datasets(id),
                storage_area TEXT NOT NULL CHECK (storage_area IN ('raw', 'derived')),
                original_relative_path TEXT NOT NULL,
                trash_relative_path TEXT NOT NULL,
                source_device INTEGER,
                source_inode INTEGER,
                state TEXT NOT NULL CHECK (
                    state IN ('moving', 'trashed', 'restoring', 'recovery_required')
                ),
                requested_by_profile_id TEXT NOT NULL REFERENCES profiles(id),
                requested_at TEXT NOT NULL,
                trashed_at TEXT,
                updated_at TEXT NOT NULL
            )
            """,
            "CREATE INDEX dataset_trash_state_idx ON dataset_trash(state, updated_at)",
        ),
    ),
    (
        16,
        (
            "ALTER TABLE jobs ADD COLUMN validation_job_id TEXT REFERENCES jobs(id)",
            "ALTER TABLE jobs ADD COLUMN validation_ready INTEGER NOT NULL DEFAULT 1",
            "CREATE INDEX jobs_validation_idx ON jobs(validation_job_id, status)",
            """CREATE TABLE dataset_purge_reservations (
                dataset_id TEXT PRIMARY KEY REFERENCES datasets(id),
                job_id TEXT NOT NULL REFERENCES jobs(id),
                state TEXT NOT NULL CHECK (state IN ('reserved','purging','purged','failed')),
                updated_at TEXT NOT NULL
            )""",
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


def _random_eval_episode_indices(
    episode_indices: list[int], *, eval_percent: float, seed: int
) -> list[int]:
    """Choose round-half-up(N * percent / 100) episodes reproducibly."""
    count = math.floor(len(episode_indices) * eval_percent / 100 + 0.5)
    return sorted(random.Random(seed).sample(episode_indices, count))


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
                connection.execute("BEGIN IMMEDIATE")
                self._assert_job_creation_allowed(connection, payload, kind=kind)
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

    def create_delivery_with_validation(
        self,
        *,
        kind: str,
        profile_id: str,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> tuple[dict[str, Any], bool]:
        self.get_profile(profile_id)
        existing = self._job_for_idempotency(profile_id, idempotency_key)
        if existing is not None:
            return self._resolve_idempotent_job(
                existing, kind=kind, queue_name="io", payload=payload
            )
        now = utc_now()
        job_id = str(uuid.uuid4())
        check_payload = {
            key: payload[key]
            for key in ("dataset_id", "fingerprint", "storage_area", "relative_path")
        }
        check_payload["mode"] = "export_gate"
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT id FROM jobs WHERE profile_id = ? AND idempotency_key = ?",
                (profile_id, idempotency_key),
            ).fetchone()
            if existing:
                existing_id = existing["id"]
            else:
                existing_id = None
                self._assert_job_creation_allowed(connection, payload, kind=kind)
                check = connection.execute(
                    """SELECT id FROM jobs WHERE kind = 'datasets.delivery_preflight'
                    AND profile_id = ? AND status IN ('queued','running') AND cancellation_requested_at IS NULL
                    AND payload_json = ? ORDER BY created_at LIMIT 1""",
                    (profile_id, _json_dump(check_payload)),
                ).fetchone()
                check_id = check["id"] if check else str(uuid.uuid4())
                if check is None:
                    connection.execute(
                        """INSERT INTO jobs(id,kind,queue_name,status,profile_id,payload_json,idempotency_key,created_at)
                        VALUES (?,'datasets.delivery_preflight','cpu','queued',?,?,?,?)""",
                        (
                            check_id,
                            profile_id,
                            _json_dump(check_payload),
                            f"delivery-check-{check_id}",
                            now,
                        ),
                    )
                    connection.execute(
                        "INSERT INTO validation_runs(job_id,dataset_id,dataset_fingerprint,mode,created_at) VALUES (?,?,?,'export_gate',?)",
                        (check_id, payload["dataset_id"], payload["fingerprint"], now),
                    )
                    self._append_event(
                        connection,
                        check_id,
                        "queued",
                        {"kind": "datasets.delivery_preflight", "queue": "cpu"},
                        now=now,
                    )
                ready = -1 if kind == "datasets.copy_pc_password" else 0
                connection.execute(
                    """INSERT INTO jobs(id,kind,queue_name,status,profile_id,payload_json,idempotency_key,created_at,validation_job_id,validation_ready)
                    VALUES (?,?,'io','queued',?,?,?,?,?,?)""",
                    (
                        job_id,
                        kind,
                        profile_id,
                        _json_dump(payload),
                        idempotency_key,
                        now,
                        check_id,
                        ready,
                    ),
                )
                self._append_event(
                    connection,
                    job_id,
                    "queued",
                    {"kind": kind, "queue": "io", "validation_job_id": check_id},
                    now=now,
                )
        if existing_id:
            return self._resolve_idempotent_job(
                self.get_job(existing_id), kind=kind, queue_name="io", payload=payload
            )
        return self.get_job(job_id), True

    def release_delivery_validation(self, job_id: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE jobs SET validation_ready = 1 WHERE id = ? AND status = 'queued' AND validation_ready = 0",
                (job_id,),
            )

    def fail_waiting_job(self, job_id: str, code: str, message: str) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                "UPDATE jobs SET status='failed',error_code=?,error_message=?,finished_at=? WHERE id=? AND status='queued'",
                (code, message, now, job_id),
            ).rowcount
            if changed:
                self._append_event(
                    connection, job_id, "failed", {"error_code": code}, now=now
                )

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
                self._assert_job_creation_allowed(
                    connection, {"repo_id": repo_id}, kind="hf.import"
                )
                target_relative_path = (
                    f"hf/rainbowrobotics/{dataset_name}/revisions/{commit_sha}"
                )
                if (
                    connection.execute(
                        """
                    SELECT 1
                    FROM dataset_trash t
                    JOIN datasets d ON d.id = t.dataset_id
                    WHERE d.storage_area = 'raw' AND d.relative_path = ?
                    LIMIT 1
                    """,
                        (target_relative_path,),
                    ).fetchone()
                    is not None
                ):
                    raise DatasetTrashConflictError("dataset_in_use")
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
                  AND (error_code IS NULL OR error_code != 'queue_unavailable')
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
                    attempt = attempt + 1, progress_json = NULL
                WHERE id = ? AND status = 'queued' AND validation_ready = 1
                """,
                (now, worker_id, now, lease_expires_at, job_id),
            ).rowcount
            if updated != 1:
                return None
            self._append_event(connection, job_id, "running", {}, now=now)
        return self.get_job(job_id)

    def request_job_cancellation(
        self, job_id: str, *, profile_id: str
    ) -> dict[str, Any]:
        """Cancel queued work or persist a cooperative running cancellation."""

        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT profile_id, status, cancellation_requested_at,
                       cancellation_guarded_at
                FROM jobs WHERE id = ?
                """,
                (job_id,),
            ).fetchone()
            if row is None:
                raise JobNotFoundError(job_id)
            if row["profile_id"] != profile_id:
                raise JobOwnershipError(job_id)
            if row["status"] == "cancelled":
                pass
            elif row["status"] == "queued":
                connection.execute(
                    """
                    UPDATE jobs
                    SET status = 'cancelled', finished_at = ?,
                        error_code = NULL, error_message = NULL,
                        cancellation_requested_at = ?
                    WHERE id = ? AND status = 'queued'
                    """,
                    (now, now, job_id),
                )
                self._append_event(
                    connection,
                    job_id,
                    "cancelled",
                    {"reason": "cancelled_by_user"},
                    now=now,
                )
            elif row["status"] == "running":
                if row["cancellation_guarded_at"] is not None:
                    raise JobCancellationConflictError(
                        job_id, "running", reason="finalizing"
                    )
                if row["cancellation_requested_at"] is None:
                    connection.execute(
                        """
                        UPDATE jobs SET cancellation_requested_at = ?
                        WHERE id = ? AND status = 'running'
                          AND cancellation_guarded_at IS NULL
                          AND cancellation_requested_at IS NULL
                        """,
                        (now, job_id),
                    )
                    self._append_event(
                        connection,
                        job_id,
                        "cancellation_requested",
                        {"reason": "cancelled_by_user"},
                        now=now,
                    )
            else:
                raise JobCancellationConflictError(job_id, row["status"])
        return self.get_job(job_id)

    def cancel_queued_job(self, job_id: str, *, profile_id: str) -> dict[str, Any]:
        """Compatibility wrapper for callers using the original endpoint method."""

        return self.request_job_cancellation(job_id, profile_id=profile_id)

    def is_job_cancellation_requested(self, job_id: str, *, worker_id: str) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT cancellation_requested_at FROM jobs
                WHERE id = ? AND status = 'running' AND worker_id = ?
                """,
                (job_id, worker_id),
            ).fetchone()
        return row is not None and row["cancellation_requested_at"] is not None

    def begin_job_finalization(self, job_id: str, *, worker_id: str) -> None:
        """Atomically close cancellation before an irreversible publication."""

        from datasetui.job_cancellation import JobCancellationRequested

        now = utc_now()
        with self.connect() as connection:
            updated = connection.execute(
                """
                UPDATE jobs SET cancellation_guarded_at = ?
                WHERE id = ? AND status = 'running' AND worker_id = ?
                  AND lease_expires_at >= ?
                  AND cancellation_requested_at IS NULL
                  AND cancellation_guarded_at IS NULL
                """,
                (now, job_id, worker_id, now),
            ).rowcount
            if updated == 1:
                self._append_event(connection, job_id, "finalizing", {}, now=now)
                return
            row = connection.execute(
                """
                SELECT status, worker_id, lease_expires_at,
                       cancellation_requested_at, cancellation_guarded_at,
                       validation_job_id, validation_ready,
                       (SELECT name FROM profiles WHERE profiles.id = jobs.profile_id) AS profile_name
                FROM jobs WHERE id = ?
                """,
                (job_id,),
            ).fetchone()
            if (
                row is not None
                and row["status"] == "running"
                and row["worker_id"] == worker_id
                and row["lease_expires_at"] is not None
                and row["lease_expires_at"] >= now
                and row["cancellation_requested_at"] is None
                and row["cancellation_guarded_at"] is not None
            ):
                return
            if row is not None and row["cancellation_requested_at"] is not None:
                raise JobCancellationRequested(job_id)
        raise JobLeaseLostError(job_id)

    def complete_job_cancellation(
        self, job_id: str, *, worker_id: str
    ) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as connection:
            updated = connection.execute(
                """
                UPDATE jobs
                SET status = 'cancelled', finished_at = ?, result_json = NULL,
                    error_code = NULL, error_message = NULL,
                    worker_id = NULL, heartbeat_at = NULL, lease_expires_at = NULL
                WHERE id = ? AND status = 'running' AND worker_id = ?
                  AND cancellation_requested_at IS NOT NULL
                  AND cancellation_guarded_at IS NULL
                """,
                (now, job_id, worker_id),
            ).rowcount
            if updated != 1:
                raise JobLeaseLostError(job_id)
            self._append_event(
                connection,
                job_id,
                "cancelled",
                {"reason": "cancelled_by_user"},
                now=now,
            )
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
                SELECT id, kind, cancellation_requested_at,
                       cancellation_guarded_at FROM jobs
                WHERE status = 'running'
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at < ?
                """,
                (now,),
            ).fetchall()
            for row in rows:
                if (
                    row["cancellation_requested_at"] is not None
                    and row["cancellation_guarded_at"] is None
                ):
                    updated = connection.execute(
                        """
                        UPDATE jobs
                        SET status = 'cancelled', finished_at = ?,
                            error_code = NULL, error_message = NULL,
                            worker_id = NULL, heartbeat_at = NULL,
                            lease_expires_at = NULL
                        WHERE id = ? AND status = 'running'
                          AND lease_expires_at < ?
                          AND cancellation_requested_at IS NOT NULL
                          AND cancellation_guarded_at IS NULL
                        """,
                        (now, row["id"], now),
                    ).rowcount
                    if updated == 1:
                        self._append_event(
                            connection,
                            row["id"],
                            "cancelled",
                            {"reason": "worker_lost_after_cancellation"},
                            now=now,
                        )
                    continue
                if row["cancellation_guarded_at"] is not None:
                    updated = connection.execute(
                        """
                        UPDATE jobs
                        SET status = 'interrupted',
                            error_code = 'finalization_outcome_uncertain',
                            error_message = 'Check the output before retrying',
                            finished_at = ?, worker_id = NULL,
                            heartbeat_at = NULL, lease_expires_at = NULL
                        WHERE id = ? AND status = 'running'
                          AND lease_expires_at < ?
                          AND cancellation_guarded_at IS NOT NULL
                        """,
                        (now, row["id"], now),
                    ).rowcount
                    if updated == 1:
                        self._append_event(
                            connection,
                            row["id"],
                            "interrupted",
                            {"error_code": "finalization_outcome_uncertain"},
                            now=now,
                        )
                    continue
                if row["kind"] == "datasets.validate":
                    connection.execute(
                        """UPDATE jobs SET status = 'failed',
                        error_code = 'validation_interrupted',
                        error_message = 'Validation worker stopped; retry manually after checking the cause',
                        finished_at = ?, worker_id = NULL, heartbeat_at = NULL,
                        lease_expires_at = NULL WHERE id = ?""",
                        (now, row["id"]),
                    )
                    self._append_event(
                        connection,
                        row["id"],
                        "failed",
                        {"error_code": "validation_interrupted"},
                        now=now,
                    )
                    continue
                if row["kind"] in {"datasets.upload_hf", "datasets.copy_pc_key"}:
                    connection.execute(
                        """UPDATE jobs SET status = 'interrupted',
                        error_code = 'external_outcome_uncertain',
                        error_message = 'Check the external destination before retrying',
                        finished_at = ?, worker_id = NULL, heartbeat_at = NULL,
                        lease_expires_at = NULL WHERE id = ?""",
                        (now, row["id"]),
                    )
                    self._append_event(
                        connection,
                        row["id"],
                        "interrupted",
                        {"error_code": "external_outcome_uncertain"},
                        now=now,
                    )
                    continue
                updated = connection.execute(
                    """
                    UPDATE jobs
                    SET status = 'queued', error_code = 'worker_lost',
                        error_message = 'The worker stopped responding', finished_at = NULL,
                        started_at = NULL, rq_job_id = NULL,
                        worker_id = NULL, heartbeat_at = NULL, lease_expires_at = NULL,
                        dispatch_generation = dispatch_generation + 1,
                        progress_json = NULL, cancellation_requested_at = NULL,
                        cancellation_guarded_at = NULL
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
        from datasetui.job_cancellation import JobCancellationRequested

        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT cancellation_requested_at FROM jobs
                WHERE id = ? AND status = 'running' AND worker_id = ?
                  AND lease_expires_at >= ?
                """,
                (job_id, worker_id, utc_now()),
            ).fetchone()
        if row is None:
            raise JobLeaseLostError(job_id)
        if row["cancellation_requested_at"] is not None:
            raise JobCancellationRequested(job_id)

    def update_job_progress(
        self, job_id: str, *, worker_id: str, progress: dict[str, Any]
    ) -> None:
        from datasetui.job_cancellation import JobCancellationRequested

        with self.connect() as connection:
            updated = connection.execute(
                """
                UPDATE jobs SET progress_json = ?
                WHERE id = ? AND status = 'running' AND worker_id = ?
                  AND lease_expires_at >= ?
                  AND cancellation_requested_at IS NULL
                """,
                (_json_dump(progress), job_id, worker_id, utc_now()),
            ).rowcount
            if updated != 1:
                row = connection.execute(
                    """
                    SELECT cancellation_requested_at FROM jobs
                    WHERE id = ? AND status = 'running' AND worker_id = ?
                    """,
                    (job_id, worker_id),
                ).fetchone()
                if row is not None and row["cancellation_requested_at"] is not None:
                    raise JobCancellationRequested(job_id)
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
                       result_json, progress_json, error_code, error_message, idempotency_key,
                       rq_job_id, created_at, enqueued_at, started_at, finished_at,
                       cancellation_requested_at, cancellation_guarded_at,
                       validation_job_id, validation_ready,
                       (SELECT name FROM profiles WHERE profiles.id = jobs.profile_id) AS profile_name
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
        kind: str | None = None,
        limit: int = 100,
        public: bool = False,
        active_only: bool = False,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if profile_id:
            clauses.append("profile_id = ?")
            parameters.append(profile_id)
        if status:
            clauses.append("status = ?")
            parameters.append(status)
        if kind:
            clauses.append("kind = ?")
            parameters.append(kind)
        if active_only:
            clauses.append("status IN ('queued', 'running')")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.extend((limit, offset))
        payload_column = (
            "json_remove(payload_json, '$.spec.background_base64') AS payload_json"
            if public
            else "payload_json"
        )
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT id, kind, queue_name, status, profile_id, {payload_column},
                       result_json, progress_json, error_code, error_message, idempotency_key,
                       rq_job_id, created_at, enqueued_at, started_at, finished_at,
                       cancellation_requested_at, cancellation_guarded_at,
                       validation_job_id, validation_ready,
                       (SELECT name FROM profiles WHERE profiles.id = jobs.profile_id) AS profile_name
                FROM jobs
                {where}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
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
            connection.execute("BEGIN IMMEDIATE")
            if (
                connection.execute(
                    "SELECT 1 FROM dataset_trash WHERE state IN ('moving', 'restoring') LIMIT 1"
                ).fetchone()
                is not None
            ):
                raise DatasetTrashConflictError("dataset_in_use")
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
                      AND NOT EXISTS (
                          SELECT 1 FROM dataset_trash t WHERE t.dataset_id = datasets.id
                      )
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
                  AND NOT EXISTS (
                      SELECT 1 FROM dataset_trash t WHERE t.dataset_id = datasets.id
                  )
                """,
                (now, scan_generation, storage_area, scan_generation),
            ).rowcount

        return {"discovered": discovered, "missing": missing}

    def prepare_dataset_trash(
        self,
        dataset_id: str,
        *,
        profile_id: str,
        expected_name: str,
        expected_fingerprint: str,
    ) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._assert_active_profile(connection, profile_id)
            dataset = connection.execute(
                """
                SELECT id, storage_area, relative_path, name, display_name, fingerprint
                FROM datasets WHERE id = ?
                """,
                (dataset_id,),
            ).fetchone()
            if dataset is None:
                raise DatasetNotFoundError(dataset_id)
            existing = connection.execute(
                "SELECT state FROM dataset_trash WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchone()
            if existing is not None:
                raise DatasetTrashConflictError("dataset_already_trashed")
            current_name = dataset["display_name"] or dataset["name"]
            if (
                current_name != expected_name
                or dataset["fingerprint"] != expected_fingerprint
            ):
                raise DatasetTrashConflictError("confirmation_mismatch")
            if self._active_jobs_reference_dataset(connection, dataset_id):
                raise DatasetTrashConflictError("dataset_in_use")
            connection.execute(
                """
                INSERT INTO dataset_trash(
                    dataset_id, storage_area, original_relative_path,
                    trash_relative_path, source_device, source_inode, state,
                    requested_by_profile_id,
                    requested_at, trashed_at, updated_at
                ) VALUES (?, ?, ?, ?, NULL, NULL, 'moving', ?, ?, NULL, ?)
                """,
                (
                    dataset_id,
                    dataset["storage_area"],
                    dataset["relative_path"],
                    f".datasetui-trash/{dataset_id}",
                    profile_id,
                    now,
                    now,
                ),
            )
        return self.get_dataset_trash(dataset_id)

    def record_dataset_trash_source_identity(
        self, dataset_id: str, *, source_device: int, source_inode: int
    ) -> dict[str, Any]:
        with self.connect() as connection:
            updated = connection.execute(
                """
                UPDATE dataset_trash
                SET source_device = ?, source_inode = ?, updated_at = ?
                WHERE dataset_id = ? AND state = 'moving'
                  AND source_device IS NULL AND source_inode IS NULL
                """,
                (source_device, source_inode, utc_now(), dataset_id),
            ).rowcount
        if updated != 1:
            raise DatasetTrashConflictError("trash_recovery_required")
        return self.get_dataset_trash(dataset_id)

    def finalize_dataset_trash(self, dataset_id: str) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                """
                UPDATE dataset_trash
                SET state = 'trashed', trashed_at = COALESCE(trashed_at, ?),
                    updated_at = ?
                WHERE dataset_id = ? AND state = 'moving'
                """,
                (now, now, dataset_id),
            ).rowcount
            if updated != 1:
                raise DatasetTrashConflictError("trash_recovery_required")
            connection.execute(
                "UPDATE datasets SET missing_since = COALESCE(missing_since, ?) WHERE id = ?",
                (now, dataset_id),
            )
        return self.get_dataset_trash(dataset_id)

    def abort_dataset_trash(self, dataset_id: str) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM dataset_trash WHERE dataset_id = ? AND state = 'moving'",
                (dataset_id,),
            )
            connection.execute(
                "UPDATE datasets SET missing_since = NULL WHERE id = ?", (dataset_id,)
            )

    def prepare_dataset_restore(
        self, dataset_id: str, *, profile_id: str, expected_fingerprint: str
    ) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._assert_active_profile(connection, profile_id)
            if connection.execute(
                "SELECT 1 FROM dataset_purge_reservations WHERE dataset_id = ? AND state IN ('reserved','purging','purged')",
                (dataset_id,),
            ).fetchone():
                raise DatasetTrashConflictError("trash_operation_in_progress")
            row = connection.execute(
                """
                SELECT t.state, d.fingerprint
                FROM dataset_trash t JOIN datasets d ON d.id = t.dataset_id
                WHERE t.dataset_id = ?
                """,
                (dataset_id,),
            ).fetchone()
            if row is None:
                raise DatasetNotFoundError(dataset_id)
            if row["state"] != "trashed":
                raise DatasetTrashConflictError("trash_recovery_required")
            if row["fingerprint"] != expected_fingerprint:
                raise DatasetTrashConflictError("confirmation_mismatch")
            if self._active_jobs_reference_dataset(connection, dataset_id):
                raise DatasetTrashConflictError("dataset_in_use")
            connection.execute(
                """
                UPDATE dataset_trash
                SET state = 'restoring', requested_by_profile_id = ?, updated_at = ?
                WHERE dataset_id = ? AND state = 'trashed'
                """,
                (profile_id, now, dataset_id),
            )
        return self.get_dataset_trash(dataset_id)

    def finalize_dataset_restore(self, dataset_id: str) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            state = connection.execute(
                "SELECT state, storage_area FROM dataset_trash WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchone()
            if state is None or state["state"] != "restoring":
                raise DatasetTrashConflictError("trash_recovery_required")
            connection.execute(
                "DELETE FROM dataset_trash WHERE dataset_id = ?", (dataset_id,)
            )
            connection.execute(
                """
                UPDATE datasets SET missing_since = NULL, last_seen_at = ?
                WHERE id = ?
                """,
                (now, dataset_id),
            )
            connection.execute(
                """
                INSERT INTO dataset_scan_generations(storage_area, generation, started_at)
                VALUES (?, 1, ?)
                ON CONFLICT(storage_area) DO UPDATE SET
                    generation = dataset_scan_generations.generation + 1,
                    started_at = excluded.started_at
                """,
                (state["storage_area"], now),
            )
        return self.get_dataset(dataset_id)

    def return_dataset_restore_to_trash(self, dataset_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE dataset_trash SET state = 'trashed', updated_at = ?
                WHERE dataset_id = ? AND state = 'restoring'
                """,
                (utc_now(), dataset_id),
            )
        return self.get_dataset_trash(dataset_id)

    def mark_dataset_trash_recovery_required(self, dataset_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE dataset_trash SET state = 'recovery_required', updated_at = ?
                WHERE dataset_id = ?
                """,
                (utc_now(), dataset_id),
            )
        return self.get_dataset_trash(dataset_id)

    def get_dataset_trash(self, dataset_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM dataset_trash WHERE dataset_id = ?", (dataset_id,)
            ).fetchone()
        if row is None:
            raise DatasetNotFoundError(dataset_id)
        record = dict(row)
        with self.connect() as connection:
            reservation = connection.execute(
                "SELECT state FROM dataset_purge_reservations WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchone()
        if reservation and reservation["state"] in {"reserved", "purging", "purged"}:
            record["state"] = {
                "reserved": "purge_queued",
                "purging": "purging",
                "purged": "purged",
            }[reservation["state"]]
        record["dataset"] = self.get_dataset(dataset_id, include_trashed=True)
        return record

    def list_dataset_trash(
        self, *, profile_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        with self.connect() as connection:
            self._assert_active_profile(connection, profile_id)
            rows = connection.execute(
                """
                SELECT dataset_id FROM dataset_trash
                WHERE NOT EXISTS (SELECT 1 FROM dataset_purge_reservations p
                    WHERE p.dataset_id = dataset_trash.dataset_id AND p.state = 'purged')
                ORDER BY COALESCE(trashed_at, requested_at) DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self.get_dataset_trash(row["dataset_id"]) for row in rows]

    def dataset_locations_for_trash(self, dataset_id: str) -> list[dict[str, str]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, storage_area, relative_path FROM datasets
                WHERE id != ? AND missing_since IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM dataset_trash t WHERE t.dataset_id = datasets.id
                  )
                """,
                (dataset_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _assert_active_profile(connection: sqlite3.Connection, profile_id: str) -> None:
        if (
            connection.execute(
                "SELECT 1 FROM profiles WHERE id = ? AND archived_at IS NULL",
                (profile_id,),
            ).fetchone()
            is None
        ):
            raise ProfileNotFoundError(profile_id)

    @staticmethod
    def _payload_dataset_ids(
        connection: sqlite3.Connection, payload: dict[str, Any]
    ) -> set[str]:
        dataset_ids: set[str] = set()
        if isinstance(payload.get("dataset_id"), str):
            dataset_ids.add(payload["dataset_id"])
        spec = payload.get("spec")
        if isinstance(spec, dict) and isinstance(spec.get("dataset_id"), str):
            dataset_ids.add(spec["dataset_id"])
        preview_ids = []
        if isinstance(payload.get("preview_id"), str):
            preview_ids.append(payload["preview_id"])
        if isinstance(payload.get("preview_ids"), list):
            preview_ids.extend(p for p in payload["preview_ids"] if isinstance(p, str))
        if isinstance(payload.get("previews"), list):
            preview_ids.extend(
                p["preview_id"] for p in payload["previews"]
                if isinstance(p, dict) and isinstance(p.get("preview_id"), str)
            )
        # Resolve only preview jobs, never recursively follow arbitrary payloads.
        for preview_id in set(preview_ids):
            preview = connection.execute(
                "SELECT payload_json FROM jobs WHERE id = ? AND kind = 'segmentation.preview'",
                (preview_id,),
            ).fetchone()
            if preview is not None:
                source_spec = json.loads(preview["payload_json"]).get("spec", {})
                if isinstance(source_spec, dict) and isinstance(source_spec.get("dataset_id"), str):
                    dataset_ids.add(source_spec["dataset_id"])
        sources = payload.get("sources")
        if isinstance(sources, list):
            dataset_ids.update(
                source["id"]
                for source in sources
                if isinstance(source, dict) and isinstance(source.get("id"), str)
            )
        snapshot_id = payload.get("snapshot_id")
        if isinstance(snapshot_id, str):
            snapshot = connection.execute(
                "SELECT dataset_id FROM curation_recipe_snapshots WHERE id = ?",
                (snapshot_id,),
            ).fetchone()
            if snapshot is not None:
                dataset_ids.add(snapshot["dataset_id"])
        return dataset_ids

    @classmethod
    def _active_jobs_reference_dataset(
        cls, connection: sqlite3.Connection, dataset_id: str
    ) -> bool:
        rows = connection.execute(
            "SELECT kind, payload_json FROM jobs WHERE status IN ('queued', 'running')"
        ).fetchall()
        for row in rows:
            if row["kind"] in {"datasets.scan", "hf.import"}:
                return True
            payload = json.loads(row["payload_json"])
            if dataset_id in cls._payload_dataset_ids(connection, payload):
                return True
        return False

    @classmethod
    def _assert_job_creation_allowed(
        cls, connection: sqlite3.Connection, payload: dict[str, Any], *, kind: str = ""
    ) -> None:
        repo_id = payload.get("repo_id")
        if kind == "datasets.upload_hf":
            repo_id = f"rainbowrobotics/{payload.get('repo_name', '')}"
        if repo_id:
            for row in connection.execute(
                "SELECT kind, payload_json FROM jobs WHERE status IN ('queued','running') AND kind IN ('hf.import','hf.delete','datasets.upload_hf')"
            ):
                existing = json.loads(row["payload_json"])
                existing_repo = (
                    existing.get("repo_id")
                    or f"rainbowrobotics/{existing.get('repo_name', '')}"
                )
                if repo_id == existing_repo and (
                    kind == "hf.delete" or row["kind"] == "hf.delete"
                ):
                    raise IdempotencyConflictError(
                        "이 HF 저장소의 가져오기·업로드·삭제 작업이 이미 대기 또는 실행 중입니다."
                    )
        if (
            connection.execute(
                "SELECT 1 FROM dataset_trash WHERE state IN ('moving', 'restoring') LIMIT 1"
            ).fetchone()
            is not None
        ):
            raise DatasetTrashConflictError("dataset_in_use")
        dataset_ids = cls._payload_dataset_ids(connection, payload)
        if not dataset_ids:
            return
        placeholders = ",".join("?" for _ in dataset_ids)
        if (
            connection.execute(
                f"SELECT 1 FROM dataset_trash WHERE dataset_id IN ({placeholders}) LIMIT 1",
                tuple(dataset_ids),
            ).fetchone()
            is not None
        ):
            raise DatasetNotReadyError(next(iter(dataset_ids)))

    def get_dataset(
        self, dataset_id: str, *, include_trashed: bool = False
    ) -> dict[str, Any]:
        trash_clause = (
            ""
            if include_trashed
            else (
                "AND NOT EXISTS (SELECT 1 FROM dataset_trash t WHERE t.dataset_id = datasets.id)"
            )
        )
        with self.connect() as connection:
            row = connection.execute(
                f"""
                SELECT id, storage_area, relative_path, name, display_name,
                       codebase_version,
                       readiness, robot_type, total_episodes, total_frames,
                       total_tasks, fps, fingerprint, scan_error, first_seen_at,
                       last_seen_at, missing_since
                FROM datasets WHERE id = ? {trash_clause}
                """,
                (dataset_id,),
            ).fetchone()
        if row is None:
            raise DatasetNotFoundError(dataset_id)
        return self._decode_dataset(row)

    def update_dataset_name(
        self, dataset_id: str, *, name: str, expected_name: str
    ) -> dict[str, Any]:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT name, display_name FROM datasets WHERE id = ?",
                (dataset_id,),
            ).fetchone()
            if row is None:
                raise DatasetNotFoundError(dataset_id)
            current_name = (
                row["display_name"] if row["display_name"] is not None else row["name"]
            )
            if current_name != expected_name:
                raise DatasetNameConflictError(dataset_id)
            display_name = None if name == row["name"] else name
            connection.execute(
                "UPDATE datasets SET display_name = ? WHERE id = ?",
                (display_name, dataset_id),
            )
        return self.get_dataset(dataset_id)

    def list_datasets(
        self,
        *,
        storage_area: str | None = None,
        readiness: str | None = None,
        include_missing: bool = False,
        limit: int = 100,
        sort: str = "name_asc",
        query: str | None = None,
        offset: int = 0,
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
        if query:
            clauses.append(
                "(COALESCE(display_name,name) LIKE ? ESCAPE '\\' OR relative_path LIKE ? ESCAPE '\\' OR robot_type LIKE ? ESCAPE '\\')"
            )
            pattern = (
                "%"
                + query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                + "%"
            )
            parameters.extend((pattern, pattern, pattern))
        clauses.append(
            "NOT EXISTS (SELECT 1 FROM dataset_trash t WHERE t.dataset_id = datasets.id)"
        )
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.extend((limit, offset))
        ordering = {
            "name_asc": "COALESCE(display_name, name) COLLATE NOCASE ASC",
            "name_desc": "COALESCE(display_name, name) COLLATE NOCASE DESC",
            "newest": "first_seen_at DESC",
            "oldest": "first_seen_at ASC",
        }.get(sort)
        if ordering is None:
            raise ValueError("Unsupported dataset sort")
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT id, storage_area, relative_path, name, display_name,
                       codebase_version,
                       readiness, robot_type, total_episodes, total_frames,
                       total_tasks, fps, fingerprint, scan_error, first_seen_at,
                       last_seen_at, missing_since
                FROM datasets
                {where}
                ORDER BY {ordering},
                         storage_area, relative_path
                LIMIT ? OFFSET ?
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
              AND NOT EXISTS (
                  SELECT 1 FROM dataset_trash t WHERE t.dataset_id = datasets.id
              )
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

    @staticmethod
    def _episode_annotations_record(
        connection: sqlite3.Connection,
        *,
        dataset_id: str,
        dataset_fingerprint: str,
        profile_id: str,
        episode_index: int,
    ) -> dict[str, Any]:
        row = connection.execute(
            """
            SELECT revision, task_override, atoms_json, updated_at
            FROM episode_annotations
            WHERE dataset_id = ? AND dataset_fingerprint = ?
              AND profile_id = ? AND episode_index = ?
            """,
            (dataset_id, dataset_fingerprint, profile_id, episode_index),
        ).fetchone()
        return {
            "dataset_id": dataset_id,
            "dataset_fingerprint": dataset_fingerprint,
            "profile_id": profile_id,
            "episode_index": episode_index,
            "revision": row["revision"] if row else 0,
            "task_override": row["task_override"] if row else None,
            "atoms": json.loads(row["atoms_json"]) if row else [],
            "updated_at": row["updated_at"] if row else None,
        }

    def get_episode_annotations(
        self, *, dataset_id: str, profile_id: str, episode_index: int
    ) -> dict[str, Any]:
        with self.connect() as connection:
            connection.execute("BEGIN")
            dataset = self._curation_source(
                connection, dataset_id=dataset_id, profile_id=profile_id
            )
            if episode_index < 0 or episode_index >= dataset["total_episodes"]:
                raise ValueError("episode index is outside the dataset")
            return self._episode_annotations_record(
                connection,
                dataset_id=dataset_id,
                dataset_fingerprint=dataset["fingerprint"],
                profile_id=profile_id,
                episode_index=episode_index,
            )

    def replace_episode_annotations(
        self,
        *,
        dataset_id: str,
        profile_id: str,
        episode_index: int,
        expected_revision: int,
        task_override: str | None,
        atoms: list[dict[str, Any]],
    ) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            dataset = self._curation_source(
                connection, dataset_id=dataset_id, profile_id=profile_id
            )
            if episode_index < 0 or episode_index >= dataset["total_episodes"]:
                raise ValueError("episode index is outside the dataset")
            current = self._episode_annotations_record(
                connection,
                dataset_id=dataset_id,
                dataset_fingerprint=dataset["fingerprint"],
                profile_id=profile_id,
                episode_index=episode_index,
            )
            if current["revision"] != expected_revision:
                raise AnnotationRevisionConflictError(dataset_id)
            connection.execute(
                """
                INSERT INTO episode_annotations(
                    dataset_id, dataset_fingerprint, profile_id, episode_index,
                    revision, task_override, atoms_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)
                ON CONFLICT(
                    dataset_id, dataset_fingerprint, profile_id, episode_index
                ) DO UPDATE SET
                    revision = episode_annotations.revision + 1,
                    task_override = excluded.task_override,
                    atoms_json = excluded.atoms_json,
                    updated_at = excluded.updated_at
                """,
                (
                    dataset_id,
                    dataset["fingerprint"],
                    profile_id,
                    episode_index,
                    task_override,
                    _json_dump(atoms),
                    now,
                    now,
                ),
            )
            return self._episode_annotations_record(
                connection,
                dataset_id=dataset_id,
                dataset_fingerprint=dataset["fingerprint"],
                profile_id=profile_id,
                episode_index=episode_index,
            )

    def create_curation_recipe(
        self,
        *,
        dataset_id: str,
        profile_id: str,
        name: str,
        selection_mode: str,
        operation: str = "subset",
        trim_config: dict[str, Any] | None = None,
        include_annotations: bool = False,
        relative_action: dict[str, Any] | None = None,
        split_config: dict[str, Any] | None = None,
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
                        name, name_key, selection_mode, operation,
                        trim_config_json, include_annotations, relative_action_json,
                        split_config_json,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        recipe_id,
                        dataset_id,
                        dataset["fingerprint"],
                        profile_id,
                        name,
                        _recipe_name_key(name),
                        selection_mode,
                        operation,
                        _json_dump(trim_config or {"enabled": False}),
                        int(include_annotations),
                        _json_dump(
                            relative_action or {"enabled": False, "dimensions": []}
                        ),
                        _json_dump(
                            split_config
                            or {"method": "flagged", "eval_percent": 20.0, "seed": 0}
                        ),
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
                       selection_mode, operation, trim_config_json,
                       include_annotations,
                       relative_action_json,
                       split_config_json,
                       created_at, updated_at, archived_at
                FROM curation_recipes
                WHERE dataset_id = ? AND dataset_fingerprint = ?
                  AND profile_id = ? {archived_clause}
                ORDER BY updated_at DESC, name_key
                """,
                (dataset_id, dataset["fingerprint"], profile_id),
            ).fetchall()
        return [self._decode_recipe(row) for row in rows]

    def get_curation_recipe(self, recipe_id: str, *, profile_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT id, dataset_id, dataset_fingerprint, profile_id, name,
                       selection_mode, operation, trim_config_json,
                       include_annotations,
                       relative_action_json,
                       split_config_json,
                       created_at, updated_at, archived_at
                FROM curation_recipes WHERE id = ? AND profile_id = ?
                """,
                (recipe_id, profile_id),
            ).fetchone()
        if row is None:
            raise RecipeNotFoundError(recipe_id)
        return self._decode_recipe(row)

    def update_curation_recipe(
        self,
        recipe_id: str,
        *,
        profile_id: str,
        name: str | None,
        selection_mode: str | None,
        operation: str | None = None,
        trim_config: dict[str, Any] | None = None,
        include_annotations: bool | None = None,
        relative_action: dict[str, Any] | None = None,
        split_config: dict[str, Any] | None = None,
        archived: bool | None = None,
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
                next_operation = (
                    operation if operation is not None else recipe["operation"]
                )
                next_trim_config = (
                    _json_dump(trim_config)
                    if trim_config is not None
                    else recipe["trim_config_json"]
                )
                next_include_annotations = (
                    int(include_annotations)
                    if include_annotations is not None
                    else recipe["include_annotations"]
                )
                next_relative_action = (
                    _json_dump(relative_action)
                    if relative_action is not None
                    else recipe["relative_action_json"]
                )
                next_split_config = (
                    _json_dump(split_config)
                    if split_config is not None
                    else recipe["split_config_json"]
                )
                next_archived_at = recipe["archived_at"]
                if archived is True:
                    next_archived_at = now
                elif archived is False:
                    next_archived_at = None
                connection.execute(
                    """
                    UPDATE curation_recipes
                    SET name = ?, name_key = ?, selection_mode = ?, operation = ?,
                        trim_config_json = ?, include_annotations = ?, relative_action_json = ?,
                        split_config_json = ?,
                        archived_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        next_name,
                        _recipe_name_key(next_name),
                        next_selection,
                        next_operation,
                        next_trim_config,
                        next_include_annotations,
                        next_relative_action,
                        next_split_config,
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
            if recipe["operation"] == "delete_flagged":
                selected = [
                    index
                    for index in range(dataset["total_episodes"])
                    if index not in flagged_set
                ]
            elif recipe["selection_mode"] == "all":
                selected = list(range(dataset["total_episodes"]))
            elif recipe["selection_mode"] == "flagged":
                selected = flagged
            else:
                selected = [
                    index
                    for index in range(dataset["total_episodes"])
                    if index not in flagged_set
                ]
            split_config = json.loads(recipe["split_config_json"])
            if (
                recipe["operation"] == "train_eval_split"
                and split_config["method"] == "flagged"
            ):
                selected = list(range(dataset["total_episodes"]))
            if not selected:
                raise ValueError("recipe selection cannot be empty")
            eval_episode_indices: list[int] = []
            if recipe["operation"] == "train_eval_split":
                if split_config["method"] == "random":
                    eval_episode_indices = _random_eval_episode_indices(
                        selected,
                        eval_percent=split_config["eval_percent"],
                        seed=split_config["seed"],
                    )
                else:
                    eval_episode_indices = flagged
                    if not flagged or len(flagged) == dataset["total_episodes"]:
                        raise ValueError(
                            "flag-based train/eval split requires non-empty train and eval sets"
                        )
            connection.execute(
                """
                INSERT INTO curation_recipe_snapshots(
                    id, recipe_id, dataset_id, dataset_fingerprint, profile_id,
                    recipe_name, selection_mode, flag_revision, operation,
                    trim_config_json, include_annotations, relative_action_json,
                    split_config_json, eval_episode_indices_json,
                    flagged_episode_indices_json, selected_episode_indices_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    recipe["operation"],
                    recipe["trim_config_json"],
                    recipe["include_annotations"],
                    recipe["relative_action_json"],
                    recipe["split_config_json"],
                    _json_dump(eval_episode_indices),
                    _json_dump(flagged),
                    _json_dump(selected),
                    now,
                ),
            )
            annotation_episode_indices: list[int] = []
            if recipe["include_annotations"]:
                selected_set = set(selected)
                annotation_rows = connection.execute(
                    """
                    SELECT episode_index, revision, task_override, atoms_json
                    FROM episode_annotations
                    WHERE dataset_id = ? AND dataset_fingerprint = ?
                      AND profile_id = ?
                    ORDER BY episode_index
                    """,
                    (
                        recipe["dataset_id"],
                        recipe["dataset_fingerprint"],
                        profile_id,
                    ),
                ).fetchall()
                for annotation in annotation_rows:
                    if annotation["episode_index"] not in selected_set:
                        continue
                    connection.execute(
                        """
                        INSERT INTO curation_annotation_snapshots(
                            snapshot_id, episode_index, annotation_revision,
                            task_override, atoms_json
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            snapshot_id,
                            annotation["episode_index"],
                            annotation["revision"],
                            annotation["task_override"],
                            annotation["atoms_json"],
                        ),
                    )
                    annotation_episode_indices.append(annotation["episode_index"])
        return {
            "id": snapshot_id,
            "recipe_id": recipe_id,
            "dataset_id": recipe["dataset_id"],
            "dataset_fingerprint": recipe["dataset_fingerprint"],
            "profile_id": profile_id,
            "recipe_name": recipe["name"],
            "selection_mode": recipe["selection_mode"],
            "operation": recipe["operation"],
            "trim_config": json.loads(recipe["trim_config_json"]),
            "include_annotations": bool(recipe["include_annotations"]),
            "relative_action": json.loads(recipe["relative_action_json"]),
            "split_config": split_config,
            "annotation_episode_indices": annotation_episode_indices,
            "flag_revision": flags["revision"],
            "flagged_episode_indices": flagged,
            "selected_episode_indices": selected,
            "eval_episode_indices": eval_episode_indices,
            "created_at": now,
        }

    def get_curation_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT s.*, d.storage_area, d.relative_path, d.name AS dataset_name,
                       d.codebase_version, d.readiness, d.missing_since, d.fingerprint
                FROM curation_recipe_snapshots s
                JOIN datasets d ON d.id = s.dataset_id
                WHERE s.id = ?
                """,
                (snapshot_id,),
            ).fetchone()
        if row is None:
            raise RecipeNotFoundError(snapshot_id)
        result = dict(row)
        result["flagged_episode_indices"] = json.loads(
            result.pop("flagged_episode_indices_json")
        )
        result["selected_episode_indices"] = json.loads(
            result.pop("selected_episode_indices_json")
        )
        result["eval_episode_indices"] = json.loads(
            result.pop("eval_episode_indices_json")
        )
        result["trim_config"] = json.loads(result.pop("trim_config_json"))
        result["include_annotations"] = bool(result["include_annotations"])
        result["relative_action"] = json.loads(result.pop("relative_action_json"))
        result["split_config"] = json.loads(result.pop("split_config_json"))
        if (
            result["operation"] == "train_eval_split"
            and result["split_config"]["method"] == "flagged"
            and not result["eval_episode_indices"]
        ):
            result["eval_episode_indices"] = list(result["flagged_episode_indices"])
        with self.connect() as connection:
            result["annotation_episode_indices"] = [
                annotation["episode_index"]
                for annotation in connection.execute(
                    """
                    SELECT episode_index FROM curation_annotation_snapshots
                    WHERE snapshot_id = ? ORDER BY episode_index
                    """,
                    (snapshot_id,),
                )
            ]
        return result

    def get_curation_snapshot_annotations(
        self, snapshot_id: str
    ) -> dict[int, dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT episode_index, annotation_revision, task_override, atoms_json
                FROM curation_annotation_snapshots
                WHERE snapshot_id = ? ORDER BY episode_index
                """,
                (snapshot_id,),
            ).fetchall()
        return {
            row["episode_index"]: {
                "revision": row["annotation_revision"],
                "task_override": row["task_override"],
                "atoms": json.loads(row["atoms_json"]),
            }
            for row in rows
        }

    def record_curation_run(
        self, *, job_id: str, snapshot_id: str, output_name: str
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO curation_runs(
                    job_id, snapshot_id, output_name, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (job_id, snapshot_id, output_name, utc_now()),
            )

    def record_validation_run(
        self,
        *,
        job_id: str,
        dataset_id: str,
        dataset_fingerprint: str,
        mode: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO validation_runs(
                    job_id, dataset_id, dataset_fingerprint, mode, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (job_id, dataset_id, dataset_fingerprint, mode, utc_now()),
            )

    def update_validation_progress(
        self, job_id: str, *, worker_id: str, progress: dict[str, Any]
    ) -> None:
        with self.connect() as connection:
            updated = connection.execute(
                """UPDATE validation_runs SET progress_json = ?
                WHERE job_id = ? AND EXISTS (
                    SELECT 1 FROM jobs WHERE id = validation_runs.job_id
                    AND status = 'running' AND worker_id = ?
                    AND lease_expires_at >= ?
                )""",
                (_json_dump(progress), job_id, worker_id, utc_now()),
            ).rowcount
            if updated != 1:
                raise JobLeaseLostError(job_id)

    def list_validation_runs(
        self, dataset_id: str, *, limit: int = 100
    ) -> list[dict[str, Any]]:
        self.get_dataset(dataset_id)
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT v.job_id, v.dataset_id, v.dataset_fingerprint, v.mode,
                       v.created_at, j.status, j.result_json, j.error_code,
                       j.error_message, j.finished_at, j.started_at, v.progress_json
                FROM validation_runs v
                JOIN jobs j ON j.id = v.job_id
                WHERE v.dataset_id = ? AND v.deleted_at IS NULL
                ORDER BY v.created_at DESC
                LIMIT ?
                """,
                (dataset_id, limit),
            ).fetchall()
        return [
            {
                **{key: row[key] for key in row.keys() if key != "progress_json"},
                "progress": json.loads(row["progress_json"])
                if row["progress_json"]
                else None,
                "result": json.loads(row["result_json"])
                if row["result_json"]
                else None,
            }
            for row in rows
        ]

    def delete_validation_run(self, dataset_id: str, job_id: str) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT j.status, v.deleted_at
                FROM validation_runs v
                JOIN jobs j ON j.id = v.job_id
                WHERE v.dataset_id = ? AND v.job_id = ?
                """,
                (dataset_id, job_id),
            ).fetchone()
            if row is None:
                raise ValidationRunNotFoundError(job_id)
            if row["status"] not in {
                "succeeded",
                "failed",
                "cancelled",
                "interrupted",
            }:
                raise ValidationRunActiveError(job_id)
            if row["deleted_at"] is None:
                connection.execute(
                    "UPDATE validation_runs SET deleted_at = ? WHERE job_id = ?",
                    (utc_now(), job_id),
                )

    def has_successful_export_gate(
        self, *, dataset_id: str, dataset_fingerprint: str
    ) -> bool:
        from datasetui.validation_integrity import VALIDATOR_POLICY

        result = self.export_gate_result(
            dataset_id=dataset_id, dataset_fingerprint=dataset_fingerprint
        )
        return bool(
            result
            and result.get("job_status") == "succeeded"
            and result.get("passed") is True
            and result.get("validator_policy") == VALIDATOR_POLICY
            and isinstance(result.get("content_manifest"), dict)
            and result["content_manifest"].get("tree_sha256")
        )

    def export_gate_result(
        self,
        *,
        dataset_id: str,
        dataset_fingerprint: str,
        exclude_job_id: str | None = None,
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT j.result_json, j.status, v.deleted_at
                FROM validation_runs v JOIN jobs j ON j.id = v.job_id
                WHERE v.dataset_id = ? AND v.dataset_fingerprint = ?
                  AND v.mode = 'export_gate'
                  AND (? IS NULL OR v.job_id != ?)
                ORDER BY v.created_at DESC, v.rowid DESC LIMIT 1
            """,
                (dataset_id, dataset_fingerprint, exclude_job_id, exclude_job_id),
            ).fetchone()
        if not row:
            return None
        if row["deleted_at"] is not None:
            return None
        result = json.loads(row["result_json"]) if row["result_json"] else {}
        return {**result, "job_status": row["status"]}

    @staticmethod
    def _decode_recipe(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["trim_config"] = json.loads(result.pop("trim_config_json"))
        result["include_annotations"] = bool(result["include_annotations"])
        result["relative_action"] = json.loads(result.pop("relative_action_json"))
        result["split_config"] = json.loads(result.pop("split_config_json"))
        return result

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

    def get_job_for_idempotency(
        self, profile_id: str, idempotency_key: str
    ) -> dict[str, Any] | None:
        return self._job_for_idempotency(profile_id, idempotency_key)

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
                  AND cancellation_requested_at IS NULL
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
                row = connection.execute(
                    "SELECT cancellation_requested_at FROM jobs WHERE id = ?",
                    (job_id,),
                ).fetchone()
                if row is not None and row["cancellation_requested_at"] is not None:
                    from datasetui.job_cancellation import JobCancellationRequested

                    raise JobCancellationRequested(job_id)
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
