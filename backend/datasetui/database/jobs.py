"""Job lifecycle: creation, idempotency, leases, cancellation, events."""

from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any

from datasetui.database.errors import (
    DatasetNotReadyError,
    DatasetTrashConflictError,
    IdempotencyConflictError,
    JobCancellationConflictError,
    JobLeaseLostError,
    JobNotFoundError,
    JobOwnershipError,
)
from datasetui.database.schema import json_dump, utc_after, utc_now


class JobsMixin:
    """Job lifecycle: creation, idempotency, leases, cancellation, events."""

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
                        json_dump(payload),
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
                    (profile_id, json_dump(check_payload)),
                ).fetchone()
                check_id = check["id"] if check else str(uuid.uuid4())
                if check is None:
                    connection.execute(
                        """INSERT INTO jobs(id,kind,queue_name,status,profile_id,payload_json,idempotency_key,created_at)
                        VALUES (?,'datasets.delivery_preflight','cpu','queued',?,?,?,?)""",
                        (
                            check_id,
                            profile_id,
                            json_dump(check_payload),
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
                        json_dump(payload),
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
        lease_expires_at = utc_after(lease_seconds)
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
        return self.request_job_cancellation(job_id, profile_id=profile_id)

    def is_job_cancellation_requested(
        self, job_id: str, *, worker_id: str
    ) -> bool:
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

    def complete_job_cancellation(self, job_id: str, *, worker_id: str) -> dict[str, Any]:
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
                    (now, utc_after(lease_seconds), job_id, worker_id),
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
                (json_dump(progress), job_id, worker_id, utc_now()),
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
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT id, kind, queue_name, status, profile_id, payload_json,
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
                    json_dump(result) if result is not None else None,
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
