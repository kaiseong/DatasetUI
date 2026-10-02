"""Hugging Face import jobs and pinned revisions."""

from __future__ import annotations

import sqlite3
import uuid
from typing import Any

from datasetui.database.errors import (
    DatasetTrashConflictError,
    IdempotencyConflictError,
    ImmutableRevisionConflictError,
)
from datasetui.database.schema import json_dump, utc_now


class HuggingFaceMixin:
    """Hugging Face import jobs and pinned revisions."""

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
                    WHERE d.storage_area = 'raw' AND d.relative_path = ? LIMIT 1
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
                    (job_id, profile_id, json_dump(payload), idempotency_key, now),
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
