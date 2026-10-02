"""Validation runs and the content-bound export gate."""

from __future__ import annotations

import json
from typing import Any

from datasetui.database.errors import (
    JobLeaseLostError,
    ValidationRunActiveError,
    ValidationRunNotFoundError,
)
from datasetui.database.schema import json_dump, utc_now


class ValidationMixin:
    """Validation runs and the content-bound export gate."""

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
                (json_dump(progress), job_id, worker_id, utc_now()),
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
        from datasetui.validation.integrity import VALIDATOR_POLICY

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
