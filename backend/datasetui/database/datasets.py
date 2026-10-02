"""Dataset registry scans, names, and the recoverable trash."""

from __future__ import annotations

import uuid
from typing import Any

from datasetui.database.errors import (
    DatasetNameConflictError,
    DatasetNotFoundError,
    DatasetTrashConflictError,
)
from datasetui.database.schema import utc_now


class DatasetsMixin:
    """Dataset registry scans, names, and the recoverable trash."""

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
