"""Researcher profiles."""

from __future__ import annotations

import sqlite3
import uuid
from typing import Any

from datasetui.database.errors import DuplicateProfileNameError, ProfileNotFoundError
from datasetui.database.schema import profile_name_key, utc_now


class ProfilesMixin:
    """Researcher profiles."""

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
                    (profile_id, name, profile_name_key(name), now, now),
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
                        profile_name_key(next_name),
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
