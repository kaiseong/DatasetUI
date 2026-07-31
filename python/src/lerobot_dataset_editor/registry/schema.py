"""Schema DDL, version tracking, and migration runner for the project registry."""

from __future__ import annotations

import sqlite3
from typing import Protocol


CURRENT_SCHEMA_VERSION = 1


class RegistryVersionError(Exception):
    """Raised when the DB schema version is newer than this code supports."""

    def __init__(self, db_version: int, code_version: int) -> None:
        super().__init__(
            f"Registry schema version {db_version} is newer than supported version "
            f"{code_version}. Please upgrade the application."
        )
        self.db_version = db_version
        self.code_version = code_version


class Migration(Protocol):
    """Protocol for a forward migration module."""

    def up(self, conn: sqlite3.Connection) -> None: ...


def get_migrations() -> list[Migration]:
    """Return ordered list of all migrations."""
    from .migrations import v001_initial

    return [v001_initial]


def bootstrap_or_migrate(conn: sqlite3.Connection) -> None:
    """Ensure schema is at CURRENT_SCHEMA_VERSION, creating or migrating as needed."""
    # Check if registry_meta exists
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='registry_meta'"
    )
    if cur.fetchone() is None:
        # Fresh database — run all migrations from scratch
        _run_all_migrations(conn)
        return

    # Existing database — check version
    cur = conn.execute("SELECT schema_version FROM registry_meta")
    row = cur.fetchone()
    if row is None:
        # Corrupt meta table — re-initialize
        _run_all_migrations(conn)
        return

    db_version = row[0]
    if db_version > CURRENT_SCHEMA_VERSION:
        raise RegistryVersionError(db_version, CURRENT_SCHEMA_VERSION)
    if db_version == CURRENT_SCHEMA_VERSION:
        return  # No-op, already current

    # Forward migration needed
    migrations = get_migrations()
    for i in range(db_version, CURRENT_SCHEMA_VERSION):
        migrations[i].up(conn)
    conn.execute(
        "UPDATE registry_meta SET schema_version = ?, updated_at = datetime('now')",
        (CURRENT_SCHEMA_VERSION,),
    )
    conn.commit()


def _run_all_migrations(conn: sqlite3.Connection) -> None:
    """Run all migrations on a fresh database."""
    migrations = get_migrations()
    for migration in migrations:
        migration.up(conn)
    conn.commit()
