"""Tests for registry schema bootstrap, migrations, and version checks."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


def test_fresh_db_creates_tables_and_version_1(tmp_path: Path) -> None:
    """Opening a fresh DB should create registry_meta (version=1) and projects table."""
    from lerobot_dataset_editor.registry.db import open_registry

    db_path = tmp_path / "datasetui" / "registry.sqlite3"
    conn = open_registry(db_path)
    try:
        # Check registry_meta exists and has version 1
        cur = conn.execute("SELECT schema_version FROM registry_meta")
        row = cur.fetchone()
        assert row is not None
        assert row[0] == 1

        # Check projects table exists with expected columns
        cur = conn.execute("PRAGMA table_info(projects)")
        columns = {r[1] for r in cur.fetchall()}
        expected = {
            "id", "name", "source_path", "output_path", "selected_revision",
            "target_format", "runtime_mode", "python_path", "ffmpeg_path",
            "created_at", "updated_at", "last_opened_at",
        }
        assert expected.issubset(columns)

        # Check WAL mode
        cur = conn.execute("PRAGMA journal_mode")
        assert cur.fetchone()[0] == "wal"
    finally:
        conn.close()


def test_open_newer_schema_raises_version_error(tmp_path: Path) -> None:
    """If the DB has a newer schema version than the code supports, raise RegistryVersionError."""
    from lerobot_dataset_editor.registry.db import open_registry
    from lerobot_dataset_editor.registry.schema import RegistryVersionError

    db_path = tmp_path / "datasetui" / "registry.sqlite3"
    # First create a valid DB
    conn = open_registry(db_path)
    # Bump the version to something future
    conn.execute("UPDATE registry_meta SET schema_version = 999")
    conn.commit()
    conn.close()

    # Re-opening should raise
    with pytest.raises(RegistryVersionError):
        open_registry(db_path)


def test_migration_v001_is_idempotent_with_existing_db(tmp_path: Path) -> None:
    """Opening the same DB twice at the same version should be a no-op."""
    from lerobot_dataset_editor.registry.db import open_registry

    db_path = tmp_path / "datasetui" / "registry.sqlite3"
    conn1 = open_registry(db_path)
    conn1.close()

    # Second open must succeed without raising or modifying version
    conn2 = open_registry(db_path)
    try:
        cur = conn2.execute("SELECT schema_version FROM registry_meta")
        assert cur.fetchone()[0] == 1
    finally:
        conn2.close()
