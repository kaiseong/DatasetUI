"""Initial schema migration: registry_meta + projects tables."""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    """Create the initial schema (version 1)."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS registry_meta (
            schema_version INTEGER NOT NULL,
            created_at     TEXT NOT NULL,
            updated_at     TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS projects (
            id              TEXT PRIMARY KEY,
            name            TEXT NOT NULL,
            source_path     TEXT NOT NULL,
            output_path     TEXT,
            selected_revision TEXT,
            target_format   TEXT NOT NULL CHECK(target_format IN ('v2.1', 'v3')),
            runtime_mode    TEXT NOT NULL DEFAULT 'embedded' CHECK(runtime_mode IN ('embedded', 'external')),
            python_path     TEXT,
            ffmpeg_path     TEXT,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            last_opened_at  TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_projects_last_opened ON projects(last_opened_at DESC);
    """)

    # Insert meta row only if not present (idempotent)
    cur = conn.execute("SELECT COUNT(*) FROM registry_meta")
    if cur.fetchone()[0] == 0:
        conn.execute(
            "INSERT INTO registry_meta (schema_version, created_at, updated_at) "
            "VALUES (1, datetime('now'), datetime('now'))"
        )
