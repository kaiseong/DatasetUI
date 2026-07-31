"""CRUD operations for the project registry."""

from __future__ import annotations

import re
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from .path_health import check_path

# HF Hub token pattern: hf_ followed by 30+ alphanumeric chars
_HF_TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{30,}")

_VALID_FORMATS = ("v2.1", "v3")
_VALID_MODES = ("embedded", "external")


def _check_no_token(fields: dict[str, Any]) -> None:
    """Raise ValueError if any string field contains an HF Hub token."""
    for key, value in fields.items():
        if isinstance(value, str) and _HF_TOKEN_PATTERN.search(value):
            raise ValueError(
                f"Hub token detected in field '{key}'. "
                "Token storage is not permitted in the project registry."
            )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def register_project(
    conn: sqlite3.Connection,
    *,
    name: str,
    source_path: str,
    target_format: str,
    runtime_mode: str,
    output_path: str | None = None,
    selected_revision: str | None = None,
    python_path: str | None = None,
    ffmpeg_path: str | None = None,
) -> dict[str, str]:
    """Register a new project. Returns {"id": ..., "created_at": ...}."""
    if target_format not in _VALID_FORMATS:
        raise ValueError(f"target_format must be one of {_VALID_FORMATS}")
    if runtime_mode not in _VALID_MODES:
        raise ValueError(f"runtime_mode must be one of {_VALID_MODES}")

    # Token check on all string fields
    fields = {
        "name": name,
        "source_path": source_path,
        "output_path": output_path,
        "selected_revision": selected_revision,
        "python_path": python_path,
        "ffmpeg_path": ffmpeg_path,
    }
    _check_no_token({k: v for k, v in fields.items() if v is not None})

    project_id = str(uuid.uuid4())
    now = _now_iso()

    conn.execute(
        """INSERT INTO projects
        (id, name, source_path, output_path, selected_revision,
         target_format, runtime_mode, python_path, ffmpeg_path,
         created_at, updated_at, last_opened_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            project_id, name, source_path, output_path, selected_revision,
            target_format, runtime_mode, python_path, ffmpeg_path,
            now, now, now,
        ),
    )
    conn.commit()
    return {"id": project_id, "created_at": now}


def list_recent(conn: sqlite3.Connection, *, limit: int = 20) -> list[dict[str, Any]]:
    """Return projects ordered by last_opened_at descending, with path health."""
    cur = conn.execute(
        "SELECT id, name, source_path, output_path, selected_revision, "
        "target_format, runtime_mode, python_path, ffmpeg_path, "
        "created_at, updated_at, last_opened_at "
        "FROM projects ORDER BY last_opened_at DESC LIMIT ?",
        (limit,),
    )
    results = []
    for row in cur.fetchall():
        project = _row_to_dict(row)
        project["path_health"] = check_path(project["source_path"])
        results.append(project)
    return results


def get_project(conn: sqlite3.Connection, project_id: str) -> dict[str, Any]:
    """Get a single project by ID. Raises KeyError if not found."""
    cur = conn.execute(
        "SELECT id, name, source_path, output_path, selected_revision, "
        "target_format, runtime_mode, python_path, ffmpeg_path, "
        "created_at, updated_at, last_opened_at "
        "FROM projects WHERE id = ?",
        (project_id,),
    )
    row = cur.fetchone()
    if row is None:
        raise KeyError(f"Project not found: {project_id}")
    return _row_to_dict(row)


def update_project(conn: sqlite3.Connection, project_id: str, **kwargs: Any) -> None:
    """Update project fields. Only specified kwargs are changed."""
    allowed = {
        "name", "source_path", "output_path", "selected_revision",
        "target_format", "runtime_mode", "python_path", "ffmpeg_path",
    }
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return

    # Validate format/mode if being changed
    if "target_format" in updates and updates["target_format"] not in _VALID_FORMATS:
        raise ValueError(f"target_format must be one of {_VALID_FORMATS}")
    if "runtime_mode" in updates and updates["runtime_mode"] not in _VALID_MODES:
        raise ValueError(f"runtime_mode must be one of {_VALID_MODES}")

    # Token check
    _check_no_token({k: v for k, v in updates.items() if isinstance(v, str)})

    updates["updated_at"] = _now_iso()
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [project_id]
    conn.execute(f"UPDATE projects SET {set_clause} WHERE id = ?", values)
    conn.commit()


def remove_project(conn: sqlite3.Connection, project_id: str) -> None:
    """Remove a project from the registry (does NOT delete from disk)."""
    conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    conn.commit()


def _row_to_dict(row: tuple) -> dict[str, Any]:
    """Convert a SELECT row to a dict."""
    keys = [
        "id", "name", "source_path", "output_path", "selected_revision",
        "target_format", "runtime_mode", "python_path", "ffmpeg_path",
        "created_at", "updated_at", "last_opened_at",
    ]
    return dict(zip(keys, row))


def mark_project_opened(conn: sqlite3.Connection, project_id: str) -> None:
    """Move a project to the front of recents without accessing its dataset."""
    now = _now_iso()
    cursor = conn.execute(
        "UPDATE projects SET last_opened_at = ?, updated_at = ? WHERE id = ?",
        (now, now, project_id),
    )
    if cursor.rowcount != 1:
        conn.rollback()
        raise KeyError(f"Project not found: {project_id}")
    conn.commit()
