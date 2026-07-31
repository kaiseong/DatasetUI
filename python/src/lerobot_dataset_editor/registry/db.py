"""XDG-compliant SQLite registry database management."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from .schema import bootstrap_or_migrate
from .xdg import ensure_xdg_dirs, xdg_paths


def default_registry_path() -> Path:
    """Return ``$XDG_DATA_HOME/lerobot-dataset-editor/projects.sqlite``."""
    return xdg_paths().registry_db


def open_registry(db_path: Path | None = None) -> sqlite3.Connection:
    """Open or create the private migration-capable project registry."""
    if db_path is None:
        db_path = ensure_xdg_dirs().registry_db
    db_path = Path(db_path)

    db_dir = db_path.parent
    db_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(db_dir, 0o700)

    conn = sqlite3.connect(str(db_path), isolation_level="DEFERRED")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    bootstrap_or_migrate(conn)
    return conn
