"""Tests for security boundaries: token rejection, SQL injection, permissions."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest


@pytest.fixture
def registry_conn(tmp_path: Path):
    from lerobot_dataset_editor.registry.db import open_registry

    db_path = tmp_path / "datasetui" / "registry.sqlite3"
    conn = open_registry(db_path)
    yield conn
    conn.close()


def test_register_rejects_hub_token_in_any_field(registry_conn, tmp_path: Path) -> None:
    """Any field containing an HF Hub token pattern must be rejected."""
    from lerobot_dataset_editor.registry.repository import register_project

    source = tmp_path / "ds"
    source.mkdir()

    # Token in name
    with pytest.raises(ValueError, match="[Tt]oken"):
        register_project(
            registry_conn,
            name="hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456",
            source_path=str(source),
            target_format="v3",
            runtime_mode="embedded",
        )

    # Token in source_path
    with pytest.raises(ValueError, match="[Tt]oken"):
        register_project(
            registry_conn,
            name="Safe",
            source_path="/tmp/hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456/data",
            target_format="v3",
            runtime_mode="embedded",
        )

    # Token in output_path
    with pytest.raises(ValueError, match="[Tt]oken"):
        register_project(
            registry_conn,
            name="Safe",
            source_path=str(source),
            output_path="/tmp/hf_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456",
            target_format="v3",
            runtime_mode="embedded",
        )


def test_parameterized_queries_resist_injection(registry_conn, tmp_path: Path) -> None:
    """Adversarial SQL in path fields is stored and retrieved safely."""
    from lerobot_dataset_editor.registry.repository import get_project, register_project

    source = tmp_path / "ds"
    source.mkdir()
    adversarial_name = "'; DROP TABLE projects; --"

    result = register_project(
        registry_conn,
        name=adversarial_name,
        source_path=str(source),
        target_format="v3",
        runtime_mode="embedded",
    )

    # Table still exists
    cur = registry_conn.execute("SELECT COUNT(*) FROM projects")
    assert cur.fetchone()[0] >= 1

    # Value stored correctly
    project = get_project(registry_conn, result["id"])
    assert project["name"] == adversarial_name


def test_db_directory_permissions_0700(tmp_path: Path) -> None:
    """The registry DB directory must be created with mode 0o700."""
    from lerobot_dataset_editor.registry.db import open_registry

    db_path = tmp_path / "fresh_dir" / "registry.sqlite3"
    conn = open_registry(db_path)
    conn.close()

    db_dir = db_path.parent
    mode = db_dir.stat().st_mode & 0o777
    assert mode == 0o700, f"Expected 0o700, got {oct(mode)}"
