"""Tests for project registry CRUD operations."""

from __future__ import annotations

import time
from pathlib import Path

import pytest


@pytest.fixture
def registry_conn(tmp_path: Path):
    """Provide a fresh open registry connection."""
    from lerobot_dataset_editor.registry.db import open_registry

    db_path = tmp_path / "datasetui" / "registry.sqlite3"
    conn = open_registry(db_path)
    yield conn
    conn.close()


def test_register_project_persists_and_returns_id(registry_conn, tmp_path: Path) -> None:
    """Registering a project persists it and returns a valid UUID id."""
    from lerobot_dataset_editor.registry.repository import register_project

    source = tmp_path / "datasets" / "pick_place"
    source.mkdir(parents=True)

    result = register_project(
        registry_conn,
        name="PickPlace",
        source_path=str(source),
        target_format="v3",
        runtime_mode="embedded",
    )
    assert "id" in result
    assert len(result["id"]) == 36  # UUID4 string

    # Verify persistence
    cur = registry_conn.execute(
        "SELECT name, source_path, target_format, runtime_mode FROM projects WHERE id = ?",
        (result["id"],),
    )
    row = cur.fetchone()
    assert row is not None
    assert row[0] == "PickPlace"
    assert row[1] == str(source)
    assert row[2] == "v3"
    assert row[3] == "embedded"


def test_list_recent_orders_by_last_opened(registry_conn, tmp_path: Path) -> None:
    """list_recent returns projects ordered by last_opened_at descending."""
    from lerobot_dataset_editor.registry.repository import list_recent, register_project

    source1 = tmp_path / "ds1"
    source1.mkdir()
    source2 = tmp_path / "ds2"
    source2.mkdir()

    register_project(registry_conn, name="First", source_path=str(source1), target_format="v2.1", runtime_mode="embedded")
    time.sleep(0.01)  # Ensure different timestamps
    register_project(registry_conn, name="Second", source_path=str(source2), target_format="v3", runtime_mode="embedded")

    projects = list_recent(registry_conn, limit=10)
    assert len(projects) == 2
    assert projects[0]["name"] == "Second"
    assert projects[1]["name"] == "First"


def test_update_project_changes_format_and_runtime(registry_conn, tmp_path: Path) -> None:
    """update_project modifies fields and bumps updated_at."""
    from lerobot_dataset_editor.registry.repository import get_project, register_project, update_project

    source = tmp_path / "ds"
    source.mkdir()
    result = register_project(registry_conn, name="Test", source_path=str(source), target_format="v3", runtime_mode="embedded")
    project_id = result["id"]

    original = get_project(registry_conn, project_id)
    time.sleep(0.01)

    update_project(registry_conn, project_id, target_format="v2.1", runtime_mode="external", python_path="/usr/bin/python3.12")

    updated = get_project(registry_conn, project_id)
    assert updated["target_format"] == "v2.1"
    assert updated["runtime_mode"] == "external"
    assert updated["python_path"] == "/usr/bin/python3.12"
    assert updated["updated_at"] > original["updated_at"]


def test_remove_project_deletes_from_db_not_disk(registry_conn, tmp_path: Path) -> None:
    """remove_project removes from DB but does not delete the source directory."""
    from lerobot_dataset_editor.registry.repository import register_project, remove_project

    source = tmp_path / "ds"
    source.mkdir()
    (source / "info.json").write_text("{}")

    result = register_project(registry_conn, name="Test", source_path=str(source), target_format="v3", runtime_mode="embedded")
    project_id = result["id"]

    remove_project(registry_conn, project_id)

    cur = registry_conn.execute("SELECT id FROM projects WHERE id = ?", (project_id,))
    assert cur.fetchone() is None
    # Directory must still exist
    assert source.exists()
    assert (source / "info.json").exists()


def test_register_duplicate_source_path_allowed(registry_conn, tmp_path: Path) -> None:
    """The same source path can be registered as different projects."""
    from lerobot_dataset_editor.registry.repository import register_project

    source = tmp_path / "ds"
    source.mkdir()

    r1 = register_project(registry_conn, name="ProjectA", source_path=str(source), target_format="v3", runtime_mode="embedded")
    r2 = register_project(registry_conn, name="ProjectB", source_path=str(source), target_format="v2.1", runtime_mode="embedded")

    assert r1["id"] != r2["id"]


def test_mark_project_opened_updates_recency_without_touching_source(registry_conn, tmp_path: Path) -> None:
    from lerobot_dataset_editor.registry.repository import (
        list_recent,
        mark_project_opened,
        register_project,
    )

    first_source = tmp_path / "first"
    first_source.mkdir()
    sentinel = first_source / "sensor.bin"
    sentinel.write_bytes(b"immutable-source")
    second_source = tmp_path / "second"
    second_source.mkdir()

    first = register_project(
        registry_conn,
        name="First",
        source_path=str(first_source),
        target_format="v3",
        runtime_mode="embedded",
    )
    time.sleep(0.01)
    register_project(
        registry_conn,
        name="Second",
        source_path=str(second_source),
        target_format="v2.1",
        runtime_mode="embedded",
    )
    assert list_recent(registry_conn)[0]["name"] == "Second"

    time.sleep(0.01)
    mark_project_opened(registry_conn, first["id"])
    assert list_recent(registry_conn)[0]["name"] == "First"
    assert sentinel.read_bytes() == b"immutable-source"
