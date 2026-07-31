"""Tests for path health detection (missing, read-only, moved)."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest


def test_existing_readable_path_reports_ok(tmp_path: Path) -> None:
    """An existing, readable, writable path reports status='ok'."""
    from lerobot_dataset_editor.registry.path_health import check_path

    target = tmp_path / "dataset"
    target.mkdir()
    result = check_path(str(target))
    assert result["status"] == "ok"


def test_missing_path_reports_missing(tmp_path: Path) -> None:
    """A path that does not exist reports status='missing'."""
    from lerobot_dataset_editor.registry.path_health import check_path

    result = check_path(str(tmp_path / "nonexistent"))
    assert result["status"] == "missing"


def test_readonly_path_reports_readonly(tmp_path: Path) -> None:
    """A path that exists but is not writable reports status='read_only'."""
    from lerobot_dataset_editor.registry.path_health import check_path

    target = tmp_path / "readonly_dir"
    target.mkdir()
    # Remove write permission
    target.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        result = check_path(str(target))
        assert result["status"] == "read_only"
    finally:
        # Restore write permission for cleanup
        target.chmod(stat.S_IRWXU)


def test_moved_path_detects_absence(tmp_path: Path) -> None:
    """A previously-known path that no longer exists reports status='missing'."""
    from lerobot_dataset_editor.registry.path_health import check_path

    target = tmp_path / "was_here"
    target.mkdir()
    # Confirm OK first
    assert check_path(str(target))["status"] == "ok"
    # Remove it (simulating a move)
    target.rmdir()
    # Now reports missing
    result = check_path(str(target))
    assert result["status"] == "missing"
