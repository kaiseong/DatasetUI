"""Tests for runtime environment resolution (embedded vs external)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_embedded_resolves_to_venv_python(tmp_path: Path) -> None:
    """Embedded mode resolves to the .venv/bin/python within the app root."""
    from lerobot_dataset_editor.runtime.environment import resolve_runtime_paths

    # Create a fake venv structure
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    python_exe = venv_bin / "python"
    python_exe.touch()
    python_exe.chmod(0o755)

    result = resolve_runtime_paths(mode="embedded", app_root=str(tmp_path))
    assert result["python_path"] == str(python_exe)


def test_external_validates_executable_exists(tmp_path: Path) -> None:
    """External mode raises an error if the specified python_path does not exist."""
    from lerobot_dataset_editor.runtime.environment import resolve_runtime_paths

    with pytest.raises((FileNotFoundError, ValueError)):
        resolve_runtime_paths(
            mode="external",
            python_path="/nonexistent/python3.12",
            app_root=str(tmp_path),
        )
