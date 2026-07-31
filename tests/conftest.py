"""Pytest configuration and shared fixtures for dataset_editor tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the src directory is on the path
SRC_DIR = Path(__file__).resolve().parent.parent / "python" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(scope="session")
def fixtures_dir(tmp_path_factory) -> Path:
    """Generate all fixtures in a session-scoped temp directory."""
    from lerobot_dataset_editor.fixtures import materialize_all

    base = tmp_path_factory.mktemp("fixtures")
    materialize_all(base)
    return base


@pytest.fixture
def v21_fixture(fixtures_dir) -> Path:
    return fixtures_dir / "v21_valid"


@pytest.fixture
def v30_fixture(fixtures_dir) -> Path:
    return fixtures_dir / "v30_valid"


@pytest.fixture
def v30_annotated_fixture(fixtures_dir) -> Path:
    return fixtures_dir / "v30_annotated"


@pytest.fixture
def corrupt_fixture(fixtures_dir) -> Path:
    return fixtures_dir / "corrupt"
