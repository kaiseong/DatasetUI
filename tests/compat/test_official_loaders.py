"""Opt-in exact official-loader compatibility gates.

Set LEROBOT_V033_PYTHON and/or LEROBOT_V060_PYTHON to isolated interpreter
paths. The default project environment intentionally does not install LeRobot.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "scripts" / "compat" / "validate_official.py"


@pytest.mark.parametrize(
    ("environment", "format_name", "adapter", "fixture", "expected_version", "expected_frames"),
    [
        ("LEROBOT_V033_PYTHON", "v2.1", "v033", "v21_valid", "0.3.3", 20),
        ("LEROBOT_V060_PYTHON", "v3", "v060", "v30_valid", "0.6.0", 30),
        ("LEROBOT_V060_PYTHON", "v3", "v060", "v30_annotated", "0.6.0", 20),
    ],
)
def test_contract_fixture_with_exact_official_full_loader(
    environment: str,
    format_name: str,
    adapter: str,
    fixture: str,
    expected_version: str,
    expected_frames: int,
) -> None:
    python = os.environ.get(environment)
    if not python:
        pytest.skip(f"set {environment} to run the exact official-loader gate")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "tests" / "compat" / "stubs" / adapter)
    completed = subprocess.run(
        [
            python,
            str(VALIDATOR),
            "--format",
            format_name,
            "--dataset",
            str(ROOT / "fixtures" / fixture),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result["lerobot_version"] == expected_version
    assert result["all_frames_loaded"] == expected_frames
    assert result["full_loader_passed"] is True
