"""Tests for runtime environment doctor probes."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest


def test_doctor_reports_python_version() -> None:
    """Doctor report includes the current Python version."""
    from lerobot_dataset_editor.runtime.doctor import run_doctor

    report = run_doctor()
    assert report["python"]["available"] is True
    expected_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    assert report["python"]["version"] == expected_version
    assert report["python"]["path"] == sys.executable


def test_doctor_reports_ffmpeg_missing_gracefully() -> None:
    """When ffmpeg is not found, doctor reports available=false without crashing."""
    from lerobot_dataset_editor.runtime.doctor import run_doctor

    with patch("shutil.which", return_value=None):
        report = run_doctor()
    assert report["ffmpeg"]["available"] is False
    assert report["ffmpeg"]["path"] is None


def test_doctor_reports_cuda_unavailable_gracefully() -> None:
    """When CUDA is not available, doctor reports it cleanly."""
    from lerobot_dataset_editor.runtime.doctor import run_doctor

    # Mock nvidia-smi not found and no torch
    with patch("shutil.which", side_effect=lambda name: None if name == "nvidia-smi" else f"/usr/bin/{name}"), \
         patch.dict(sys.modules, {"torch": None}):
        report = run_doctor()
    assert report["compute"]["cuda"]["available"] is False
