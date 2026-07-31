"""Tests for device selection policy (auto/gpu-only/cpu-only)."""

from __future__ import annotations

from unittest.mock import patch

import pytest


def test_auto_falls_back_to_cpu_on_cuda_init_failure() -> None:
    """Auto policy falls back to cpu when CUDA initialization fails."""
    from lerobot_dataset_editor.runtime.device_policy import resolve_device

    def fake_probe():
        return {"available": False, "error": "CUDA initialization failed"}

    result = resolve_device("auto", cuda_probe=fake_probe)
    assert result["selected"] == "cpu"
    assert result["fallback_reason"] is not None
    assert "CUDA" in result["fallback_reason"] or "cuda" in result["fallback_reason"]


def test_gpu_only_raises_on_no_cuda() -> None:
    """gpu-only policy raises RuntimeError when CUDA is unavailable."""
    from lerobot_dataset_editor.runtime.device_policy import resolve_device

    def fake_probe():
        return {"available": False, "error": "No CUDA devices found"}

    with pytest.raises(RuntimeError, match="[Cc][Uu][Dd][Aa]|GPU"):
        resolve_device("gpu-only", cuda_probe=fake_probe)


def test_auto_selects_gpu_when_available() -> None:
    """Auto policy selects cuda:0 when CUDA probe succeeds."""
    from lerobot_dataset_editor.runtime.device_policy import resolve_device

    def fake_probe():
        return {"available": True, "device": "cuda:0"}

    result = resolve_device("auto", cuda_probe=fake_probe)
    assert result["selected"] == "cuda:0"
    assert result["fallback_reason"] is None
