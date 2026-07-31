from __future__ import annotations

import json
from pathlib import Path

import pytest


def _executable(path: Path, content: str) -> str:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
    return str(path)


def _fake_python(path: Path, *, python: str, lerobot: str) -> str:
    prefix = {
        "python": {"path": str(path), "version": python, "available": True},
        "lerobot": {"version": lerobot, "available": True, "import_error": None},
        "torch": {
            "version": "2.7.1+cu128",
            "cuda_runtime": "12.8",
            "cuda_available": False,
        },
    }
    encoded = json.dumps(prefix, separators=(",", ":"))
    return _executable(
        path,
        "#!/bin/sh\n"
        "if [ -n \"${HF_TOKEN:-}${AWS_SECRET_ACCESS_KEY:-}\" ]; then secret=true; else secret=false; fi\n"
        f"printf '%s' '{encoded[:-1]},\"secret_seen\":'\n"
        "printf '%s\\n' \"$secret}\"\n",
    )


def _fake_ffmpeg(path: Path) -> str:
    return _executable(
        path,
        "#!/bin/sh\n"
        "if [ \"$1\" = \"-version\" ]; then echo 'ffmpeg version 7.1.1 static'; exit 0; fi\n"
        "if [ \"$1\" = \"-encoders\" ]; then echo ' V....D libx264 H.264'; exit 0; fi\n"
        "exit 2\n",
    )


def test_external_doctor_uses_selected_executables_and_accepts_exact_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python = _fake_python(tmp_path / "python3.12", python="3.12.8", lerobot="0.6.0")
    ffmpeg = _fake_ffmpeg(tmp_path / "ffmpeg")
    monkeypatch.setenv("HF_TOKEN", "must-not-cross-probe-boundary")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-cross-probe-boundary")

    from lerobot_dataset_editor.runtime.doctor import run_doctor

    report = run_doctor(
        mode="external",
        python_path=python,
        ffmpeg_path=ffmpeg,
        device_policy="cpu-only",
    )
    assert report["mode"] == "external"
    assert report["python"]["path"] == python
    assert report["python"]["version"] == "3.12.8"
    assert report["lerobot"]["version"] == "0.6.0"
    assert report["ffmpeg"]["path"] == ffmpeg
    assert report["ffmpeg"]["version"] == "7.1.1"
    assert "libx264" in report["ffmpeg"]["codecs"]
    assert report["compatible"] is True
    assert report["issues"] == []
    assert report["probe_environment_secret_seen"] is False


def test_external_doctor_rejects_wrong_python_and_lerobot(tmp_path: Path) -> None:
    python = _fake_python(tmp_path / "python3", python="3.11.9", lerobot="0.5.2")
    ffmpeg = _fake_ffmpeg(tmp_path / "ffmpeg")

    from lerobot_dataset_editor.runtime.doctor import run_doctor

    report = run_doctor(
        mode="external",
        python_path=python,
        ffmpeg_path=ffmpeg,
        device_policy="cpu-only",
    )
    assert report["compatible"] is False
    assert any("Python 3.12" in issue for issue in report["issues"])
    assert any("lerobot==0.6.0" in issue for issue in report["issues"])


def test_external_doctor_rejects_missing_ffmpeg(tmp_path: Path) -> None:
    python = _fake_python(tmp_path / "python3.12", python="3.12.8", lerobot="0.6.0")

    from lerobot_dataset_editor.runtime.doctor import run_doctor

    report = run_doctor(
        mode="external",
        python_path=python,
        ffmpeg_path=str(tmp_path / "missing-ffmpeg"),
        device_policy="cpu-only",
    )
    assert report["compatible"] is False
    assert any("FFmpeg" in issue for issue in report["issues"])


def test_gpu_only_failure_is_not_reported_as_a_successful_doctor_result() -> None:
    from lerobot_dataset_editor.runtime.doctor import run_doctor

    with pytest.raises(RuntimeError, match="GPU-only"):
        run_doctor(
            device_policy="gpu-only",
            cuda_probe=lambda: {"available": False, "error": "CUDA init failed"},
        )
