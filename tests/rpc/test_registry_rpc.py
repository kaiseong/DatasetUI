"""End-to-end RPC subprocess tests for project registry and runtime methods."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from lerobot_dataset_editor.rpc.framing import encode_frame, read_frame

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def rpc_process(tmp_path: Path) -> Iterator[subprocess.Popen[bytes]]:
    """Spawn RPC subprocess with isolated XDG_DATA_HOME."""
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(ROOT / "python" / "src"),
        "HOME": str(tmp_path),
        "XDG_DATA_HOME": str(tmp_path / "xdg_data"),
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "lerobot_dataset_editor.rpc"],
        cwd=str(ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def _read(process: subprocess.Popen[bytes]) -> dict:
    assert process.stdout is not None
    message = read_frame(process.stdout)
    assert isinstance(message, dict)
    return message


def _request(process: subprocess.Popen[bytes], request_id: int, method: str, params=None) -> dict:
    assert process.stdin is not None
    message = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    process.stdin.write(encode_frame(message))
    process.stdin.flush()
    return _read(process)


def _initialize(process: subprocess.Popen[bytes]) -> dict:
    _read(process)  # server.ready
    return _request(process, 1, "initialize", {
        "client": {"name": "pytest", "version": "1"},
        "protocolVersion": 1,
    })


def _fake_runtime(tmp_path: Path, *, python_version: str, lerobot_version: str) -> tuple[str, str]:
    python = tmp_path / f"python-{python_version}"
    payload = {
        "python": {"path": str(python), "version": python_version, "available": True},
        "lerobot": {"version": lerobot_version, "available": True, "import_error": None},
        "torch": {
            "version": "2.7.1+cu128",
            "cuda_runtime": "12.8",
            "cuda_available": False,
        },
        "secret_seen": False,
    }
    python.write_text(
        "#!/bin/sh\nprintf '%s\\n' '" + json.dumps(payload, separators=(",", ":")) + "'\n",
        encoding="utf-8",
    )
    python.chmod(0o755)

    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"-version\" ]; then echo 'ffmpeg version 7.1.1 static'; exit 0; fi\n"
        "if [ \"$1\" = \"-encoders\" ]; then echo ' V....D libx264 H.264'; exit 0; fi\n"
        "exit 2\n",
        encoding="utf-8",
    )
    ffmpeg.chmod(0o755)
    return str(python), str(ffmpeg)


def _register(process: subprocess.Popen[bytes], tmp_path: Path, request_id: int = 2) -> str:
    source = tmp_path / f"ds-{request_id}"
    source.mkdir()
    response = _request(process, request_id, "project.register", {
        "name": "TestProject",
        "source_path": str(source),
        "target_format": "v3",
        "runtime_mode": "embedded",
    })
    return response["result"]["id"]


def test_project_register_via_rpc(rpc_process: subprocess.Popen[bytes], tmp_path: Path) -> None:
    _initialize(rpc_process)
    project_id = _register(rpc_process, tmp_path)
    assert len(project_id) == 36


def test_project_list_via_rpc(rpc_process: subprocess.Popen[bytes], tmp_path: Path) -> None:
    _initialize(rpc_process)
    source1 = tmp_path / "ds1"
    source1.mkdir()
    source2 = tmp_path / "ds2"
    source2.mkdir()

    _request(rpc_process, 2, "project.register", {
        "name": "First", "source_path": str(source1), "target_format": "v3", "runtime_mode": "embedded",
    })
    _request(rpc_process, 3, "project.register", {
        "name": "Second", "source_path": str(source2), "target_format": "v2.1", "runtime_mode": "embedded",
    })

    response = _request(rpc_process, 4, "project.list", {"limit": 10})
    assert response["result"]["total"] == 2
    assert response["result"]["projects"][0]["name"] == "Second"


def test_runtime_doctor_via_rpc(rpc_process: subprocess.Popen[bytes], tmp_path: Path) -> None:
    _initialize(rpc_process)
    project_id = _register(rpc_process, tmp_path)

    response = _request(rpc_process, 3, "runtime.doctor", {"project_id": project_id})
    report = response["result"]
    assert report["mode"] == "embedded"
    assert report["python"]["available"] is True
    assert "compatible" in report
    assert "compute" in report


def test_runtime_select_accepts_compatible_external_environment(
    rpc_process: subprocess.Popen[bytes], tmp_path: Path
) -> None:
    _initialize(rpc_process)
    project_id = _register(rpc_process, tmp_path)
    python, ffmpeg = _fake_runtime(tmp_path, python_version="3.12.8", lerobot_version="0.6.0")

    response = _request(rpc_process, 3, "runtime.select", {
        "project_id": project_id,
        "runtime_mode": "external",
        "python_path": python,
        "ffmpeg_path": ffmpeg,
        "device_policy": "cpu-only",
    })
    assert response["result"]["runtime_mode"] == "external"

    doctor = _request(rpc_process, 4, "runtime.doctor", {"project_id": project_id, "device_policy": "cpu-only"})
    assert doctor["result"]["compatible"] is True
    assert doctor["result"]["python"]["version"] == "3.12.8"


def test_runtime_select_rejects_incompatible_external_environment_without_persisting(
    rpc_process: subprocess.Popen[bytes], tmp_path: Path
) -> None:
    _initialize(rpc_process)
    project_id = _register(rpc_process, tmp_path)
    python, ffmpeg = _fake_runtime(tmp_path, python_version="3.11.9", lerobot_version="0.5.2")

    response = _request(rpc_process, 3, "runtime.select", {
        "project_id": project_id,
        "runtime_mode": "external",
        "python_path": python,
        "ffmpeg_path": ffmpeg,
        "device_policy": "cpu-only",
    })
    assert response["error"]["code"] == -32602
    assert "Python 3.12" in response["error"]["message"]

    project = _request(rpc_process, 4, "project.get", {"id": project_id})
    assert project["result"]["runtime_mode"] == "embedded"
    assert project["result"]["python_path"] is None
