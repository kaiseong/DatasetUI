"""Subprocess lifecycle tests for ``python -m lerobot_dataset_editor.rpc``."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import jsonschema
import pytest

from lerobot_dataset_editor.rpc.framing import encode_frame, read_frame
from lerobot_dataset_editor.schemas import rpc_schema

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def rpc_process() -> Iterator[subprocess.Popen[bytes]]:
    process = subprocess.Popen(
        [sys.executable, "-m", "lerobot_dataset_editor.rpc"],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT / "python" / "src")},
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


def test_server_lifecycle_ping_report_and_shutdown(rpc_process: subprocess.Popen[bytes]) -> None:
    ready = _read(rpc_process)
    assert ready == {
        "jsonrpc": "2.0",
        "method": "server.ready",
        "params": {"protocolVersion": 1},
    }

    before_initialize = _request(rpc_process, 1, "system.ping")
    assert before_initialize["id"] == 1
    assert before_initialize["error"]["code"] == -32002

    initialized = _request(
        rpc_process,
        2,
        "initialize",
        {"client": {"name": "pytest", "version": "1"}, "protocolVersion": 1},
    )
    assert initialized["id"] == 2
    assert initialized["result"]["protocolVersion"] == 1
    assert initialized["result"]["methods"] == [
        "initialize",
        "project.get",
        "project.list",
        "project.register",
        "project.remove",
        "project.update",
        "report.get",
        "runtime.doctor",
        "runtime.select",
        "shutdown",
        "system.ping",
    ]

    first_ping = _request(rpc_process, 3, "system.ping")["result"]
    second_ping = _request(rpc_process, 4, "system.ping")["result"]
    assert first_ping["service"] == "lerobot-dataset-editor"
    assert first_ping["protocolVersion"] == 1
    assert first_ping["timestamp"].endswith("+00:00")
    assert second_ping["uptimeMs"] >= first_ping["uptimeMs"]

    report = _request(rpc_process, 5, "report.get")["result"]
    jsonschema.Draft202012Validator(rpc_schema()).validate(report)
    assert report["status"] == "ok"

    missing = _request(rpc_process, 6, "missing.method")
    assert missing["error"]["code"] == -32601

    shutdown = _request(rpc_process, 7, "shutdown")
    assert shutdown == {"jsonrpc": "2.0", "id": 7, "result": None}
    assert _read(rpc_process) == {"jsonrpc": "2.0", "method": "server.exit", "params": {"code": 0}}
    assert rpc_process.wait(timeout=5) == 0
    assert rpc_process.stdout is not None
    assert rpc_process.stdout.read() == b""


def test_server_returns_parse_and_invalid_request_errors(rpc_process: subprocess.Popen[bytes]) -> None:
    _read(rpc_process)
    assert rpc_process.stdin is not None

    malformed = b"{broken"
    rpc_process.stdin.write(
        f"Content-Length: {len(malformed)}\r\n\r\n".encode("ascii") + malformed
    )
    rpc_process.stdin.flush()
    parse_error = _read(rpc_process)
    assert parse_error["id"] is None
    assert parse_error["error"]["code"] == -32700

    batch = json.dumps([]).encode()
    rpc_process.stdin.write(f"Content-Length: {len(batch)}\r\n\r\n".encode() + batch)
    rpc_process.stdin.flush()
    invalid = _read(rpc_process)
    assert invalid["id"] is None
    assert invalid["error"]["code"] == -32600
