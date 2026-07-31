"""Executable schema contract for the Electron/Python transport."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[2]


def test_rpc_transport_schema_is_valid_and_freezes_framing() -> None:
    schema = json.loads((ROOT / "contracts" / "rpc-transport.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["$id"] == "urn:lerobot:dataset-editor:rpc-transport-v1"
    assert schema["x-framing"] == {
        "header": "Content-Length",
        "delimiter": "\\r\\n\\r\\n",
        "encoding": "utf-8",
        "max_content_length": 16777216,
    }
    assert schema["x-protocol-version"] == 1
    assert schema["x-methods"] == ["initialize", "project.get", "project.list", "project.register", "project.remove", "project.update", "report.get", "runtime.doctor", "runtime.select", "shutdown", "system.ping"]


def test_rpc_transport_schema_accepts_all_envelope_kinds() -> None:
    schema = json.loads((ROOT / "contracts" / "rpc-transport.schema.json").read_text())
    validator = jsonschema.Draft202012Validator(schema)
    examples = [
        {"jsonrpc": "2.0", "id": 1, "method": "system.ping"},
        {"jsonrpc": "2.0", "id": "a", "result": {"ok": True}},
        {"jsonrpc": "2.0", "id": 2, "error": {"code": -32601, "message": "missing"}},
        {"jsonrpc": "2.0", "method": "server.ready", "params": {"protocolVersion": 1}},
    ]
    for example in examples:
        validator.validate(example)
