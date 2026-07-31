"""JSON-RPC 2.0 envelope validation tests."""

from __future__ import annotations

import pytest

from lerobot_dataset_editor.rpc.protocol import (
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    SERVER_NOT_INITIALIZED,
    RpcRequest,
    error_response,
    parse_request,
    success_response,
)


def test_parse_request_accepts_integer_and_string_ids() -> None:
    assert parse_request({"jsonrpc": "2.0", "id": 4, "method": "system.ping"}) == RpcRequest(
        id=4, method="system.ping", params=None
    )
    assert parse_request(
        {"jsonrpc": "2.0", "id": "request-1", "method": "initialize", "params": {}}
    ) == RpcRequest(id="request-1", method="initialize", params={})


@pytest.mark.parametrize(
    "message",
    [
        [],
        {},
        {"jsonrpc": "1.0", "id": 1, "method": "ping"},
        {"jsonrpc": "2.0", "id": None, "method": "ping"},
        {"jsonrpc": "2.0", "id": 1, "method": ""},
        {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": "bad"},
    ],
)
def test_parse_request_rejects_invalid_envelopes(message: object) -> None:
    with pytest.raises(ValueError):
        parse_request(message)


def test_success_and_error_responses_are_spec_conformant() -> None:
    assert success_response("x", {"ok": True}) == {
        "jsonrpc": "2.0",
        "id": "x",
        "result": {"ok": True},
    }
    assert error_response(9, METHOD_NOT_FOUND, "No such method", {"method": "missing"}) == {
        "jsonrpc": "2.0",
        "id": 9,
        "error": {
            "code": -32601,
            "message": "No such method",
            "data": {"method": "missing"},
        },
    }
    assert INVALID_REQUEST == -32600
    assert METHOD_NOT_FOUND == -32601
    assert SERVER_NOT_INITIALIZED == -32002
