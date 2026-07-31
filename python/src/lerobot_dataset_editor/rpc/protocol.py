"""Small strict JSON-RPC 2.0 envelope model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
SERVER_NOT_INITIALIZED = -32002

RequestId = int | str


@dataclass(frozen=True)
class RpcRequest:
    id: RequestId
    method: str
    params: dict[str, Any] | list[Any] | None = None


def parse_request(message: object) -> RpcRequest:
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        raise ValueError("invalid JSON-RPC version or envelope")
    request_id = message.get("id")
    if isinstance(request_id, bool) or not isinstance(request_id, (int, str)):
        raise ValueError("request id must be an integer or string")
    method = message.get("method")
    if not isinstance(method, str) or not method:
        raise ValueError("request method must be a non-empty string")
    params = message.get("params")
    if params is not None and not isinstance(params, (dict, list)):
        raise ValueError("request params must be an object or array")
    return RpcRequest(id=request_id, method=method, params=params)


def success_response(request_id: RequestId, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def error_response(
    request_id: RequestId | None,
    code: int,
    message: str,
    data: Any | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def notification(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        message["params"] = params
    return message
