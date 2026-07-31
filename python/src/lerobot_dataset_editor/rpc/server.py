"""Stateful JSON-RPC server for the Electron subprocess boundary."""

from __future__ import annotations

import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, BinaryIO

from ..cli import generate_report
from .framing import (
    FrameProtocolError,
    FrameTooLargeError,
    JsonPayloadError,
    encode_frame,
    read_frame,
)
from .protocol import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    SERVER_NOT_INITIALIZED,
    RpcRequest,
    error_response,
    notification,
    parse_request,
    success_response,
)

PROTOCOL_VERSION = 1
METHODS = ["initialize", "report.get", "shutdown", "system.ping"]


class RpcDispatchError(Exception):
    def __init__(self, code: int, message: str, data: Any | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


class RpcServer:
    def __init__(self) -> None:
        self.initialized = False
        self.shutdown_requested = False
        self.started_at = time.monotonic()

    def dispatch(self, request: RpcRequest) -> Any:
        if request.method == "initialize":
            return self._initialize(request.params)
        if request.method == "shutdown":
            self.shutdown_requested = True
            return None
        if not self.initialized:
            raise RpcDispatchError(
                SERVER_NOT_INITIALIZED,
                "Server not initialized",
                {"requiredMethod": "initialize"},
            )
        if request.method == "system.ping":
            return {
                "service": "lerobot-dataset-editor",
                "protocolVersion": PROTOCOL_VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptimeMs": int((time.monotonic() - self.started_at) * 1000),
            }
        if request.method == "report.get":
            return generate_report()
        raise RpcDispatchError(
            METHOD_NOT_FOUND,
            "Method not found",
            {"method": request.method},
        )

    def _initialize(self, params: dict[str, Any] | list[Any] | None) -> dict[str, Any]:
        if self.initialized:
            raise RpcDispatchError(INVALID_REQUEST, "Server is already initialized")
        if not isinstance(params, dict) or params.get("protocolVersion") != PROTOCOL_VERSION:
            raise RpcDispatchError(
                INVALID_PARAMS,
                "initialize requires the supported protocolVersion",
                {"supportedProtocolVersion": PROTOCOL_VERSION},
            )
        client = params.get("client")
        if not isinstance(client, dict) or not isinstance(client.get("name"), str):
            raise RpcDispatchError(
                INVALID_PARAMS,
                "initialize requires client.name",
            )
        self.initialized = True
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "server": {"name": "lerobot-dataset-editor", "version": "0.1.0"},
            "methods": METHODS,
        }


def _write(stream: BinaryIO, message: dict[str, Any]) -> None:
    stream.write(encode_frame(message))
    stream.flush()


def run_server(input_stream: BinaryIO, output_stream: BinaryIO) -> int:
    server = RpcServer()
    _write(
        output_stream,
        notification("server.ready", {"protocolVersion": PROTOCOL_VERSION}),
    )
    while True:
        try:
            message = read_frame(input_stream)
        except JsonPayloadError as exc:
            _write(output_stream, error_response(None, PARSE_ERROR, "Parse error", str(exc)))
            continue
        except FrameTooLargeError as exc:
            _write(output_stream, error_response(None, INVALID_REQUEST, "Invalid Request", str(exc)))
            return 2
        except FrameProtocolError as exc:
            _write(output_stream, error_response(None, INVALID_REQUEST, "Invalid Request", str(exc)))
            continue
        if message is None:
            return 0

        try:
            request = parse_request(message)
        except ValueError as exc:
            _write(output_stream, error_response(None, INVALID_REQUEST, "Invalid Request", str(exc)))
            continue

        try:
            result = server.dispatch(request)
            response = success_response(request.id, result)
        except RpcDispatchError as exc:
            response = error_response(request.id, exc.code, exc.message, exc.data)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            response = error_response(request.id, INTERNAL_ERROR, "Internal error")
        _write(output_stream, response)

        if server.shutdown_requested:
            _write(output_stream, notification("server.exit", {"code": 0}))
            return 0


def main() -> None:
    raise SystemExit(run_server(sys.stdin.buffer, sys.stdout.buffer))
