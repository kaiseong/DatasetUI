"""Bounded Content-Length framing for JSON-RPC over binary stdio."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, BinaryIO

MAX_CONTENT_LENGTH = 16 * 1024 * 1024
MAX_HEADER_LENGTH = 8 * 1024
_CONTENT_LENGTH = re.compile(r"0|[1-9][0-9]*")


class FrameProtocolError(ValueError):
    """The stream does not contain a valid framed JSON object."""


class FrameTooLargeError(FrameProtocolError):
    """The declared payload exceeds the transport safety limit."""


class JsonPayloadError(FrameProtocolError):
    """The frame body is not valid JSON."""


def encode_frame(message: Mapping[str, Any]) -> bytes:
    payload = json.dumps(
        dict(message), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(payload) > MAX_CONTENT_LENGTH:
        raise FrameTooLargeError(
            f"payload length {len(payload)} exceeds maximum {MAX_CONTENT_LENGTH}"
        )
    return f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii") + payload


def _read_exact(stream: BinaryIO, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            received = length - remaining
            raise FrameProtocolError(
                f"truncated frame body: expected {length} bytes, received {received}"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(stream: BinaryIO) -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    header_bytes = 0
    saw_header = False
    while True:
        line = stream.readline(MAX_HEADER_LENGTH + 1)
        if line == b"" and not saw_header:
            return None
        if line == b"":
            raise FrameProtocolError("truncated frame headers")
        saw_header = True
        header_bytes += len(line)
        if header_bytes > MAX_HEADER_LENGTH:
            raise FrameProtocolError(f"frame headers exceed {MAX_HEADER_LENGTH} bytes")
        if line == b"\r\n":
            break
        if not line.endswith(b"\r\n"):
            raise FrameProtocolError("frame headers must use CRLF line endings")
        try:
            name, value = line[:-2].decode("ascii").split(":", 1)
        except (UnicodeDecodeError, ValueError) as exc:
            raise FrameProtocolError("malformed frame header") from exc
        normalized = name.strip().lower()
        if not normalized:
            raise FrameProtocolError("empty frame header name")
        if normalized in headers:
            raise FrameProtocolError(f"duplicate frame header: {name.strip()}")
        headers[normalized] = value.strip()

    raw_length = headers.get("content-length")
    if raw_length is None:
        raise FrameProtocolError("missing Content-Length header")
    if not _CONTENT_LENGTH.fullmatch(raw_length):
        raise FrameProtocolError("Content-Length must be a non-negative integer")
    length = int(raw_length)
    if length > MAX_CONTENT_LENGTH:
        raise FrameTooLargeError(
            f"declared payload exceeds maximum {MAX_CONTENT_LENGTH} bytes"
        )

    payload = _read_exact(stream, length)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JsonPayloadError("frame body is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise FrameProtocolError("JSON-RPC frame body must be an object")
    return value
