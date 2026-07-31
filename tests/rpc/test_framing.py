"""Contract tests for the stdio Content-Length transport."""

from __future__ import annotations

import io
import json

import pytest

from lerobot_dataset_editor.rpc.framing import (
    MAX_CONTENT_LENGTH,
    FrameProtocolError,
    FrameTooLargeError,
    encode_frame,
    read_frame,
)


def test_encode_frame_uses_utf8_byte_length() -> None:
    message = {"jsonrpc": "2.0", "id": 1, "method": "echo", "params": {"text": "로봇"}}
    frame = encode_frame(message)
    header, payload = frame.split(b"\r\n\r\n", 1)
    assert header == f"Content-Length: {len(payload)}".encode("ascii")
    assert json.loads(payload) == message


def test_read_frame_round_trips_and_accepts_unknown_headers() -> None:
    payload = json.dumps({"jsonrpc": "2.0", "id": "abc", "result": {"ok": True}}).encode()
    stream = io.BytesIO(
        b"X-DatasetUI: 1\r\nContent-Type: application/json\r\n"
        + f"Content-Length: {len(payload)}\r\n\r\n".encode()
        + payload
    )
    assert read_frame(stream) == {"jsonrpc": "2.0", "id": "abc", "result": {"ok": True}}


def test_read_frame_returns_none_on_clean_eof() -> None:
    assert read_frame(io.BytesIO()) is None


def test_read_frame_rejects_missing_or_duplicate_content_length() -> None:
    with pytest.raises(FrameProtocolError, match="Content-Length"):
        read_frame(io.BytesIO(b"X-Test: true\r\n\r\n{}"))
    with pytest.raises(FrameProtocolError, match="duplicate"):
        read_frame(
            io.BytesIO(b"Content-Length: 2\r\nContent-Length: 2\r\n\r\n{}")
        )


def test_read_frame_rejects_invalid_and_oversized_lengths_before_body_read() -> None:
    with pytest.raises(FrameProtocolError, match="non-negative integer"):
        read_frame(io.BytesIO(b"Content-Length: nope\r\n\r\n"))
    with pytest.raises(FrameTooLargeError, match=str(MAX_CONTENT_LENGTH)):
        read_frame(
            io.BytesIO(f"Content-Length: {MAX_CONTENT_LENGTH + 1}\r\n\r\n".encode())
        )


def test_read_frame_rejects_truncated_or_non_object_json() -> None:
    with pytest.raises(FrameProtocolError, match="truncated"):
        read_frame(io.BytesIO(b"Content-Length: 5\r\n\r\n{}"))
    for payload in (b"not json", b"[]"):
        with pytest.raises(FrameProtocolError):
            read_frame(
                io.BytesIO(f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload)
            )
