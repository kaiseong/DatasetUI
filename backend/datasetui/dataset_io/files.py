"""Symlink-safe file reads and atomic JSON writes inside dataset trees."""

from __future__ import annotations

import json
import os
import stat
import uuid
from pathlib import Path
from typing import Any

from datasetui.transform_errors import CurationTransformError

MAX_METADATA_BYTES = 256 * 1024 * 1024


def copy_open_regular_file(
    descriptor: int,
    destination: Path,
    *,
    source_metadata: os.stat_result,
) -> None:
    identity = (
        source_metadata.st_dev,
        source_metadata.st_ino,
        source_metadata.st_size,
        source_metadata.st_mtime_ns,
        source_metadata.st_ctime_ns,
    )
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        with destination.open("xb") as output:
            while chunk := os.read(descriptor, 1024 * 1024):
                output.write(chunk)
        after = os.fstat(descriptor)
        if identity != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise CurationTransformError(
                "Source episode video changed while it was being copied"
            )
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def safe_dataset_root(nas_root: Path, storage_area: str, relative_path: str) -> Path:
    root = real_directory(nas_root / storage_area)
    current = root
    for component in Path(relative_path).parts:
        if component in {"", ".", ".."}:
            raise CurationTransformError("Unsafe dataset path")
        current = current / component
        if current.is_symlink() or not current.is_dir():
            raise CurationTransformError("Dataset source is unavailable")
    return current


def safe_child(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise CurationTransformError("Unsafe dataset file path")
    return root / path


def real_directory(path: Path) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise CurationTransformError("Dataset storage is unavailable")
    return path


def read_json(path: Path) -> dict[str, Any]:
    value = decode_json_object(read_regular_bytes(path, max_bytes=MAX_METADATA_BYTES))
    return value


def decode_json_object(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CurationTransformError("Dataset metadata is invalid") from exc
    if not isinstance(value, dict):
        raise CurationTransformError("Dataset metadata is invalid")
    return value


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    try:
        text = read_regular_bytes(path, max_bytes=MAX_METADATA_BYTES).decode("utf-8")
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CurationTransformError("Dataset metadata is invalid") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise CurationTransformError("Dataset metadata is invalid")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )


def write_json_lines(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        write_json(temporary, value)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def require_regular_file(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CurationTransformError("Dataset file is unavailable") from exc
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise CurationTransformError("Dataset contains an unsafe file entry")


def read_regular_bytes(path: Path, *, max_bytes: int) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise CurationTransformError("Dataset file is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > max_bytes:
            raise CurationTransformError("Dataset metadata is invalid")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > max_bytes:
            raise CurationTransformError("Dataset metadata is invalid")
        return raw
    finally:
        os.close(descriptor)
