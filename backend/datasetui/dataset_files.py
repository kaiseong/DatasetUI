from __future__ import annotations

import mimetypes
import os
import stat
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


READ_CHUNK_BYTES = 1024 * 1024
MAX_REQUEST_PATH_BYTES = 4096


class DatasetFilePathError(ValueError):
    pass


class DatasetFileUnavailableError(FileNotFoundError):
    pass


class DatasetRangeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class OpenDatasetFile:
    descriptor: int
    size: int
    content_type: str


@dataclass(frozen=True, slots=True)
class ByteRange:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def open_dataset_file(
    *,
    nas_root: Path,
    storage_area: str,
    dataset_relative_path: str,
    requested_path: str,
) -> OpenDatasetFile:
    if storage_area not in {"raw", "derived"}:
        raise DatasetFileUnavailableError("dataset storage is unavailable")

    dataset_components = _safe_components(dataset_relative_path)
    file_components = _safe_components(requested_path)
    if not dataset_components or not file_components:
        raise DatasetFilePathError("invalid dataset file path")

    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    directory_fd: int | None = None
    file_fd: int | None = None
    try:
        directory_fd = os.open(nas_root, directory_flags)
        for component in (storage_area, *dataset_components, *file_components[:-1]):
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd

        file_fd = os.open(file_components[-1], file_flags, dir_fd=directory_fd)
        file_stat = os.fstat(file_fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise DatasetFileUnavailableError("dataset file is unavailable")
        result = OpenDatasetFile(
            descriptor=file_fd,
            size=file_stat.st_size,
            content_type=mimetypes.guess_type(file_components[-1])[0]
            or "application/octet-stream",
        )
        file_fd = None
        return result
    except DatasetFileUnavailableError:
        raise
    except OSError as exc:
        raise DatasetFileUnavailableError("dataset file is unavailable") from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def parse_byte_range(value: str | None, size: int) -> ByteRange | None:
    if value is None:
        return None
    if not value.startswith("bytes=") or "," in value:
        raise DatasetRangeError("unsupported byte range")

    spec = value[6:].strip()
    if "-" not in spec:
        raise DatasetRangeError("invalid byte range")
    start_text, end_text = spec.split("-", 1)
    try:
        if start_text:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
            if start < 0 or end < start or start >= size:
                raise DatasetRangeError("unsatisfiable byte range")
            return ByteRange(start=start, end=min(end, size - 1))

        suffix_length = int(end_text)
        if suffix_length <= 0 or size == 0:
            raise DatasetRangeError("unsatisfiable byte range")
        length = min(suffix_length, size)
        return ByteRange(start=size - length, end=size - 1)
    except ValueError as exc:
        if isinstance(exc, DatasetRangeError):
            raise
        raise DatasetRangeError("invalid byte range") from exc


def iter_open_file(
    descriptor: int,
    *,
    start: int,
    length: int,
    chunk_size: int = READ_CHUNK_BYTES,
) -> Iterator[bytes]:
    os.lseek(descriptor, start, os.SEEK_SET)
    remaining = length
    while remaining > 0:
        chunk = os.read(descriptor, min(chunk_size, remaining))
        if not chunk:
            break
        remaining -= len(chunk)
        yield chunk


def _safe_components(value: str) -> tuple[str, ...]:
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or "\x00" in value
        or len(value.encode("utf-8")) > MAX_REQUEST_PATH_BYTES
    ):
        raise DatasetFilePathError("invalid dataset file path")
    components = tuple(value.split("/"))
    if any(component in {"", ".", ".."} for component in components):
        raise DatasetFilePathError("invalid dataset file path")
    return components
