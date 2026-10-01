"""Filesystem safety helpers: symlink-free regular paths and output locks."""

from __future__ import annotations

import fcntl
import hashlib
import stat
from contextlib import contextmanager
from pathlib import Path

from datasetui.config import Settings
from datasetui.segmentation.errors import SegmentationError

DATASET_FILE_MESSAGES = {
    "invalid": "Dataset file path is unsafe",
    "unavailable": "Dataset file is unavailable",
    "unsafe_path": "Dataset file path is unsafe",
    "unsafe_file": "Dataset file is unsafe",
}


def safe_regular_path(
    root: Path,
    path: Path,
    *,
    error: type[Exception] = SegmentationError,
    messages: dict[str, str] = DATASET_FILE_MESSAGES,
) -> Path:
    """`path` below `root` as a regular file reached without any symlink.

    Callers choose the exception type and wording so each boundary keeps its
    own error contract (job error vs. HTTP 409).
    """
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise error(messages["invalid"]) from exc
    current = root
    for index, component in enumerate(relative.parts):
        if component in {"", ".", ".."}:
            raise error(messages["invalid"])
        current = current / component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise error(messages["unavailable"]) from exc
        final = index == len(relative.parts) - 1
        if current.is_symlink():
            raise error(messages["unsafe_path"])
        if final and not stat.S_ISREG(metadata.st_mode):
            raise error(messages["unsafe_file"])
        if not final and not stat.S_ISDIR(metadata.st_mode):
            raise error(messages["unsafe_path"])
    return current


def safe_directory_path(root: Path, path: Path) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise SegmentationError("Dataset directory path is unsafe") from exc
    current = root
    for component in relative.parts:
        current = current / component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise SegmentationError("Dataset directory is unavailable") from exc
        if current.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise SegmentationError("Dataset directory path is unsafe")
    return current


@contextmanager
def output_lock(settings: Settings, output_name: str):
    lock_root = settings.nas_root / "manifests/segmentation/.locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_name = hashlib.sha256(output_name.encode()).hexdigest() + ".lock"
    with (lock_root / lock_name).open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
