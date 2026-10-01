"""Filesystem safety helpers: symlink-free regular paths and output locks."""

from __future__ import annotations

import fcntl
import hashlib
import stat
from contextlib import contextmanager
from pathlib import Path

from datasetui.config import Settings
from datasetui.segmentation.errors import SegmentationError


def _safe_regular_path(root: Path, path: Path) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise SegmentationError("Dataset file path is unsafe") from exc
    current = root
    for index, component in enumerate(relative.parts):
        if component in {"", ".", ".."}:
            raise SegmentationError("Dataset file path is unsafe")
        current = current / component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise SegmentationError("Dataset file is unavailable") from exc
        if current.is_symlink():
            raise SegmentationError("Dataset file path is unsafe")
        final = index == len(relative.parts) - 1
        if final and not stat.S_ISREG(metadata.st_mode):
            raise SegmentationError("Dataset file is unsafe")
        if not final and not stat.S_ISDIR(metadata.st_mode):
            raise SegmentationError("Dataset file path is unsafe")
    return current


def _safe_directory_path(root: Path, path: Path) -> Path:
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
def _output_lock(settings: Settings, output_name: str):
    lock_root = settings.nas_root / "manifests/segmentation/.locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_name = hashlib.sha256(output_name.encode()).hexdigest() + ".lock"
    with (lock_root / lock_name).open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
