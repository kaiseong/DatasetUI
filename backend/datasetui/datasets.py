from __future__ import annotations

import hashlib
import json
import logging
import os
import stat as stat_module
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from datasetui.content_integrity import ContentIntegrityError, dataset_tree_identity


StorageArea = Literal["raw", "derived"]
SUPPORTED_VERSIONS = frozenset({"v2.0", "v2.1", "v3.0"})
MAX_INFO_BYTES = 2 * 1024 * 1024
SKIPPED_DIRECTORIES = frozenset(
    {".git", ".cache", ".datasetui-trash", "__pycache__"}
)
logger = logging.getLogger("datasetui.datasets")


class DatasetRootUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DatasetCandidate:
    storage_area: StorageArea
    relative_path: str
    name: str
    codebase_version: str | None
    readiness: str
    robot_type: str | None
    total_episodes: int | None
    total_frames: int | None
    total_tasks: int | None
    fps: float | None
    fingerprint: str
    info_mtime_ns: int
    info_size: int
    scan_error: str | None

    def as_record(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


def scan_storage_area(
    nas_root: Path,
    storage_area: StorageArea,
    *,
    max_depth: int,
    on_discovered: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
    if max_depth < 1 or max_depth > 20:
        raise ValueError("dataset scan max depth must be between 1 and 20")

    configured_root = nas_root / storage_area
    if (
        configured_root.is_symlink()
        or not configured_root.exists()
        or not configured_root.is_dir()
    ):
        raise DatasetRootUnavailableError(
            f"Dataset storage area is unavailable: {storage_area}"
        )
    area_root = configured_root.resolve(strict=True)
    candidates: list[dict[str, Any]] = []
    traversal_errors: list[OSError] = []

    for current, directory_names, file_names in os.walk(
        area_root,
        topdown=True,
        followlinks=False,
        onerror=traversal_errors.append,
    ):
        current_path = Path(current)
        depth = len(current_path.relative_to(area_root).parts)
        directory_names[:] = sorted(
            name
            for name in directory_names
            if not _skip_directory(name)
            and not (current_path / name).is_symlink()
            and depth < max_depth
        )
        if current_path.name != "meta" or "info.json" not in file_names:
            continue

        dataset_root = current_path.parent
        try:
            relative_path = dataset_root.relative_to(area_root).as_posix()
        except ValueError:
            continue
        if relative_path == ".":
            relative_path = ""
        candidates.append(
            inspect_dataset(
                area_root=area_root,
                storage_area=storage_area,
                relative_path=relative_path,
            ).as_record()
        )
        if on_discovered is not None:
            on_discovered(len(candidates))
        directory_names[:] = []

    if traversal_errors:
        raise DatasetRootUnavailableError(
            f"Dataset storage area could not be read completely: {storage_area}"
        )
    return candidates


def inspect_dataset(
    *,
    area_root: Path,
    storage_area: StorageArea,
    relative_path: str,
) -> DatasetCandidate:
    dataset_root = area_root / relative_path
    error: str | None = None
    metadata: dict[str, Any] = {}
    raw = b""
    info_mtime_ns = 0
    info_size = 0

    try:
        raw, info_mtime_ns, info_size = _read_info_safely(area_root, relative_path)
        decoded = json.loads(raw)
        if not isinstance(decoded, dict):
            raise ValueError("top-level JSON value must be an object")
        metadata = decoded
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        error = f"Invalid meta/info.json: {exc}"
    except OSError:
        logger.warning(
            "unable to read dataset metadata in %s at relative path %r",
            storage_area,
            relative_path,
            exc_info=True,
        )
        error = "Unable to read meta/info.json safely"

    version = _optional_string(metadata.get("codebase_version"))
    if error:
        readiness = "invalid"
    elif version not in SUPPORTED_VERSIONS:
        readiness = "unsupported"
        error = "Unsupported or missing codebase_version"
    elif not isinstance(metadata.get("features"), dict):
        readiness = "invalid"
        error = "meta/info.json is missing the features object"
    elif (dataset_root / "data").is_symlink() or not (dataset_root / "data").is_dir():
        readiness = "incomplete"
        error = "Dataset data directory is missing"
    else:
        readiness = "ready"

    if readiness == "ready":
        # Stat-only walk: jobs refuse unsafe trees, so flag them at scan time.
        try:
            dataset_tree_identity(dataset_root)
        except ContentIntegrityError:
            logger.warning(
                "unsafe entry in dataset tree in %s at relative path %r",
                storage_area,
                relative_path,
                exc_info=True,
            )
            readiness = "invalid"
            error = "Dataset contains a symlink or an unsafe file"

    fingerprint_source = raw or error.encode()
    display_name = _display_name(relative_path, storage_area)
    return DatasetCandidate(
        storage_area=storage_area,
        relative_path=relative_path,
        name=display_name,
        codebase_version=version,
        readiness=readiness,
        robot_type=_optional_string(metadata.get("robot_type")),
        total_episodes=_optional_nonnegative_int(metadata.get("total_episodes")),
        total_frames=_optional_nonnegative_int(metadata.get("total_frames")),
        total_tasks=_optional_nonnegative_int(metadata.get("total_tasks")),
        fps=_optional_positive_number(metadata.get("fps")),
        fingerprint=hashlib.sha256(fingerprint_source).hexdigest(),
        info_mtime_ns=info_mtime_ns,
        info_size=info_size,
        scan_error=error,
    )


def _read_info_safely(area_root: Path, relative_path: str) -> tuple[bytes, int, int]:
    relative = Path(relative_path)
    if relative.is_absolute() or any(part in {".", ".."} for part in relative.parts):
        raise OSError("unsafe relative dataset path")

    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    directory_fd = os.open(area_root, directory_flags)
    try:
        for component in (*relative.parts, "meta"):
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd

        info_fd = os.open("info.json", file_flags, dir_fd=directory_fd)
        try:
            info_stat = os.fstat(info_fd)
            if not stat_module.S_ISREG(info_stat.st_mode):
                raise OSError("metadata is not a regular file")
            if info_stat.st_size > MAX_INFO_BYTES:
                raise OSError("metadata exceeds size limit")

            chunks: list[bytes] = []
            remaining = MAX_INFO_BYTES + 1
            while remaining:
                chunk = os.read(info_fd, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            if len(raw) > MAX_INFO_BYTES:
                raise OSError("metadata exceeds size limit")
            return raw, info_stat.st_mtime_ns, info_stat.st_size
        finally:
            os.close(info_fd)
    finally:
        os.close(directory_fd)


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _optional_positive_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return float(value)


def _skip_directory(name: str) -> bool:
    return name in SKIPPED_DIRECTORIES or name.startswith(".incoming-")


def _display_name(relative_path: str, storage_area: StorageArea) -> str:
    parts = Path(relative_path).parts
    if (
        storage_area == "raw"
        and len(parts) == 5
        and parts[0] == "hf"
        and parts[1] == "rainbowrobotics"
        and parts[3] == "revisions"
    ):
        return parts[2]
    return Path(relative_path).name if relative_path else storage_area
