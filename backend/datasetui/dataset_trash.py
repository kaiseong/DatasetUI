"""Filesystem-safe, recoverable dataset trash moves."""

from __future__ import annotations

import ctypes
import errno
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


TRASH_DIRECTORY = ".datasetui-trash"
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
_RENAME_NOREPLACE = 1
_before_rename_hook = None


class DatasetTrashPathError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def registered_dataset_identity(
    nas_root: Path, *, storage_area: str, relative_path: str
) -> tuple[int, int]:
    area_root = _area_root(nas_root, storage_area)
    relative = _safe_relative(relative_path, allow_trash=False)
    return _location_identity(area_root, relative, code="trash_path_conflict")


def dataset_location_identity(
    nas_root: Path, record: dict[str, Any], *, trashed: bool
) -> tuple[int, int]:
    area_root = _area_root(nas_root, record["storage_area"])
    value = (
        record["trash_relative_path"] if trashed else record["original_relative_path"]
    )
    relative = _safe_relative(value, allow_trash=trashed)
    if trashed and relative != Path(TRASH_DIRECTORY) / record["dataset_id"]:
        raise DatasetTrashPathError("trash_recovery_required")
    return _location_identity(area_root, relative, code="trash_recovery_required")


def identity_matches(record: dict[str, Any], identity: tuple[int, int]) -> bool:
    return (
        record.get("source_device") is not None
        and record.get("source_inode") is not None
        and identity == (record["source_device"], record["source_inode"])
    )


def purge_dataset_from_trash(
    nas_root: Path, record: dict[str, Any], *, lease_check
) -> None:
    """Delete only the pinned directory, traversing descriptors without symlinks."""
    area = _area_root(nas_root, record["storage_area"])
    name = record["dataset_id"]
    if (
        _safe_relative(record["trash_relative_path"], allow_trash=True)
        != Path(TRASH_DIRECTORY) / name
    ):
        raise DatasetTrashPathError("trash_path_conflict")

    def mount_id(descriptor: int) -> str:
        # Linux deployment: reject nested mounts, including same-device bind
        # mounts that st_dev checks alone cannot distinguish.
        for line in Path(f"/proc/self/fdinfo/{descriptor}").read_text().splitlines():
            if line.startswith("mnt_id:"):
                return line.split(":", 1)[1].strip()
        raise DatasetTrashPathError("trash_recovery_required")

    def empty_directory(descriptor: int, expected_mount: str, check_location):
        check_location()
        with os.scandir(descriptor) as entries:
            names = [entry.name for entry in entries]
        for entry_name in names:
            lease_check()
            check_location()
            before = os.stat(entry_name, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISDIR(before.st_mode):
                child = os.open(entry_name, _DIRECTORY_FLAGS, dir_fd=descriptor)
                try:
                    if (
                        _fd_identity(child) != (before.st_dev, before.st_ino)
                        or mount_id(child) != expected_mount
                    ):
                        raise DatasetTrashPathError("trash_recovery_required")

                    def check_child():
                        check_location()
                        _assert_leaf_identity(
                            descriptor,
                            entry_name,
                            child,
                            code="trash_recovery_required",
                        )

                    empty_directory(child, expected_mount, check_child)
                    check_child()
                    _assert_leaf_identity(
                        descriptor, entry_name, child, code="trash_recovery_required"
                    )
                    os.rmdir(entry_name, dir_fd=descriptor)
                finally:
                    os.close(child)
            else:
                # Unlink the directory entry itself; never follow a link.
                check_location()
                os.unlink(entry_name, dir_fd=descriptor)

    with _open_area(area, code="trash_recovery_required") as area_fd:
        trash_fd = os.open(TRASH_DIRECTORY, _DIRECTORY_FLAGS, dir_fd=area_fd)
        try:
            source_fd = _open_leaf_directory(
                trash_fd, name, code="trash_recovery_required"
            )
            try:
                if not identity_matches(record, _fd_identity(source_fd)):
                    raise DatasetTrashPathError("trash_recovery_required")
                _assert_fd_path(
                    area / TRASH_DIRECTORY, trash_fd, code="trash_recovery_required"
                )
                lease_check()
                if mount_id(source_fd) != mount_id(trash_fd):
                    raise DatasetTrashPathError("trash_recovery_required")

                def check_root():
                    _assert_fd_path(area, area_fd, code="trash_recovery_required")
                    _assert_fd_path(
                        area / TRASH_DIRECTORY, trash_fd, code="trash_recovery_required"
                    )
                    _assert_leaf_identity(
                        trash_fd, name, source_fd, code="trash_recovery_required"
                    )

                empty_directory(source_fd, mount_id(source_fd), check_root)
                check_root()
                _assert_leaf_identity(
                    trash_fd, name, source_fd, code="trash_recovery_required"
                )
                os.rmdir(name, dir_fd=trash_fd)
            finally:
                os.close(source_fd)
        finally:
            os.close(trash_fd)


def move_dataset_to_trash(
    nas_root: Path,
    record: dict[str, Any],
    *,
    registered_locations: list[dict[str, str]],
) -> None:
    area_root = _area_root(nas_root, record["storage_area"])
    relative = _safe_relative(record["original_relative_path"], allow_trash=False)

    for other in registered_locations:
        if other["storage_area"] != record["storage_area"]:
            continue
        other_relative = _safe_relative(other["relative_path"], allow_trash=False)
        if _paths_overlap(relative, other_relative):
            raise DatasetTrashPathError("trash_path_conflict")
        if _location_identity(
            area_root, other_relative, code="trash_path_conflict"
        ) == (record["source_device"], record["source_inode"]):
            raise DatasetTrashPathError("trash_path_conflict")

    with _open_area(area_root, code="trash_path_conflict") as area_fd:
        try:
            os.mkdir(TRASH_DIRECTORY, mode=0o700, dir_fd=area_fd)
        except FileExistsError:
            pass
        try:
            trash_fd = os.open(TRASH_DIRECTORY, _DIRECTORY_FLAGS, dir_fd=area_fd)
        except OSError as exc:
            raise DatasetTrashPathError("trash_path_conflict") from exc
        try:
            with _open_parent(
                area_root, area_fd, relative, code="trash_path_conflict"
            ) as (source_parent_fd, source_parent_path, source_name):
                source_fd = _open_leaf_directory(
                    source_parent_fd, source_name, code="trash_path_conflict"
                )
                try:
                    source_identity = _fd_identity(source_fd)
                    if not identity_matches(record, source_identity):
                        raise DatasetTrashPathError("trash_recovery_required")
                    if os.fstat(source_fd).st_dev != os.fstat(trash_fd).st_dev:
                        raise DatasetTrashPathError("trash_path_conflict")
                    _call_before_rename(
                        area_root / relative,
                        area_root / TRASH_DIRECTORY / record["dataset_id"],
                    )
                    _assert_fd_path(
                        area_root / TRASH_DIRECTORY,
                        trash_fd,
                        code="trash_path_conflict",
                    )
                    _assert_fd_path(
                        source_parent_path,
                        source_parent_fd,
                        code="trash_path_conflict",
                    )
                    _assert_leaf_identity(
                        source_parent_fd,
                        source_name,
                        source_fd,
                        code="trash_path_conflict",
                    )
                    _rename_noreplace(
                        source_parent_fd,
                        source_name,
                        trash_fd,
                        record["dataset_id"],
                        code="trash_path_conflict",
                    )
                    destination_fd = _open_leaf_directory(
                        trash_fd,
                        record["dataset_id"],
                        code="trash_recovery_required",
                    )
                    try:
                        if _fd_identity(destination_fd) != source_identity:
                            raise DatasetTrashPathError("trash_recovery_required")
                    finally:
                        os.close(destination_fd)
                finally:
                    os.close(source_fd)
        finally:
            os.close(trash_fd)


def restore_dataset_from_trash(nas_root: Path, record: dict[str, Any]) -> None:
    area_root = _area_root(nas_root, record["storage_area"])
    relative = _safe_relative(record["original_relative_path"], allow_trash=False)
    trash_relative = _safe_relative(record["trash_relative_path"], allow_trash=True)
    if trash_relative != Path(TRASH_DIRECTORY) / record["dataset_id"]:
        raise DatasetTrashPathError("trash_path_conflict")

    with _open_area(area_root, code="trash_recovery_required") as area_fd:
        try:
            trash_fd = os.open(TRASH_DIRECTORY, _DIRECTORY_FLAGS, dir_fd=area_fd)
        except OSError as exc:
            raise DatasetTrashPathError("trash_recovery_required") from exc
        try:
            source_fd = _open_leaf_directory(
                trash_fd, record["dataset_id"], code="trash_recovery_required"
            )
            try:
                source_identity = _fd_identity(source_fd)
                if not identity_matches(record, source_identity):
                    raise DatasetTrashPathError("trash_recovery_required")
                with _open_parent(
                    area_root, area_fd, relative, code="restore_path_occupied"
                ) as (destination_parent_fd, destination_parent_path, destination_name):
                    if (
                        os.fstat(source_fd).st_dev
                        != os.fstat(destination_parent_fd).st_dev
                    ):
                        raise DatasetTrashPathError("trash_path_conflict")
                    _call_before_rename(
                        area_root / trash_relative, area_root / relative
                    )
                    _assert_fd_path(
                        area_root / TRASH_DIRECTORY,
                        trash_fd,
                        code="trash_recovery_required",
                    )
                    _assert_fd_path(
                        destination_parent_path,
                        destination_parent_fd,
                        code="restore_path_occupied",
                    )
                    _assert_leaf_identity(
                        trash_fd,
                        record["dataset_id"],
                        source_fd,
                        code="trash_recovery_required",
                    )
                    _rename_noreplace(
                        trash_fd,
                        record["dataset_id"],
                        destination_parent_fd,
                        destination_name,
                        code="restore_path_occupied",
                    )
                    destination_fd = _open_leaf_directory(
                        destination_parent_fd,
                        destination_name,
                        code="trash_recovery_required",
                    )
                    try:
                        if _fd_identity(destination_fd) != source_identity:
                            raise DatasetTrashPathError("trash_recovery_required")
                    finally:
                        os.close(destination_fd)
            finally:
                os.close(source_fd)
        finally:
            os.close(trash_fd)


def dataset_trash_locations(
    nas_root: Path, record: dict[str, Any]
) -> tuple[bool, bool]:
    area_root = _area_root(nas_root, record["storage_area"])
    original = _safe_relative(record["original_relative_path"], allow_trash=False)
    trashed = _safe_relative(record["trash_relative_path"], allow_trash=True)
    if trashed != Path(TRASH_DIRECTORY) / record["dataset_id"]:
        raise DatasetTrashPathError("trash_recovery_required")
    return (
        _location_exists(area_root, original, code="trash_recovery_required"),
        _location_exists(area_root, trashed, code="trash_recovery_required"),
    )


def _area_root(nas_root: Path, storage_area: str) -> Path:
    if storage_area not in {"raw", "derived"}:
        raise DatasetTrashPathError("trash_path_conflict")
    configured = nas_root / storage_area
    if configured.is_symlink() or not configured.is_dir():
        raise DatasetTrashPathError("trash_path_conflict")
    try:
        resolved = configured.resolve(strict=True)
    except OSError as exc:
        raise DatasetTrashPathError("trash_path_conflict") from exc
    if resolved != configured.absolute():
        raise DatasetTrashPathError("trash_path_conflict")
    return resolved


def _safe_relative(value: str, *, allow_trash: bool) -> Path:
    relative = Path(value)
    if (
        not value
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise DatasetTrashPathError("trash_path_conflict")
    if not allow_trash and relative.parts[0] == TRASH_DIRECTORY:
        raise DatasetTrashPathError("trash_path_conflict")
    return relative


@contextmanager
def _open_area(area_root: Path, *, code: str) -> Iterator[int]:
    try:
        fd = os.open(area_root, _DIRECTORY_FLAGS)
    except OSError as exc:
        raise DatasetTrashPathError(code) from exc
    try:
        _assert_fd_path(area_root, fd, code=code)
        yield fd
    finally:
        os.close(fd)


@contextmanager
def _open_parent(
    area_root: Path, area_fd: int, relative: Path, *, code: str
) -> Iterator[tuple[int, Path, str]]:
    current_fd = os.dup(area_fd)
    current_path = area_root
    try:
        for component in relative.parts[:-1]:
            try:
                next_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=current_fd)
            except OSError as exc:
                raise DatasetTrashPathError(code) from exc
            os.close(current_fd)
            current_fd = next_fd
            current_path /= component
            _assert_fd_path(current_path, current_fd, code=code)
        yield current_fd, current_path, relative.parts[-1]
    finally:
        os.close(current_fd)


def _open_leaf_directory(parent_fd: int, name: str, *, code: str) -> int:
    try:
        return os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise DatasetTrashPathError(code) from exc


def _location_identity(
    area_root: Path, relative: Path, *, code: str
) -> tuple[int, int]:
    with _open_area(area_root, code=code) as area_fd:
        with _open_parent(area_root, area_fd, relative, code=code) as (
            parent_fd,
            _parent_path,
            name,
        ):
            leaf_fd = _open_leaf_directory(parent_fd, name, code=code)
            try:
                return _fd_identity(leaf_fd)
            finally:
                os.close(leaf_fd)


def _location_exists(area_root: Path, relative: Path, *, code: str) -> bool:
    try:
        _location_identity(area_root, relative, code=code)
        return True
    except DatasetTrashPathError as exc:
        if _caused_by_missing_path(exc):
            return False
        raise


def _caused_by_missing_path(exc: BaseException) -> bool:
    cause: BaseException | None = exc
    while cause is not None:
        if isinstance(cause, OSError) and cause.errno == errno.ENOENT:
            return True
        cause = cause.__cause__
    return False


def _fd_identity(fd: int) -> tuple[int, int]:
    info = os.fstat(fd)
    return info.st_dev, info.st_ino


def _assert_fd_path(path: Path, fd: int, *, code: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise DatasetTrashPathError(code) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise DatasetTrashPathError(code)
    if (info.st_dev, info.st_ino) != _fd_identity(fd):
        raise DatasetTrashPathError(code)


def _assert_leaf_identity(
    parent_fd: int, name: str, leaf_fd: int, *, code: str
) -> None:
    try:
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise DatasetTrashPathError(code) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise DatasetTrashPathError(code)
    if (info.st_dev, info.st_ino) != _fd_identity(leaf_fd):
        raise DatasetTrashPathError(code)


def _rename_noreplace(
    source_parent_fd: int,
    source_name: str,
    destination_parent_fd: int,
    destination_name: str,
    *,
    code: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise DatasetTrashPathError(code)
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_parent_fd,
        os.fsencode(source_name),
        destination_parent_fd,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise DatasetTrashPathError(code)
    raise DatasetTrashPathError("trash_path_conflict")


def _call_before_rename(source: Path, destination: Path) -> None:
    if _before_rename_hook is not None:
        _before_rename_hook(source, destination)


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents
