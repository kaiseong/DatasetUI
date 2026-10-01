from __future__ import annotations

import hashlib
import os
import stat
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any


class ContentIntegrityError(OSError):
    pass


_FILE_DIGEST_CACHE: OrderedDict[tuple, tuple[int, str]] = OrderedDict()
_FILE_DIGEST_CACHE_LIMIT = 200_000
_FILE_DIGEST_CACHE_LOCK = threading.Lock()


def dataset_content_fingerprint(root: Path, *, reuse_file_digests: bool = False) -> str:
    return str(
        dataset_content_manifest(root, reuse_file_digests=reuse_file_digests)[
            "tree_sha256"
        ]
    )


def _file_identity(path: Path, metadata: os.stat_result) -> tuple:
    return (
        str(path),
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def dataset_content_manifest(
    root: Path, *, reuse_file_digests: bool = False
) -> dict[str, Any]:
    """Hash every regular file.

    ``reuse_file_digests`` reuses a digest computed earlier in this process only
    while the file's device, inode, size, mtime and ctime are all unchanged. Any
    write, replacement or metadata change produces a new ctime and forces a full
    re-hash, so repeated verification of an immutable source stays cheap.
    """
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise ContentIntegrityError("dataset root is unavailable") from exc
    if root.is_symlink() or not stat.S_ISDIR(root_metadata.st_mode):
        raise ContentIntegrityError("dataset root is unsafe")

    entries: list[tuple[Path, os.stat_result]] = []
    traversal_errors: list[OSError] = []
    for current, directories, files in os.walk(
        root, followlinks=False, onerror=traversal_errors.append
    ):
        current_path = Path(current)
        directories.sort()
        for name in directories:
            path = current_path / name
            try:
                metadata = path.lstat()
            except OSError as exc:
                raise ContentIntegrityError("dataset tree changed during scan") from exc
            if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
                raise ContentIntegrityError("dataset tree contains an unsafe directory")
        for name in sorted(files):
            path = current_path / name
            try:
                metadata = path.lstat()
            except OSError as exc:
                raise ContentIntegrityError("dataset tree changed during scan") from exc
            if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                raise ContentIntegrityError("dataset tree contains an unsafe file")
            entries.append((path, metadata))
    if traversal_errors:
        raise ContentIntegrityError("dataset tree could not be read completely")

    digest = hashlib.sha256()
    total_bytes = 0
    for path, metadata in sorted(
        entries, key=lambda item: item[0].relative_to(root).as_posix()
    ):
        cached = None
        if reuse_file_digests:
            with _FILE_DIGEST_CACHE_LOCK:
                cached = _FILE_DIGEST_CACHE.get(_file_identity(path, metadata))
        if cached is not None:
            size, file_sha256 = cached
        else:
            size, file_sha256, identity = _hash_regular_file(path)
            if reuse_file_digests:
                with _FILE_DIGEST_CACHE_LOCK:
                    _FILE_DIGEST_CACHE[identity] = (size, file_sha256)
                    _FILE_DIGEST_CACHE.move_to_end(identity)
                    while len(_FILE_DIGEST_CACHE) > _FILE_DIGEST_CACHE_LIMIT:
                        _FILE_DIGEST_CACHE.popitem(last=False)
        relative = path.relative_to(root).as_posix()
        digest.update(f"{relative}\0{size}\0{file_sha256}\n".encode())
        total_bytes += size
    return {
        "tree_sha256": digest.hexdigest(),
        "file_count": len(entries),
        "total_bytes": total_bytes,
    }


def _hash_regular_file(path: Path) -> tuple[int, str, tuple]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise ContentIntegrityError("dataset file is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ContentIntegrityError("dataset tree contains an unsafe file")
        file_digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            file_digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if (
            identity_before != identity_after
            or size != after.st_size
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            raise ContentIntegrityError("dataset file changed during scan")
        return size, file_digest.hexdigest(), _file_identity(path, before)
    finally:
        os.close(descriptor)
