"""Give upstream only private byte copies, never writable source hardlinks."""

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import stat
import tempfile

from datasetui.transform_errors import CurationTransformError


def _identity(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _copy_directory(descriptor, destination, manifest, relative, on_progress):
    for name in sorted(os.listdir(descriptor)):
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        key = f"{relative}/{name}" if relative else name
        if stat.S_ISDIR(metadata.st_mode):
            child = os.open(
                name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            try:
                (destination / name).mkdir()
                _copy_directory(child, destination / name, manifest, key, on_progress)
            finally:
                os.close(child)
        elif stat.S_ISREG(metadata.st_mode):
            child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
            digest = hashlib.sha256()
            try:
                before = os.fstat(child)
                if not stat.S_ISREG(before.st_mode) or _identity(before) != _identity(
                    metadata
                ):
                    raise CurationTransformError(
                        "Source changed while preparing private input"
                    )
                with (destination / name).open("xb") as output:
                    while block := os.read(child, 1024 * 1024):
                        digest.update(block)
                        output.write(block)
                if _identity(before) != _identity(os.fstat(child)):
                    raise CurationTransformError(
                        "Source changed while preparing private input"
                    )
            finally:
                os.close(child)
            manifest[key] = digest.hexdigest()
            if on_progress:
                on_progress(
                    {
                        "stage": "preparing",
                        "completed": len(manifest),
                        "total": 0,
                        "unit": "files",
                        "current_item": f"원본 보호용 복사 · {key}",
                    }
                )
        else:
            raise CurationTransformError("Source contains a symlink or special file")


@contextmanager
def private_sources(roots: list[Path], parent: Path, *, on_progress=None):
    from datasetui.official_operations import _inventory

    with tempfile.TemporaryDirectory(
        prefix=".official-inputs-", dir=parent
    ) as temporary:
        copies, manifests = [], []
        for index, root in enumerate(roots):
            before = _inventory(root)
            copy = Path(temporary) / str(index)
            copy.mkdir()
            descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                copied = {}
                _copy_directory(descriptor, copy, copied, "", on_progress)
            finally:
                os.close(descriptor)
            if copied != before or _inventory(root) != before:
                raise CurationTransformError(
                    "Source changed during protected input capture"
                )
            copies.append(copy)
            manifests.append(before)
        try:
            yield copies
        finally:
            if manifests != [_inventory(root) for root in roots]:
                raise CurationTransformError(
                    "Original source changed during processing; refuse publication"
                )
