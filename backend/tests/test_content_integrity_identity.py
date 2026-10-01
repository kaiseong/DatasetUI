from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from datasetui.content_integrity import ContentIntegrityError, dataset_tree_identity


def _tree(root: Path) -> Path:
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "meta/info.json").write_text("{}", encoding="utf-8")
    (root / "data/a.parquet").write_bytes(b"first")
    return root


def test_tree_identity_is_stable_for_an_untouched_tree(tmp_path: Path) -> None:
    root = _tree(tmp_path / "ds")
    assert dataset_tree_identity(root) == dataset_tree_identity(root)


@pytest.mark.parametrize("change", ["rewrite", "add", "remove", "replace"])
def test_tree_identity_detects_any_file_change(tmp_path: Path, change: str) -> None:
    root = _tree(tmp_path / "ds")
    before = dataset_tree_identity(root)
    time.sleep(0.05)  # inode timestamps use a coarse clock tick
    target = root / "data/a.parquet"
    if change == "rewrite":
        stat = target.stat()
        target.write_bytes(b"other")  # same size
        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))  # same mtime
    elif change == "add":
        (root / "data/b.parquet").write_bytes(b"new")
    elif change == "remove":
        target.unlink()
    else:
        replacement = root / "data/tmp"
        replacement.write_bytes(b"first")
        replacement.replace(target)
    assert dataset_tree_identity(root) != before


def test_tree_identity_rejects_symlinks(tmp_path: Path) -> None:
    root = _tree(tmp_path / "ds")
    (root / "data/link.parquet").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ContentIntegrityError):
        dataset_tree_identity(root)
