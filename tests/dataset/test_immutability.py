from __future__ import annotations

import hashlib
from pathlib import Path

from lerobot_dataset_editor.dataset.index import DatasetIndex
from lerobot_dataset_editor.dataset.loader import load_document
from lerobot_dataset_editor.dataset.validation import validate_dataset


def _snapshot(root: Path) -> dict[str, tuple[str, int, int, int]]:
    result: dict[str, tuple[str, int, int, int]] = {}
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        stat = path.stat()
        result[path.relative_to(root).as_posix()] = (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_mode,
        )
    return result


def test_all_task4_read_paths_preserve_source_bytes_and_metadata(
    v30_annotated_fixture: Path, tmp_path: Path
) -> None:
    before = _snapshot(v30_annotated_fixture)

    load_document(v30_annotated_fixture)
    index = DatasetIndex(db_path=tmp_path / "cache" / "index.sqlite")
    index.get_or_build(v30_annotated_fixture)
    index.close()
    assert validate_dataset(v30_annotated_fixture).valid is True

    assert _snapshot(v30_annotated_fixture) == before
    assert not any(path.name.endswith(("-wal", "-shm")) for path in v30_annotated_fixture.rglob("*"))
