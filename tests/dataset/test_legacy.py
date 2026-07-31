from __future__ import annotations

import json
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot_dataset_editor.dataset.loader import load_document
from lerobot_dataset_editor.dataset.validation import validate_dataset


def _legacy_fixture(root: Path, version: str) -> Path:
    meta = root / "meta"
    meta.mkdir(parents=True)
    info = {
        "codebase_version": version,
        "fps": 20,
        "total_episodes": 2,
        "total_frames": 7,
        "features": {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "action": {"dtype": "float32", "shape": [2], "names": ["a0", "a1"]},
        },
        "video_path": None,
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    episodes = [
        {"episode_index": 0, "length": 3, "tasks": ["pick"]},
        {"episode_index": 1, "length": 4, "tasks": ["place"]},
    ]
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episodes), encoding="utf-8"
    )
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "pick"}) + "\n" +
        json.dumps({"task_index": 1, "task": "place"}) + "\n",
        encoding="utf-8",
    )
    data = root / "data" / "chunk-000"
    data.mkdir(parents=True)
    for episode_index, length in ((0, 3), (1, 4)):
        pq.write_table(
            pa.table(
                {
                    "timestamp": pa.array(
                        [frame / 20.0 for frame in range(length)], type=pa.float32()
                    ),
                    "action": pa.array(
                        [[float(frame), float(frame + 1)] for frame in range(length)],
                        type=pa.list_(pa.float32(), 2),
                    ),
                }
            ),
            data / f"episode_{episode_index:06d}.parquet",
        )
    return root


@pytest.mark.parametrize("version", ["v1.6", "v2.0"])
def test_legacy_metadata_opens_as_common_read_only_document(tmp_path: Path, version: str) -> None:
    root = _legacy_fixture(tmp_path / version, version)
    doc = load_document(root)
    assert doc.version.version == version
    assert doc.total_frames == 7
    assert [episode.index for episode in doc.episodes] == [0, 1]
    assert [episode.length for episode in doc.episodes] == [3, 4]
    assert doc.tasks == ("pick", "place")


@pytest.mark.parametrize("version", ["v1.6", "v2.0"])
def test_legacy_layout_validates_through_common_path(tmp_path: Path, version: str) -> None:
    root = _legacy_fixture(tmp_path / f"validated-{version}", version)
    result = validate_dataset(root)
    assert result.valid is True
    assert result.errors == ()
