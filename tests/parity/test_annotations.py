from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa
import pytest

from lerobot_dataset_editor.parity.annotations import export_annotations, list_annotations, save_annotations


def _digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def test_draft_is_atomic_source_safe_and_export_rewrites_language(
    v30_fixture: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    before = _digest(v30_fixture)
    atoms = [
        {"role": "user", "content": "pick", "style": "task_aug", "timestamp": 0},
        {"role": "user", "content": "slow", "style": "interjection", "timestamp": .14},
        {"kind": "vqa", "question": "where?",
         "answer": {"detections": [{"label": "item", "bbox_format": "xyxy", "bbox": [0.1, 0.2, 0.8, 0.9]}]},
         "timestamp": .2, "camera": "observation.images.front"},
    ]
    assert save_annotations(v30_fixture, "project-1", 0, atoms)["count"] == 4
    assert list_annotations(v30_fixture, "project-1", 0)["draft"] is True
    output = tmp_path / "exported"
    result = export_annotations(v30_fixture, "project-1", output)
    assert result["exported"] is True
    assert _digest(v30_fixture) == before
    table = pq.read_table(next((output / "data").rglob("*.parquet")))
    assert "language_persistent" in table.column_names
    assert "language_events" in table.column_names
    assert table["language_persistent"][0].as_py()[0]["style"] == "task_aug"
    assert sum(bool(row) for row in table["language_events"].to_pylist()) == 2


def test_vqa_geometry_must_be_normalized(v30_fixture: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    atom = {"role": "assistant", "content": '{"bbox":[0,0,2,1]}', "style": "vqa",
            "timestamp": 0, "camera": "observation.images.front"}
    try:
        save_annotations(v30_fixture, "project-2", 0, [atom])
    except ValueError as exc:
        assert "[0,1]" in str(exc)
    else:
        raise AssertionError("invalid normalized geometry was accepted")


def test_quick_constructors_snap_to_nearest_frame_and_round_geometry(
    v30_fixture: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    atoms = [
        {"kind": "speech", "text": "ready", "timestamp": .14},
        {"kind": "vqa", "question": "where?", "answer": {"label": "grasp", "point_format": "xy", "point": [.123456, .987654]},
         "camera": "observation.images.front", "timestamp": .14},
    ]
    assert save_annotations(v30_fixture, "constructors", 0, atoms)["count"] == 3
    saved = list_annotations(v30_fixture, "constructors", 0)["atoms"]
    assert {atom["timestamp"] for atom in saved} == {.1}
    speech = saved[0]
    assert speech["content"] is None and speech["tool_calls"][0]["function"]["name"] == "say"
    answer = json.loads(saved[2]["content"])
    assert answer["point"] == [.1235, .9877]


@pytest.mark.parametrize("atoms,message", [
    ([{"role": "assistant", "style": "task_aug", "content": "x", "timestamp": 0}], "requires role"),
    ([{"role": "user", "style": "interjection", "content": "x", "timestamp": 0,
       "camera": "observation.images.front"}], "camera must be null"),
    ([{"role": "assistant", "style": None, "content": None, "timestamp": 0,
       "tool_calls": [{"type": "function", "function": {"name": "other", "arguments": {"text": "x"}}}]}], "canonical say"),
    ([{"role": "user", "style": "vqa", "content": "question", "timestamp": 0,
       "camera": "observation.images.front"}], "requires an assistant answer"),
    ([{"kind": "vqa", "question": "where", "answer": {"unknown": 1}, "timestamp": 0,
       "camera": "observation.images.front"}], "does not match"),
])
def test_annotation_semantic_validator_rejects_noncanonical_atoms(
    v30_fixture: Path, tmp_path: Path, monkeypatch, atoms, message: str
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    with pytest.raises(ValueError, match=message):
        save_annotations(v30_fixture, "invalid", 0, atoms)


def test_export_strips_legacy_columns_and_merges_canonical_say_tool(
    v30_fixture: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    dataset = tmp_path / "legacy"
    import shutil
    shutil.copytree(v30_fixture, dataset)
    parquet = next((dataset / "data").rglob("*.parquet"))
    table = pq.read_table(parquet).append_column("subtask_index", pa.array([0] * 30, type=pa.int64()))
    pq.write_table(table, parquet)
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["subtask_index"] = {"dtype": "int64", "shape": [1], "names": None}
    info["features"]["tools"] = {"dtype": "string", "shape": [1], "names": None}
    custom = {"type": "function", "function": {"name": "custom", "parameters": {}}}
    info["tools"] = [custom]
    info_path.write_text(json.dumps(info))
    output = tmp_path / "clean"
    export_annotations(dataset, "clean-export", output)
    exported = pq.read_table(next((output / "data").rglob("*.parquet")))
    assert "subtask_index" not in exported.column_names and "tools" not in exported.column_names
    exported_info = json.loads((output / "meta" / "info.json").read_text())
    assert "subtask_index" not in exported_info["features"] and "tools" not in exported_info["features"]
    assert exported_info["tools"][0] == custom
    say = next(tool for tool in exported_info["tools"] if tool["function"]["name"] == "say")
    assert say["function"]["description"] == "Speak a short utterance to the user via the TTS executor."
    assert say["function"]["parameters"]["properties"]["text"]["description"] == "The verbatim text to speak."


@pytest.mark.parametrize("atoms,message", [
    ([{"role": "user", "style": "task_aug", "content": "x", "timestamp": .1}], "timestamp must be 0"),
    ([{"kind": "vqa", "question": "where", "answer": {
        "detections": [{"label": "x", "bbox_format": "xywh", "bbox": [.1, .2, .4, .5]}]},
       "timestamp": 0, "camera": "observation.images.front"}], "invalid vqa bbox"),
    ([{"kind": "vqa", "question": "where", "answer": {
        "detections": [{"label": "x", "bbox_format": "xyxy", "bbox": [.8, .2, .1, .5]}]},
       "timestamp": 0, "camera": "observation.images.front"}], "ordered xyxy"),
])
def test_backend_rejects_noncanonical_task_and_bbox_semantics(
    v30_fixture: Path, tmp_path: Path, monkeypatch, atoms, message: str
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    with pytest.raises(ValueError, match=message):
        save_annotations(v30_fixture, "canonical", 0, atoms)


def test_backend_rejects_missing_episode_and_export_symlinks_and_removes_legacy_meta(
    v30_fixture: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    with pytest.raises(ValueError, match="episode does not exist"):
        save_annotations(v30_fixture, "missing", 999, [
            {"role": "user", "style": "interjection", "content": "x", "timestamp": 0},
        ])

    import shutil
    dataset = tmp_path / "legacy-meta"
    shutil.copytree(v30_fixture, dataset)
    (dataset / "meta" / "subtasks.parquet").write_bytes(b"legacy")
    (dataset / "meta" / "tasks_high_level.parquet").write_bytes(b"legacy")
    output = tmp_path / "legacy-clean"
    export_annotations(dataset, "legacy-clean", output)
    assert not (output / "meta" / "subtasks.parquet").exists()
    assert not (output / "meta" / "tasks_high_level.parquet").exists()

    outside = tmp_path / "outside-secret"
    outside.write_text("do not copy")
    (dataset / "meta" / "escape").symlink_to(outside)
    with pytest.raises(ValueError, match="unsupported symlink"):
        export_annotations(dataset, "legacy-clean", tmp_path / "symlink-export")


def test_annotation_list_and_save_never_full_read_media_payloads(
    v30_fixture: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    original = pq.read_table

    def guarded_read_table(where, columns=None, *args, **kwargs):
        if columns is None or any("images" in column for column in columns):
            raise AssertionError("annotation draft path decoded a media/full table")
        return original(where, columns=columns, *args, **kwargs)

    monkeypatch.setattr(pq, "read_table", guarded_read_table)
    save_annotations(v30_fixture, "bounded-media", 0, [
        {"role": "user", "style": "interjection", "content": "x", "timestamp": .1},
    ])
    result = list_annotations(v30_fixture, "bounded-media", 0)
    assert result["draft"] is True


def test_export_streams_every_data_file_to_keep_declared_language_schema_consistent(
    v30_fixture: Path, tmp_path: Path, monkeypatch
) -> None:
    import shutil

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    dataset = tmp_path / "multi-file"
    shutil.copytree(v30_fixture, dataset)
    original = next((dataset / "data").rglob("*.parquet"))
    sibling = original.with_name("file-001.parquet")
    shutil.copy2(original, sibling)
    save_annotations(dataset, "multi-file", 0, [
        {"role": "user", "style": "interjection", "content": "x", "timestamp": .1},
    ])

    output = tmp_path / "multi-file-export"
    export_annotations(dataset, "multi-file", output)

    schemas = [pq.read_schema(path).names for path in sorted((output / "data").rglob("*.parquet"))]
    assert len(schemas) == 2
    assert all("language_persistent" in names and "language_events" in names for names in schemas)
