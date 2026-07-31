"""Regression tests for exact format and Space-parity contracts.

These assertions intentionally target the official v0.3.3 v2.1 layout, the
v0.6.0 v3.0 version marker, and the pinned Space annotation behavior.  They
exist to prevent self-consistent but non-interoperable fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from lerobot_dataset_editor.schemas import validate_info_v21, validate_info_v30


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_v21_uses_exact_official_version_and_core_layout(v21_fixture: Path) -> None:
    info = json.loads((v21_fixture / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["codebase_version"] == "v2.1"
    assert info["data_path"] == "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    assert info["video_path"] is None
    assert {
        "total_videos",
        "total_chunks",
        "chunks_size",
        "total_episodes",
        "total_frames",
        "total_tasks",
    } <= info.keys()

    expected_data = {
        v21_fixture / "data" / "chunk-000" / "episode_000000.parquet",
        v21_fixture / "data" / "chunk-000" / "episode_000001.parquet",
    }
    assert set((v21_fixture / "data").glob("*/*.parquet")) == expected_data
    assert not (v21_fixture / "data" / "chunk-000" / "file-000.parquet").exists()

    expected_meta = {
        "info.json",
        "episodes.jsonl",
        "episodes_stats.jsonl",
        "tasks.jsonl",
        "stats.json",
    }
    assert expected_meta <= {path.name for path in (v21_fixture / "meta").iterdir()}
    assert not (v21_fixture / "meta" / "tasks.parquet").exists()
    assert not (v21_fixture / "meta" / "episodes").exists()

    episodes = _jsonl(v21_fixture / "meta" / "episodes.jsonl")
    episode_stats = _jsonl(v21_fixture / "meta" / "episodes_stats.jsonl")
    tasks = _jsonl(v21_fixture / "meta" / "tasks.jsonl")
    assert [row["episode_index"] for row in episodes] == [0, 1]
    assert [row["episode_index"] for row in episode_stats] == [0, 1]
    assert tasks == [{"task_index": 0, "task": "pick up object"}]
    assert sum(row["length"] for row in episodes) == info["total_frames"]


def test_v21_schema_accepts_only_canonical_marker(v21_fixture: Path) -> None:
    info = json.loads((v21_fixture / "meta" / "info.json").read_text(encoding="utf-8"))
    assert validate_info_v21(info) == []
    noncanonical = {**info, "codebase_version": "2.1.0"}
    assert validate_info_v21(noncanonical)


def test_v30_uses_v060_canonical_marker_and_paths(v30_fixture: Path) -> None:
    info = json.loads((v30_fixture / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["codebase_version"] == "v3.0"
    assert info["data_path"] == "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    assert info["video_path"] is None
    assert validate_info_v30(info) == []
    noncanonical = {**info, "codebase_version": "3.0.0"}
    assert validate_info_v30(noncanonical)


def test_annotation_fixture_uses_arrow_list_of_struct_and_all_space_atoms(
    v30_annotated_fixture: Path,
) -> None:
    info = json.loads((v30_annotated_fixture / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["features"]["language_persistent"]["dtype"] == "language"
    assert info["features"]["language_events"]["dtype"] == "language"
    say_tools = [tool for tool in info["tools"] if tool.get("function", {}).get("name") == "say"]
    assert len(say_tools) == 1
    assert say_tools[0]["function"]["parameters"]["required"] == ["text"]

    table = pq.read_table(
        v30_annotated_fixture / "data" / "chunk-000" / "file-000.parquet"
    )
    persistent_type = table.schema.field("language_persistent").type
    events_type = table.schema.field("language_events").type
    assert pa.types.is_list(persistent_type)
    assert pa.types.is_struct(persistent_type.value_type)
    assert persistent_type.value_type.names == [
        "role",
        "content",
        "style",
        "timestamp",
        "camera",
        "tool_calls",
    ]
    assert pa.types.is_list(events_type)
    assert pa.types.is_struct(events_type.value_type)
    assert events_type.value_type.names == ["role", "content", "style", "camera", "tool_calls"]

    persistent_rows = table.column("language_persistent").to_pylist()
    assert persistent_rows and all(rows == persistent_rows[0] for rows in persistent_rows)
    assert {row["style"] for row in persistent_rows[0]} >= {
        "task_aug",
        "subtask",
        "plan",
        "memory",
    }

    event_rows = [row for rows in table.column("language_events").to_pylist() for row in rows]
    assert {row["style"] for row in event_rows} >= {None, "interjection", "vqa"}
    speech = [row for row in event_rows if row["style"] is None]
    assert speech
    assert all(row["role"] == "assistant" and row["content"] is None for row in speech)
    assert all(row["tool_calls"][0]["function"]["name"] == "say" for row in speech)

    vqa_answers = [
        json.loads(row["content"])
        for row in event_rows
        if row["style"] == "vqa" and row["role"] == "assistant"
    ]
    kinds = set()
    for answer in vqa_answers:
        keys = set(answer)
        if "detections" in keys:
            kinds.add("bbox")
        elif {"label", "point_format", "point"} <= keys:
            kinds.add("keypoint")
        elif {"label", "count"} <= keys:
            kinds.add("count")
        elif {"label", "attribute", "value"} <= keys:
            kinds.add("attribute")
        elif {"subject", "relation", "object"} <= keys:
            kinds.add("spatial")
    assert kinds == {"bbox", "keypoint", "count", "attribute", "spatial"}


def test_space_parity_contract_is_feature_complete_and_truthful() -> None:
    contract_path = Path(__file__).resolve().parents[1] / "contracts" / "space-parity.yaml"
    data = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in data["parity_features"]}
    required = {
        "dataset-search-private-access",
        "episode-navigation-pagination",
        "synchronized-multi-camera-video",
        "segmented-video-looping",
        "camera-controls",
        "depth-grayscale-colormaps",
        "synchronized-action-state-charts",
        "statistics-histogram",
        "first-last-frame-gallery",
        "episode-flags-export",
        "length-low-movement-filtering",
        "action-autocorrelation-chunk",
        "action-velocity-activity-discrete",
        "jerky-episode-ranking",
        "cross-episode-variance-heatmap",
        "demonstrator-speed-cv",
        "state-action-temporal-alignment",
        "progress-parquet",
        "urdf-replay-robots",
        "end-effector-trails",
        "doctor-link",
        "annotation-task-aug",
        "annotation-subtask",
        "annotation-plan",
        "annotation-memory",
        "annotation-interjection",
        "annotation-speech-say",
        "annotation-vqa-bbox",
        "annotation-vqa-keypoint",
        "annotation-vqa-count",
        "annotation-vqa-attribute",
        "annotation-vqa-spatial",
        "annotation-frame-snapping",
        "annotation-editable-timeline",
        "annotation-camera-overlays",
        "annotation-bbox-drag",
        "annotation-keypoint-click",
    }
    assert required <= by_id.keys(), f"missing parity capabilities: {sorted(required - by_id.keys())}"
    for item in by_id.values():
        assert item["status"] in {"contract-tested", "planned"}
        assert "::test_" in item["test_id"]
        assert 1 <= item["implementation_task"] <= 23
