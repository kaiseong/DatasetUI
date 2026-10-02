from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from test_transforms import _write_v21
from test_validation_conversion import _write_v3

from datasetui.dataset_io.source import DatasetSource
from datasetui.validation import validate_dataset_root


@pytest.mark.parametrize(
    "defect",
    [
        "fractional_index",
        "missing_tasks",
        "missing_episodes",
        "invalid_task",
        "invalid_split",
        "shifted_time",
        "duplicate_time",
        "string_action",
        "wrong_shape",
        "sensor_nan",
        "episode_length",
        "duplicate_metadata",
    ],
)
def test_structural_defects_cannot_pass_quick_validation(tmp_path: Path, defect: str):
    root = tmp_path / "dataset"
    _write_v21(root)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    path = root / "data/chunk-000/episode_000000.parquet"
    data = pd.read_parquet(path)
    if defect == "fractional_index":
        data["frame_index"] = data["frame_index"] + 0.25
    elif defect == "missing_tasks":
        (root / "meta/tasks.jsonl").unlink()
    elif defect == "missing_episodes":
        (root / "meta/episodes.jsonl").unlink()
    elif defect == "invalid_task":
        data["task_index"] = 999
    elif defect == "invalid_split":
        info["splits"] = {"train": "0:999"}
    elif defect == "shifted_time":
        data["timestamp"] += 100
    elif defect == "duplicate_time":
        data["timestamp"] = 0.0
    elif defect == "string_action":
        data["action"] = [[str(value[0])] for value in data["action"]]
    elif defect == "wrong_shape":
        info["features"]["action"]["shape"] = [1, 1]
    elif defect == "sensor_nan":
        info["features"]["observation.force"] = {"dtype": "float32", "shape": [1]}
        data["observation.force"] = [[float("nan")]] * len(data)
    elif defect in {"episode_length", "duplicate_metadata"}:
        ep_path = root / "meta/episodes.jsonl"
        rows = [json.loads(line) for line in ep_path.read_text().splitlines()]
        if defect == "episode_length":
            rows[0]["length"] = 99
        else:
            rows.append(rows[0])
        ep_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    data.to_parquet(path, index=False)
    info_path.write_text(json.dumps(info))
    result = validate_dataset_root(root, mode="quick")
    assert result["passed"] is False, defect
    assert result["failures"] > 0


@pytest.mark.parametrize("defect", ["dtype_width", "unassigned_split", "huge_total"])
def test_additional_schema_boundaries(tmp_path, defect):
    root = tmp_path / "dataset"
    _write_v21(root)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    if defect == "unassigned_split":
        info["splits"] = {"train": "0:1"}
    elif defect == "huge_total":
        info["total_episodes"] = 10**18
        info["splits"] = {"train": f"0:{10**18}"}
    else:
        path = root / "data/chunk-000/episode_000000.parquet"
        data = pd.read_parquet(path)
        data["timestamp"] = data["timestamp"].astype("float64")
        data.to_parquet(path, index=False)
    info_path.write_text(json.dumps(info))
    assert validate_dataset_root(root, mode="quick")["passed"] is False


def test_dtype_failure_explains_expected_actual_location_and_remedy(tmp_path):
    root = tmp_path / "dataset"
    _write_v21(root)
    path = root / "data/chunk-000/episode_000000.parquet"
    data = pd.read_parquet(path)
    data["timestamp"] = data["timestamp"].astype("float64")
    data.to_parquet(path, index=False)
    result = validate_dataset_root(root, mode="quick")
    failure = next(
        item for item in result["issues"] if item["code"] == "feature_dtype_mismatch"
    )
    message = failure["message"]
    for expected in (
        "timestamp",
        "float32",
        "float64",
        "meta/info.json",
        "에피소드 0",
        "행 0",
        f"{len(data)}개",
        "조치:",
        "원본",
    ):
        assert expected in message


def test_v3_task_descriptions_can_be_stored_in_string_index(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_v3(root)
    task_text = "{'Sort the flowers by color': 'place the matching flowers'}"
    pd.DataFrame(
        {"task_index": [0]}, index=pd.Index([task_text], name="task")
    ).to_parquet(root / "meta/tasks.parquet")
    episodes_path = root / "meta/episodes/chunk-000/file-000.parquet"
    episodes = pd.read_parquet(episodes_path)
    episodes["tasks"] = [[task_text], [task_text]]
    episodes.to_parquet(episodes_path, index=False)

    result = validate_dataset_root(root, mode="quick")

    assert result["passed"] is True
    source = DatasetSource(
        root, json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    )
    assert source.tasks == {0: task_text}


def test_explicit_task_column_precedes_string_index_and_remains_strict(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    _write_v3(root)
    pd.DataFrame(
        {"task_index": [0], "task": [" "]},
        index=pd.Index(["valid fallback must not be used"], name="task"),
    ).to_parquet(root / "meta/tasks.parquet")

    result = validate_dataset_root(root, mode="quick")

    assert result["passed"] is False
    assert any(
        item["message"] == "Task descriptions must be nonempty strings"
        for item in result["issues"]
    )
