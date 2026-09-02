from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot_dataset_editor.parity.progress import read_progress
from lerobot_dataset_editor.parity.replay import map_replay


def test_progress_prefers_sarm_orders_columns_and_maps_duration(v30_fixture: Path, tmp_path: Path) -> None:
    dataset = tmp_path / "progress"
    shutil.copytree(v30_fixture, dataset)
    pq.write_table(pa.table({
        "episode_index": [0, 0, 0], "frame_index": [2, 0, 1],
        "zeta": [.2, 0, .1], "progress_dense": [.9, .1, .5],
        "progress_sparse": [1., 0., .5],
    }), dataset / "sarm_progress.parquet")
    pq.write_table(pa.table({"progress": [9.]}), dataset / "srm_progress.parquet")
    result = read_progress(dataset, 0, 2.0)
    assert result["source"] == "sarm_progress.parquet"
    assert [item["key"] for item in result["series"]] == ["progress_sparse"]
    assert result["series"][0]["points"][-1]["timestamp"] == 2.0


def test_progress_streaming_retains_at_most_4000_rows(v30_fixture: Path, tmp_path: Path) -> None:
    dataset = tmp_path / "progress-bounded"
    shutil.copytree(v30_fixture, dataset)
    row_count = 9001
    pq.write_table(pa.table({
        "episode_index": [0] * row_count,
        "frame_index": range(row_count),
        "progress": [index / row_count for index in range(row_count)],
        "unused": [b"payload"] * row_count,
    }), dataset / "sarm_progress.parquet", row_group_size=row_count)
    metrics: dict[str, int] = {}

    result = read_progress(dataset, 0, 2.0, metrics=metrics)

    assert len(result["series"][0]["points"]) == 4000
    assert metrics["retained_rows"] == 4000
    assert metrics["peak_batch_rows"] <= 8192


def test_replay_supports_v3_robots_and_normalizes_units(v30_fixture: Path) -> None:
    result = map_replay(v30_fixture, [2048, 180, .5, 1],
                        ["shoulder", "elbow", "wrist", "gripper"],
                        ["shoulder", "elbow", "wrist", "gripper"])
    assert result["supported"] is True
    assert result["positions"]["shoulder"] == 0
    # One mode is selected from all revolute values in the frame (ticks here).
    assert round(result["positions"]["elbow"], 6) == round((180 - 2048) / 2048 * 3.141592653589793, 6)
    assert round(result["positions"]["gripper"], 8) == .00044
    assert result["trail"] == {"seconds": 1.0, "max_points": 300}


def test_replay_uses_episode_gripper_range_without_sign_flips(v30_fixture: Path) -> None:
    result = map_replay(v30_fixture, [90, 25, -45],
                        ["left_joint_1", "left_gripper", "right_joint_2"],
                        ["openarm_left_joint1", "openarm_left_finger_joint1", "openarm_right_joint2", "openarm_left_finger_joint2"],
                        [{"min": -90, "max": 90}, {"min": 0, "max": 100}, {"min": -90, "max": 90}])
    assert result["positions"]["openarm_left_joint1"] == 3.141592653589793 / 2
    assert result["positions"]["openarm_right_joint2"] == -3.141592653589793 / 4
    assert result["positions"]["openarm_left_finger_joint1"] == .011
    assert result["positions"]["openarm_left_finger_joint2"] == .011
