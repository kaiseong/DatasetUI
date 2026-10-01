from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datasetui import transforms
from datasetui.stationary_trim import stationary_trim_bounds
from datasetui.transform_errors import CurationTransformError


VIDEO_KEY = "observation.images.top"


def _frame(episode: int, length: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "episode_index": np.full(length, episode, dtype=np.int64),
            "frame_index": np.arange(length, dtype=np.int64),
            "timestamp": np.arange(length, dtype=np.float32) / 10,
            "index": np.arange(length, dtype=np.int64),
            "task_index": np.zeros(length, dtype=np.int64),
            "observation.state": [[float(index)] for index in range(length)],
        }
    )


def test_stationary_bounds_match_local_5090_reference_criterion() -> None:
    data = pd.DataFrame(
        {
            "observation.state": [
                [0.0, 0.0],
                [0.0001, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [3.0001, 0.0],
            ]
        }
    )
    config = {
        "state_epsilon": 5e-4,
        "start_margin_s": 0.1,
        "end_margin_s": 0.0,
    }

    # The 5090 tool compares every leading/trailing state with the first/last
    # state, then retains the requested amount of stationary context.
    assert stationary_trim_bounds(data, fps=10, config=config, episode_index=7) == (
        1,
        4,
        "stationary",
    )


def test_stationary_bounds_reject_nonfinite_and_fully_stationary_episodes() -> None:
    with pytest.raises(CurationTransformError, match="finite state values"):
        stationary_trim_bounds(
            pd.DataFrame({"observation.state": [[0.0], [float("nan")], [1.0]]}),
            fps=10,
            config={"state_epsilon": 5e-4, "margin_s": 0},
            episode_index=2,
        )
    with pytest.raises(CurationTransformError, match="remove all frames"):
        stationary_trim_bounds(
            pd.DataFrame({"observation.state": [[1.0], [1.0], [1.0]]}),
            fps=10,
            config={"state_epsilon": 5e-4, "margin_s": 0},
            episode_index=3,
        )


class _SharedShardSource:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.fps = 10.0
        self.video_keys = [VIDEO_KEY]
        self.info = {
            "codebase_version": "v3.0",
            "fps": 10,
            "features": {
                VIDEO_KEY: {
                    "dtype": "video",
                    "shape": [2, 2, 3],
                    "info": {
                        "video.codec": "av1",
                        "video.crf": 30,
                        "video.preset": 12,
                    },
                    "video": {"codec": "av1", "crf": 30, "preset": 12},
                }
            },
        }
        self.video = root / f"videos/{VIDEO_KEY}/chunk-007/file-009.mp4"
        self.video.parent.mkdir(parents=True)
        self.video.write_bytes(b"unchanged-av1-bitstream" * 100)

    def video_source(self, episode_index: int, video_key: str, metadata):
        assert episode_index in {10, 11}
        assert video_key == VIDEO_KEY
        return self.video, round(
            float(metadata[f"videos/{video_key}/from_timestamp"]) * 10
        )


def _metadata(source_episode: int, start: float, end: float) -> dict:
    prefix = f"videos/{VIDEO_KEY}"
    return {
        "_source_episode_index": source_episode,
        f"{prefix}/chunk_index": 7,
        f"{prefix}/file_index": 9,
        f"{prefix}/from_timestamp": start,
        f"{prefix}/to_timestamp": end,
    }


def test_v3_stationary_trim_copies_shared_shard_once_without_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _SharedShardSource(tmp_path / "source")
    destination = tmp_path / "output"
    (destination / "meta").mkdir(parents=True)
    (destination / "data").mkdir()
    (destination / "videos").mkdir()
    monkeypatch.setattr(
        transforms,
        "_slice_video",
        lambda *args, **kwargs: pytest.fail("stationary trim must not encode video"),
    )
    episodes = [
        (_frame(0, 4), _metadata(10, 0.5, 1.5), 2, 6),
        (_frame(1, 4), _metadata(11, 1.5, 2.5), 1, 5),
    ]
    progress = []

    transforms._write_v3(
        source,
        destination,
        episodes,
        [{"task_index": 0, "task": "pick"}],
        {},
        logical_stationary=True,
        on_progress=progress.append,
    )

    output_video = destination / f"videos/{VIDEO_KEY}/chunk-000/file-000.mp4"
    assert len(list((destination / "videos").rglob("*.mp4"))) == 1
    assert (
        hashlib.sha256(output_video.read_bytes()).digest()
        == hashlib.sha256(source.video.read_bytes()).digest()
    )
    metadata = pd.read_parquet(destination / "meta/episodes/chunk-000/file-000.parquet")
    prefix = f"videos/{VIDEO_KEY}"
    assert metadata[f"{prefix}/chunk_index"].tolist() == [0, 0]
    assert metadata[f"{prefix}/file_index"].tolist() == [0, 0]
    assert metadata[f"{prefix}/from_timestamp"].tolist() == pytest.approx([0.7, 1.6])
    assert metadata[f"{prefix}/to_timestamp"].tolist() == pytest.approx([1.1, 2.0])
    info = json.loads((destination / "meta/info.json").read_text())
    assert info["total_videos"] == 1
    assert info["features"][VIDEO_KEY]["info"] == {
        "video.codec": "av1",
        "video.crf": 30,
        "video.preset": 12,
    }
    assert info["features"][VIDEO_KEY]["video"] == {
        "codec": "av1",
        "crf": 30,
        "preset": 12,
    }
    assert progress[-1]["stage"] == "video"
    assert progress[-1]["unit"] == "bytes"
    assert (
        progress[-1]["completed"]
        == progress[-1]["total"]
        == len(source.video.read_bytes())
    )


def test_v3_stationary_trim_copy_honors_cancellation_callback(tmp_path: Path) -> None:
    source = _SharedShardSource(tmp_path / "source")
    destination = tmp_path / "output"
    (destination / "meta").mkdir(parents=True)
    (destination / "data").mkdir()
    (destination / "videos").mkdir()

    def cancel_after_copy_starts(event: dict) -> None:
        if event["stage"] == "video" and event["completed"] > 0:
            raise RuntimeError("lease lost")

    with pytest.raises(RuntimeError, match="lease lost"):
        transforms._write_v3(
            source,
            destination,
            [(_frame(0, 4), _metadata(10, 0.5, 1.5), 2, 6)],
            [{"task_index": 0, "task": "pick"}],
            {},
            logical_stationary=True,
            on_progress=cancel_after_copy_starts,
        )
    assert not list((destination / "videos").rglob("*.mp4"))


def test_stationary_trim_rejects_v2_before_writing(tmp_path: Path) -> None:
    source = type(
        "Source",
        (),
        {
            "root": tmp_path / "source",
            "version": "v2.1",
        },
    )()
    with pytest.raises(CurationTransformError, match="only v3.0 to v3.0"):
        transforms._write_dataset(
            source=source,
            destination=tmp_path / "output",
            source_indices=[0],
            trim_config={"enabled": True, "method": "stationary"},
            annotations={},
        )
    assert not (tmp_path / "output").exists()


def test_missing_method_keeps_legacy_motion_trim() -> None:
    increments = np.zeros(20)
    increments[3:5] = 1
    increments[8:13] = 1
    increments[16:17] = 1
    values = [[float(value)] for value in np.r_[0, np.cumsum(increments)]]
    data = pd.DataFrame({"action": values, "observation.state": values})

    assert transforms._trim_bounds(
        data,
        {},
        10,
        {
            "enabled": True,
            "threshold": 0.01,
            "hold_time_s": 0.1,
            "margin_s": 0,
        },
        0,
    ) == (4, 18, "motion")
