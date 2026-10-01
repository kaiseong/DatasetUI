from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import pytest

from datasetui.validation_video import DecodedVideo, VideoValidator
from datasetui.validation_statistics import NumericStatisticsValidator


KEY = "observation.images.top"


class Source:
    video_keys = [KEY]

    def __init__(self, path: Path, *, version: str, fps: float = 10.0) -> None:
        self.path = path
        self.version = version
        self.fps = fps
        self.episode_metadata = {}

    def video_source(self, episode_index, video_key, metadata):
        return self.path, round(
            float(metadata.get(f"videos/{KEY}/from_timestamp", 0)) * self.fps
        )


class Reporter:
    def __init__(self) -> None:
        self.frames = 0

    def frame_decoded(self) -> None:
        self.frames += 1


def _write_video(path: Path, *, frames: int = 4, fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = 32
    stream.height = 24
    stream.pix_fmt = "yuv420p"
    for value in range(frames):
        frame = av.VideoFrame.from_ndarray(
            np.full((24, 32, 3), value, dtype=np.uint8), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _validator(
    path: Path,
    *,
    version: str = "v2.1",
    fps: float = 10.0,
    shape: list[int] | None = None,
):
    issues = []
    reporter = Reporter()
    info = {"features": {KEY: {"dtype": "video", "shape": shape or [24, 32, 3]}}}
    validator = VideoValidator(
        Source(path, version=version, fps=fps),
        info,
        lambda *item: issues.append(item),
        reporter,
    )
    return validator, issues, reporter


def test_v21_requires_exact_frame_count_dimensions_fps_and_pts(tmp_path: Path) -> None:
    path = tmp_path / "episode.mp4"
    _write_video(path)
    validator, issues, reporter = _validator(path)

    validator.validate_episode(0, {}, 4, timestamps=np.arange(4) / 10)

    assert issues == []
    assert reporter.frames == 4


def test_rejects_wrong_dimensions_fps_and_frame_count(tmp_path: Path) -> None:
    path = tmp_path / "episode.mp4"
    _write_video(path, frames=3, fps=5)
    validator, issues, _ = _validator(path, shape=[99, 32, 3])

    validator.validate_episode(0, {}, 4)

    codes = {item[1] for item in issues}
    assert "video_shape_mismatch" in codes
    assert "video_fps_mismatch" in codes
    assert "video_pts_mismatch" in codes
    assert "video_frame_mismatch" in codes


def test_v3_checks_from_to_mapping_and_decodes_shared_shard_once(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shard.mp4"
    _write_video(path, frames=8)
    validator, issues, reporter = _validator(path, version="v3.0")
    validator._source.episode_metadata = {
        0: {
            "dataset_from_index": 0,
            "dataset_to_index": 4,
            f"videos/{KEY}/from_timestamp": 0.0,
            f"videos/{KEY}/to_timestamp": 0.4,
        },
        1: {
            "dataset_from_index": 4,
            "dataset_to_index": 8,
            f"videos/{KEY}/from_timestamp": 0.4,
            f"videos/{KEY}/to_timestamp": 0.8,
        },
    }

    validator.validate_episode(
        0,
        {f"videos/{KEY}/from_timestamp": 0.0, f"videos/{KEY}/to_timestamp": 0.4},
        4,
        timestamps=np.arange(4) / 10,
    )
    validator.validate_episode(
        1,
        {f"videos/{KEY}/from_timestamp": 0.4, f"videos/{KEY}/to_timestamp": 0.8},
        4,
        timestamps=np.arange(4) / 10,
    )

    assert issues == []
    assert reporter.frames == 8
    assert validator.decoded_paths == (path,)
    assert validator.visual_statistics[KEY]["count"].tolist() == [8]


def test_v3_rejects_bad_to_timestamp_and_data_sync(tmp_path: Path) -> None:
    path = tmp_path / "shard.mp4"
    _write_video(path, frames=4)
    validator, issues, _ = _validator(path, version="v3.0")

    validator.validate_episode(
        0,
        {f"videos/{KEY}/from_timestamp": 0.0, f"videos/{KEY}/to_timestamp": 99.0},
        4,
        timestamps=np.asarray([100.0, 100.1, 100.2, 100.3]),
    )

    codes = {item[1] for item in issues}
    assert "video_segment_metadata_invalid" in codes
    assert "video_data_timestamp_mismatch" in codes


def test_progress_exception_propagates_instead_of_becoming_decode_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "episode.mp4"
    _write_video(path)

    class BrokenReporter:
        def frame_decoded(self):
            raise RuntimeError("lease lost")

    validator = VideoValidator(
        Source(path, version="v2.1"),
        {"features": {KEY: {"dtype": "video", "shape": [24, 32, 3]}}},
        lambda *_: None,
        BrokenReporter(),
    )

    with pytest.raises(RuntimeError, match="lease lost"):
        validator.validate_episode(0, {}, 4)


def test_recomputed_visual_stats_validate_and_detect_wrong_mean(tmp_path: Path) -> None:
    path = tmp_path / "episode.mp4"
    _write_video(path)
    validator, video_issues, _ = _validator(path)
    validator.validate_episode(0, {}, 4)
    assert video_issues == []
    actual = validator.visual_statistics[KEY]
    assert actual["count"].tolist() == [4]
    assert actual["mean"].shape == (3,)
    assert np.all((actual["mean"] >= 0) & (actual["mean"] <= 1))

    stats_path = tmp_path / "meta/stats.json"
    stats_path.parent.mkdir()
    stored = {
        KEY: {
            name: (
                actual[name].tolist()
                if name == "count"
                else actual[name].reshape(3, 1, 1).tolist()
            )
            for name in ("min", "max", "mean", "std", "count")
        }
    }
    stats_path.write_text(json.dumps(stored), encoding="utf-8")
    info = {
        "total_frames": 4,
        "features": {KEY: {"dtype": "video", "shape": [24, 32, 3]}},
    }
    stats_issues = []
    stats_validator = NumericStatisticsValidator(
        info, lambda *item: stats_issues.append(item)
    )

    summary = stats_validator.validate(
        tmp_path,
        complete_dataset=True,
        visual_statistics=validator.visual_statistics,
    )

    assert stats_issues == []
    assert summary.validated_features == (KEY,)
    assert summary.unverified_visual_features == ()

    stored[KEY]["mean"][0][0][0] += 0.25
    stats_path.write_text(json.dumps(stored), encoding="utf-8")
    mismatch_issues = []
    mismatch_validator = NumericStatisticsValidator(
        info, lambda *item: mismatch_issues.append(item)
    )
    mismatch_validator.validate(
        tmp_path,
        complete_dataset=True,
        visual_statistics=validator.visual_statistics,
    )
    assert any(item[1] == "stats_value_mismatch" for item in mismatch_issues)


def test_visual_sample_count_is_not_total_frames_above_one_hundred(
    tmp_path: Path,
) -> None:
    path = tmp_path / "long-episode.mp4"
    _write_video(path, frames=101)
    validator, issues, _ = _validator(path)

    validator.validate_episode(0, {}, 101)

    assert issues == []
    assert validator.visual_statistics[KEY]["count"].tolist() == [100]
    assert validator.visual_statistics[KEY]["_legacy_pixel_count"].tolist() == [76800]

    legacy = validator.visual_statistics[KEY]
    stored = {
        KEY: {
            name: legacy[name].reshape(3, 1, 1).tolist()
            for name in ("min", "max", "mean", "std")
        }
    }
    stored[KEY]["count"] = legacy["_legacy_pixel_count"].tolist()
    stats_path = tmp_path / "meta/stats.json"
    stats_path.parent.mkdir()
    stats_path.write_text(json.dumps(stored), encoding="utf-8")
    stats_issues = []
    stats_validator = NumericStatisticsValidator(
        {"features": {KEY: {"dtype": "video", "shape": [24, 32, 3]}}},
        lambda *item: stats_issues.append(item),
    )
    summary = stats_validator.validate(
        tmp_path,
        complete_dataset=True,
        visual_statistics=validator.visual_statistics,
    )
    assert stats_issues == []
    assert summary.validated_features == (KEY,)


def test_shifted_pts_and_dynamic_dimensions_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "unused.mp4"
    validator, issues, _ = _validator(path)
    shifted = DecodedVideo(
        frame_count=4,
        width=32,
        height=24,
        channels=3,
        dimensions_consistent=False,
        stream_fps=10.0,
        pts_seconds=(1.0, 1.1, 1.2, 1.3),
    )

    validator._validate_format(KEY, shifted, 0)
    validator._validate_fps(KEY, shifted, 0)

    codes = {item[1] for item in issues}
    assert "video_shape_mismatch" in codes
    assert "video_pts_mismatch" in codes


def test_canonical_depth_video_flag_prevents_false_rgb_stats(tmp_path: Path) -> None:
    path = tmp_path / "depth.mp4"
    _write_video(path)
    issues = []
    reporter = Reporter()
    validator = VideoValidator(
        Source(path, version="v2.1"),
        {
            "features": {
                KEY: {
                    "dtype": "video",
                    "shape": [24, 32, 3],
                    "video.is_depth_map": True,
                }
            }
        },
        lambda *item: issues.append(item),
        reporter,
    )

    validator.validate_episode(0, {}, 4)

    assert issues == []
    assert validator.visual_statistics == {}
