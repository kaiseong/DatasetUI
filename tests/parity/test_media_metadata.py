from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from lerobot_dataset_editor.parity.dataset import (
    _colormap_band, _depth_encoding, dataset_episode,
)


def test_episode_embedded_media_contains_every_frame_and_grayscale_band(
    v30_fixture: Path, tmp_path: Path
) -> None:
    dataset = tmp_path / "grayscale"
    shutil.copytree(v30_fixture, dataset)
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.images.front"]["shape"] = [2, 2, 1]
    info_path.write_text(json.dumps(info))
    stats_path = dataset / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats["observation.images.front"] = {"q01": [[[.15]]], "q99": [[[.85]]]}
    stats_path.write_text(json.dumps(stats))

    track = next(item for item in dataset_episode(dataset, 0)["media"]
                 if item["camera"] == "observation.images.front")
    assert len(track["frames"]) == 10
    assert all(frame["data_url"].startswith("data:image/png;base64,") for frame in track["frames"])
    assert track["shape"] == [2, 2, 1] and track["grayscale"] is True
    assert track["raw_band"] == [.15, .85]
    assert track["q01"] == .15 and track["q99"] == .85


def test_depth_mm_linear_quantization_and_min_max_fallback() -> None:
    feature = {"info": {"is_depth_map": True, "video.depth_min": 0.0,
                         "video.depth_max": 10.0, "video.shift": 0.0,
                         "video.use_log": False, "depth_unit": "mm"}}
    encoding = _depth_encoding(feature)
    assert encoding == {"depth_min": 0.0, "depth_max": 10.0, "shift": 0.0,
                        "use_log": False, "depth_unit": "mm", "unit_to_metres": .001}
    assert _colormap_band({"min": [[[2000]]], "max": [[[8000]]]}, encoding) == pytest.approx((.2, .8))


def test_depth_mm_log_quantization_matches_pinned_golden() -> None:
    encoding = _depth_encoding({"info": {"is_depth_map": True,
        "video.depth_min": .01, "video.depth_max": 10.0, "video.shift": 3.5,
        "video.use_log": True, "depth_unit": "mm"}})
    low, high = _colormap_band({"q01": [[[0.0]]], "q99": [[[801.7945796579397]]]}, encoding)  # type: ignore[misc]
    assert low == pytest.approx(0, abs=1e-5)
    assert high == pytest.approx(.151006, abs=1e-4)


def test_invalid_or_degenerate_depth_bands_are_omitted() -> None:
    assert _colormap_band({"q01": 5, "q99": 5}, None) is None
    assert _colormap_band({"q01": float("nan"), "q99": 8}, None) is None
    assert _depth_encoding({"info": {"is_depth_map": True, "video.depth_min": -5,
        "video.depth_max": 10, "video.shift": 1, "video.use_log": True}}) is None


def test_declared_external_video_cameras_do_not_require_parquet_columns(
    v30_fixture: Path, tmp_path: Path
) -> None:
    dataset = tmp_path / "external-video"
    shutil.copytree(v30_fixture, dataset)
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    info["features"]["observation.images.external"] = {
        "dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channel"]
    }
    info["features"]["observation.images.depth"] = {
        "dtype": "video", "shape": [480, 640, 1], "names": ["height", "width", "channel"],
        "info": {"is_depth_map": True, "video.depth_min": 0.0, "video.depth_max": 10.0,
                 "video.shift": 0.0, "video.use_log": False, "depth_unit": "m"},
    }
    info_path.write_text(json.dumps(info))

    tracks = {item["camera"]: item for item in dataset_episode(dataset, 0)["media"]}
    assert tracks["observation.images.external"]["kind"] == "video"
    assert tracks["observation.images.external"]["relative_path"].endswith(
        "observation.images.external/episode_000000.mp4"
    )
    assert tracks["observation.images.depth"]["kind"] == "depth"
    assert tracks["observation.images.depth"]["grayscale"] is True
