from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pytest

from test_transforms import _settings, _write_v21

from datasetui import transforms
from datasetui.dataset_io import video as dataset_video
from datasetui.database import Database
from datasetui.dataset_io.source import DatasetSource
from datasetui.datasets import inspect_dataset
from datasetui.transform_errors import CurationTransformError
from datasetui.transforms import materialize_curation_recipe
from datasetui.validation import validate_dataset_root


CAMERAS = {"observation.images.front": "av1", "observation.images.left": "h264"}


def _encode(
    path: Path, codec: str, values: list[int], *, size=64, pix_fmt="yuv420p", rate=10
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as output:
        stream = output.add_stream(
            "libaom-av1" if codec == "av1" else "libx264", rate=rate
        )
        stream.width = stream.height = size
        stream.pix_fmt = pix_fmt
        stream.options = (
            {"cpu-used": "8", "crf": "20"} if codec == "av1" else {"crf": "18"}
        )
        for value in values:
            frame = av.VideoFrame.from_ndarray(
                np.full((size, size, 3), value, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def _source(root: Path, version: str) -> None:
    _write_v21(root)
    info = json.loads((root / "meta/info.json").read_text())
    info["codebase_version"] = version
    for key, codec in CAMERAS.items():
        info["features"][key] = {
            "dtype": "video",
            "shape": [64, 64, 3],
            "info": {
                "video.codec": codec,
                "video.fps": 10,
                "video.pix_fmt": "yuv420p",
                "video.crf": 37,
                "video.preset": 12,
            },
        }
    if version == "v3.0":
        info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        info["video_path"] = (
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        )
        paths = sorted((root / "data/chunk-000").glob("*.parquet"))
        pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True).to_parquet(
            root / "data/chunk-000/file-000.parquet", index=False
        )
        for path in paths:
            path.unlink()
        rows = []
        for episode in range(2):
            row = {
                "episode_index": episode,
                "length": 10,
                "tasks": ["pick"],
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": episode * 10,
                "dataset_to_index": (episode + 1) * 10,
            }
            for key in CAMERAS:
                row.update(
                    {
                        f"videos/{key}/chunk_index": 0,
                        f"videos/{key}/file_index": 0,
                        f"videos/{key}/from_timestamp": float(episode),
                        f"videos/{key}/to_timestamp": float(episode + 1),
                    }
                )
            rows.append(row)
        meta = root / "meta/episodes/chunk-000/file-000.parquet"
        meta.parent.mkdir(parents=True)
        pd.DataFrame(rows).to_parquet(meta, index=False)
        pd.DataFrame([{"task_index": 0, "task": "pick"}]).to_parquet(
            root / "meta/tasks.parquet", index=False
        )
        for key, codec in CAMERAS.items():
            _encode(
                root / f"videos/{key}/chunk-000/file-000.mp4",
                codec,
                [20 + i * 8 for i in range(20)],
            )
    else:
        for key, codec in CAMERAS.items():
            for episode in range(2):
                _encode(
                    root / f"videos/chunk-000/{key}/episode_{episode:06d}.mp4",
                    codec,
                    [20 + i * 8 for i in range(episode * 10, episode * 10 + 10)],
                )
    (root / "meta/info.json").write_text(json.dumps(info))


@pytest.mark.parametrize("version", ["v2.1", "v3.0"])
def test_trim_materialization_follows_each_source_camera_codec(
    tmp_path: Path, version: str
) -> None:
    settings = _settings(tmp_path)
    source_root = settings.nas_root / "raw/lab/source"
    _source(source_root, version)
    before = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_root.rglob("*")
        if p.is_file()
    }
    database = Database(settings.database_path)
    database.initialize()
    record = inspect_dataset(
        area_root=settings.nas_root / "raw",
        storage_area="raw",
        relative_path="lab/source",
    ).as_record()
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw", records=[record], scan_generation=generation
    )
    dataset = database.list_datasets()[0]
    profile = database.create_profile("Trim codec regression")
    recipe = database.create_curation_recipe(
        dataset_id=dataset["id"],
        profile_id=profile["id"],
        name="Trim",
        selection_mode="all",
        trim_config={
            "enabled": True,
            "episode_overrides": {
                str(i): {"start_frame": 2, "end_frame": 7} for i in range(2)
            },
        },
    )
    snapshot = database.snapshot_curation_recipe(recipe["id"], profile_id=profile["id"])
    job, _ = database.create_job(
        kind="curation.materialize",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={"snapshot_id": snapshot["id"], "output_name": "trimmed"},
        idempotency_key="trim-codec",
    )
    database.claim_job(job["id"], worker_id="test", lease_seconds=600)
    result = materialize_curation_recipe(
        database=database,
        settings=settings,
        payload=job["payload"],
        job_id=job["id"],
        worker_id="test",
    )
    output = settings.nas_root / "derived" / result["outputs"][0]["relative_path"]
    info = json.loads((output / "meta/info.json").read_text())
    source = DatasetSource(output, info)
    assert info["total_frames"] == 10
    for episode in range(2):
        data, metadata = source.episode(episode)
        assert len(data) == 5
        assert data["frame_index"].tolist() == list(range(5))
        assert data["timestamp"].tolist() == pytest.approx([i / 10 for i in range(5)])
        for key, codec in CAMERAS.items():
            path, _ = source.video_source(episode, key, metadata)
            with av.open(str(path)) as video:
                assert (
                    video.streams.video[0].codec_context.codec.id
                    == av.codec.Codec(codec, "r").id
                )
                frames = list(video.decode(video=0))
            assert len(frames) == 5
            assert [float(f.pts * f.time_base) for f in frames] == pytest.approx(
                [i / 10 for i in range(5)]
            )
            assert [
                f.to_ndarray(format="rgb24").mean() for f in frames
            ] == pytest.approx(
                [20 + i * 8 for i in range(episode * 10 + 2, episode * 10 + 7)], abs=8
            )
            assert info["features"][key]["info"]["video.codec"] == codec
    report = validate_dataset_root(output, mode="full")
    assert report["passed"], report["issues"]
    assert before == {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_root.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize(
    ("size", "pix_fmt"), [(32, "yuv420p"), (64, "yuv420p10le"), (64, "yuv444p")]
)
def test_av1_trim_preserves_format_and_fractional_fps(tmp_path, size, pix_fmt):
    rate = Fraction(30000, 1001)
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    _encode(
        source, "av1", list(range(20, 120, 10)), size=size, pix_fmt=pix_fmt, rate=rate
    )
    dataset_video.slice_video(
        source, output, 2, 7, float(rate), 5, codec="av1", expected_source_codec="av1"
    )
    with av.open(str(output)) as video:
        assert (
            video.streams.video[0].codec_context.codec.id
            == av.codec.Codec("av1", "r").id
        )
        assert float(video.streams.video[0].average_rate) == pytest.approx(float(rate))
        frames = list(video.decode(video=0))
    assert len(frames) == 5
    assert frames[0].format.name == pix_fmt
    assert [(frame.width, frame.height) for frame in frames] == [(size, size)] * 5
    assert [float(frame.pts * frame.time_base) for frame in frames] == pytest.approx(
        [float(i / rate) for i in range(5)], abs=1e-5
    )


def test_missing_av1_encoder_never_falls_back_to_h264(monkeypatch):
    attempted = []

    def unavailable(name, mode):
        attempted.append(name)
        raise ValueError("not installed")

    monkeypatch.setattr(av.codec, "Codec", unavailable)
    with pytest.raises(CurationTransformError, match="encoder"):
        dataset_video.video_encoder("av1")
    assert attempted and all(name in {"libsvtav1", "libaom-av1"} for name in attempted)


def test_unknown_source_codec_fails_explicitly():
    with pytest.raises(CurationTransformError, match="not supported"):
        dataset_video.normalize_video_codec("unsupported_codec")


@pytest.mark.parametrize("codec", ["av1", "h264"])
def test_long_trim_timestamp_base_does_not_change_after_mux_starts(tmp_path, codec):
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    _encode(source, codec, [50 + i % 100 for i in range(200)])
    dataset_video.slice_video(
        source, output, 5, 190, 10, 185, codec=codec, expected_source_codec=codec
    )
    with av.open(str(output)) as video:
        frames = list(video.decode(video=0))
    assert len(frames) == 185
    assert [float(frame.pts * frame.time_base) for frame in frames] == pytest.approx(
        [i / 10 for i in range(185)]
    )


def test_trim_metadata_updates_all_supported_codec_locations():
    key = "observation.images.front"
    info = {
        "features": {
            key: {
                "dtype": "video",
                "shape": [64, 64, 3],
                "info": {"video.codec": "h264", "video.crf": 99},
                "video_info": {"video.codec": "h264", "video.crf": 99},
                "video": {"codec": "h264", "crf": 99},
                "video.codec": "h264",
                "video.crf": 99,
            }
        }
    }
    updated = transforms._updated_info(info, [], {}, {key: "av1"})["features"][key]
    assert updated["info"]["video.codec"] == "av1"
    assert updated["video_info"]["video.codec"] == "av1"
    assert updated["video"]["codec"] == "av1"
    assert updated["video.codec"] == "av1"
    assert "video.crf" not in updated
    assert "video.crf" not in updated["info"]
    assert "video.crf" not in updated["video_info"]
    assert "crf" not in updated["video"]
    assert info["features"][key]["info"]["video.codec"] == "h264"
