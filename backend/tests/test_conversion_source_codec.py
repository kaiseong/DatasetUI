from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datasetui.conversion import (
    CONVERSION_ENGINE_ID,
    VIDEO_CODEC_POLICY,
    convert_dataset_to_v21,
)
from datasetui.database import Database
from datasetui.datasets import inspect_dataset
from datasetui.transform_errors import CurationTransformError
from datasetui.transforms import _tree_manifest
from test_transforms import _settings
from test_validation_conversion import _exact_stats, _write_v3


def _add_shared_video(root: Path, codec: str) -> str:
    import av

    video_key = "observation.images.top"
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"][video_key] = {
        "dtype": "video",
        "shape": [64, 64, 3],
        "names": ["height", "width", "channels"],
        "info": {"video.codec": codec, "video.fps": 10},
    }
    info_path.write_text(json.dumps(info), encoding="utf-8")

    metadata_path = root / "meta/episodes/chunk-000/file-000.parquet"
    metadata = pd.read_parquet(metadata_path)
    metadata[f"videos/{video_key}/chunk_index"] = 0
    metadata[f"videos/{video_key}/file_index"] = 0
    metadata[f"videos/{video_key}/from_timestamp"] = [0.0, 0.4]
    metadata[f"videos/{video_key}/to_timestamp"] = [0.4, 0.8]
    metadata.to_parquet(metadata_path, index=False)

    video_path = root / f"videos/{video_key}/chunk-000/file-000.mp4"
    video_path.parent.mkdir(parents=True)
    encoder = "libaom-av1" if codec == "av1" else "libx264"
    with av.open(str(video_path), mode="w") as container:
        stream = container.add_stream(encoder, rate=10)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        stream.options = (
            {"cpu-used": "8", "crf": "20"} if codec == "av1" else {"crf": "18"}
        )
        for value in range(8):
            frame = av.VideoFrame.from_ndarray(
                np.full((64, 64, 3), value * 20, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    with av.open(str(video_path)) as container:
        actual_codec = container.streams.video[0].codec_context.codec.name
        pixels = np.concatenate(
            [
                frame.to_ndarray(format="rgb24").reshape(-1, 3)
                for frame in container.decode(video=0)
            ]
        ).astype(np.float64)
    stats_path = root / "meta/stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    visual_stats = _exact_stats(pixels / 255.0, count=8)
    stats[video_key] = {
        name: value if name == "count" else np.asarray(value).reshape(3, 1, 1).tolist()
        for name, value in visual_stats.items()
    }
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    return actual_codec


def _conversion_context(tmp_path: Path, output_name: str):
    settings = _settings(tmp_path)
    source = settings.nas_root / "raw/lab/v3"
    _write_v3(source)
    database = Database(settings.database_path)
    database.initialize()
    candidate = inspect_dataset(
        area_root=settings.nas_root / "raw",
        storage_area="raw",
        relative_path="lab/v3",
    )
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw", records=[candidate.as_record()], scan_generation=generation
    )
    dataset = database.list_datasets()[0]
    profile = database.create_profile("Codec converter")
    payload = {
        "dataset_id": dataset["id"],
        "fingerprint": dataset["fingerprint"],
        "storage_area": dataset["storage_area"],
        "relative_path": dataset["relative_path"],
        "output_name": output_name,
    }
    job, _ = database.create_job(
        kind="datasets.convert_v21",
        queue_name="converter-v21",
        profile_id=profile["id"],
        payload=payload,
        idempotency_key=f"convert-{output_name}",
    )
    database.claim_job(job["id"], worker_id="converter", lease_seconds=120)
    return settings, source, database, payload, job["id"]


@pytest.mark.parametrize("codec", ["av1", "h264"])
def test_v3_to_v21_conversion_preserves_each_source_codec(
    tmp_path: Path, codec: str
) -> None:
    import av

    settings, source, database, payload, job_id = _conversion_context(
        tmp_path, f"converted-{codec}"
    )
    source_codec = _add_shared_video(source, codec)
    candidate = inspect_dataset(
        area_root=settings.nas_root / "raw",
        storage_area="raw",
        relative_path="lab/v3",
    )
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw", records=[candidate.as_record()], scan_generation=generation
    )
    dataset = database.list_datasets()[0]
    payload["fingerprint"] = dataset["fingerprint"]

    result = convert_dataset_to_v21(
        database=database,
        settings=settings,
        payload=payload,
        job_id=job_id,
        worker_id="converter",
    )

    output = settings.nas_root / "derived" / payload["output_name"]
    output_info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    assert result["conversion_engine"] == CONVERSION_ENGINE_ID
    assert result["video_codec_policy"] == VIDEO_CODEC_POLICY == "source"
    assert (
        output_info["features"]["observation.images.top"]["info"]["video.codec"]
        == codec
    )
    video_paths = sorted(output.rglob("*.mp4"))
    assert len(video_paths) == 2
    for video_path in video_paths:
        with av.open(str(video_path)) as container:
            assert container.streams.video[0].codec_context.codec.name == source_codec


def test_legacy_conversion_manifest_is_not_reused_or_deleted(tmp_path: Path) -> None:
    settings, _, database, payload, job_id = _conversion_context(
        tmp_path, "converted-legacy-cache"
    )
    first = convert_dataset_to_v21(
        database=database,
        settings=settings,
        payload=payload,
        job_id=job_id,
        worker_id="converter",
    )
    output = settings.nas_root / "derived" / payload["output_name"]
    before = _tree_manifest(output)
    reused = convert_dataset_to_v21(
        database=database,
        settings=settings,
        payload=payload,
        job_id=job_id,
        worker_id="converter",
    )
    assert first["reused"] is False
    assert reused["reused"] is True
    assert reused["conversion_engine"] == CONVERSION_ENGINE_ID
    assert reused["video_codec_policy"] == VIDEO_CODEC_POLICY

    manifest_path = settings.nas_root / "manifests/conversion" / f"{job_id}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("conversion_engine")
    manifest["video_codec_policy"] = "h264"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CurationTransformError, match="legacy conversion engine"):
        convert_dataset_to_v21(
            database=database,
            settings=settings,
            payload=payload,
            job_id=job_id,
            worker_id="converter",
        )

    assert output.is_dir()
    assert _tree_manifest(output) == before
