from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from datasetui.database import Database
from datasetui.content_integrity import dataset_content_fingerprint
from datasetui.datasets import inspect_dataset
from datasetui.segmentation import (
    SegmentationError,
    create_preview,
    dataset_scope,
    export_preview,
    source_frame,
)
from test_transforms import _settings
from test_validation_conversion import _write_v3


class DeterministicEngine:
    def propagate(self, *, video_path, prompts, frame_count, output_dir, check_lease):
        with av.open(str(video_path)) as container:
            frame = next(container.decode(video=0))
            width, height = frame.width, frame.height
        for target in ("replace", "protect"):
            (output_dir / target).mkdir()
        for index in range(frame_count):
            replace = np.zeros((height, width), dtype=np.uint8)
            replace[:, : width // 2] = 255
            protect = np.zeros((height, width), dtype=np.uint8)
            protect[height // 2, width // 4] = 255
            Image.fromarray(replace, mode="L").save(
                output_dir / "replace" / f"{index:06d}.png"
            )
            Image.fromarray(protect, mode="L").save(
                output_dir / "protect" / f"{index:06d}.png"
            )
            check_lease()
        return {
            "model": "deterministic-test",
            "checkpoint_sha256": "a" * 64,
            "runtime": "test",
        }


def _write_video(path: Path, colors: list[tuple[int, int, int]], fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = 16
    stream.height = 16
    stream.pix_fmt = "yuv420p"
    for color in colors:
        array = np.empty((16, 16, 3), dtype=np.uint8)
        array[:] = color
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _background() -> str:
    image = Image.new("RGB", (8, 8), (255, 0, 0))
    stream = BytesIO()
    image.save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode()


def _nested_scalar(value) -> float:
    while isinstance(value, (np.ndarray, list)):
        value = value[0] if isinstance(value, list) else value.reshape(-1)[0]
    return float(value)


def _dataset(root: Path) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "rby1",
        "total_episodes": 1,
        "total_frames": 4,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": 10,
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [1], "names": ["joint"]},
            "observation.state": {
                "dtype": "float32",
                "shape": [1],
                "names": ["joint"],
            },
            "observation.images.top": {
                "dtype": "video",
                "shape": [16, 16, 3],
                "names": ["height", "width", "channel"],
                "info": {
                    "video.fps": 10,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "has_audio": False,
                },
            },
            "observation.images.side": {
                "dtype": "video",
                "shape": [16, 16, 3],
                "names": ["height", "width", "channel"],
                "info": {"video.fps": 10, "video.codec": "h264"},
            },
        },
    }
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        info["features"][key] = {"dtype": "float32" if key == "timestamp" else "int64", "shape": [1], "names": None}
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "pick"}) + "\n", encoding="utf-8"
    )
    (root / "meta/episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 4, "tasks": ["pick"]}) + "\n",
        encoding="utf-8",
    )
    (root / "meta/stats.json").write_text("{}", encoding="utf-8")
    frame = pd.DataFrame(
        {
            "action": [np.asarray([index], dtype=np.float32) for index in range(4)],
            "observation.state": [np.asarray([index], dtype=np.float32) for index in range(4)],
            "timestamp": (np.arange(4) / 10).astype(np.float32),
            "frame_index": np.arange(4),
            "episode_index": [0] * 4,
            "index": np.arange(4),
            "task_index": [0] * 4,
        }
    )
    frame.to_parquet(root / "data/chunk-000/episode_000000.parquet", index=False)
    colors = [
        (0, 0, 200),
        (0, 100, 200),
        (0, 200, 100),
        (0, 200, 0),
        (17, 33, 49),
        (71, 89, 107),
    ]
    _write_video(
        root / "videos/chunk-000/observation.images.top/episode_000000.mp4",
        colors[:4],
    )
    _write_video(
        root / "videos/chunk-000/observation.images.side/episode_000000.mp4",
        [(100, 100, 100)] * 4,
    )
    (root / "README.md").write_text("preserve me\n", encoding="utf-8")
    from datasetui.transforms import _write_stats
    _write_stats(root / "meta/stats.json", [frame])


def _registered(tmp_path: Path):
    settings = _settings(tmp_path)
    root = settings.nas_root / "raw/lab/source"
    _dataset(root)
    database = Database(settings.database_path)
    database.initialize()
    candidate = inspect_dataset(
        area_root=settings.nas_root / "raw",
        storage_area="raw",
        relative_path="lab/source",
    )
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw", records=[candidate.as_record()], scan_generation=generation
    )
    dataset = database.list_datasets()[0]
    dataset["fingerprint"] = dataset_content_fingerprint(root)
    return settings, database, root, dataset


def _spec(dataset: dict) -> dict:
    return {
        "dataset_id": dataset["id"],
        "fingerprint": dataset["fingerprint"],
        "episode_index": 0,
        "video_key": "observation.images.top",
        "mode": "replace_background",
        "background_base64": _background(),
        "prompts": [
            {
                "frame_index": 0,
                "target": "replace",
                "text": "background",
            }
        ],
        "corrections": [
            {
                "frame_index": 1,
                "target": "protect",
                "radius": 0.1,
                "points": [{"x": 0.25, "y": 0.5}],
            },
            {
                "frame_index": 1,
                "target": "protect",
                "operation": "add",
                "radius": 0.2,
                "points": [{"x": 0.75, "y": 0.5}],
            },
            {
                "frame_index": 1,
                "target": "protect",
                "operation": "erase",
                "radius": 0.05,
                "points": [{"x": 0.75, "y": 0.5}],
            },
        ],
    }


def _preview(settings, database, dataset):
    profile = database.create_profile("Segmenter")
    job, _ = database.create_job(
        kind="segmentation.preview",
        queue_name="gpu",
        profile_id=profile["id"],
        payload={"spec": _spec(dataset)},
        idempotency_key="preview",
    )
    database.claim_job(job["id"], worker_id="gpu", lease_seconds=120)
    result = create_preview(
        database,
        settings,
        job_id=job["id"],
        worker_id="gpu",
        spec=job["payload"]["spec"],
        engine=DeterministicEngine(),
    )
    reused = create_preview(
        database,
        settings,
        job_id=job["id"],
        worker_id="gpu",
        spec=job["payload"]["spec"],
        engine=DeterministicEngine(),
    )
    assert reused["reused"] is True
    assert reused["artifact_fingerprint"] == result["artifact_fingerprint"]
    database.succeed_job(job["id"], result, worker_id="gpu")
    return profile, job, result


def test_scope_source_frame_and_preview_render_full_protected_clip(
    tmp_path: Path,
) -> None:
    settings, database, _, dataset = _registered(tmp_path)

    assert dataset_scope(database, settings, dataset["id"])["episodes"] == [
        {"episode_index": 0, "length": 4}
    ]
    with Image.open(
        BytesIO(
            source_frame(
                database, settings, dataset["id"], 0, "observation.images.top", 2
            )
        )
    ) as image:
        assert image.size == (16, 16)

    _, job, result = _preview(settings, database, dataset)
    preview = settings.jobs_root / "segmentation" / job["id"]
    assert result["frame_count"] == 4
    assert len(result["artifact_fingerprint"]) == 64
    assert set(result["artifact_hashes"]) == {
        "original.mp4",
        "composite.mp4",
        "mask.mp4",
    }
    assert all(
        (preview / "replace" / f"{index:06d}.png").is_file() for index in range(4)
    )
    assert all(
        (preview / name).is_file()
        for name in ("original.mp4", "composite.mp4", "mask.mp4")
    )
    for name in ("original.mp4", "composite.mp4", "mask.mp4"):
        with av.open(str(preview / name)) as container:
            assert container.streams.video[0].codec_context.format.name == "yuv420p"
    manifest = json.loads((preview / "manifest.json").read_text(encoding="utf-8"))
    assert "background_base64" not in manifest["spec"]
    assert len(manifest["spec"]["background_sha256"]) == 64
    corrected = np.asarray(Image.open(preview / "protect/000001.png"))
    assert corrected[8, 4] == 255
    assert corrected[8, 11] == 0
    assert corrected[8, 9] == 255


def test_export_preserves_dataset_and_changes_only_selected_video_shard(
    tmp_path: Path,
) -> None:
    settings, database, source, dataset = _registered(tmp_path)
    profile, preview_job, _ = _preview(settings, database, dataset)
    source_data = (source / "data/chunk-000/episode_000000.parquet").read_bytes()
    side_video = source / "videos/chunk-000/observation.images.side/episode_000000.mp4"
    source_side = side_video.read_bytes()
    source_top = source / "videos/chunk-000/observation.images.top/episode_000000.mp4"
    with av.open(str(source_top)) as container:
        source_frames = [
            frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)
        ]
    export_job, _ = database.create_job(
        kind="segmentation.export",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={"preview_id": preview_job["id"], "output_name": "segmented"},
        idempotency_key="export",
    )
    database.claim_job(export_job["id"], worker_id="cpu", lease_seconds=120)

    result = export_preview(
        database,
        settings,
        job_id=export_job["id"],
        worker_id="cpu",
        preview_id=preview_job["id"],
        output_name="segmented",
        recompute_statistics=True,
    )

    output = settings.nas_root / "derived/segmented"
    assert result["relative_path"] == "segmented"
    assert (output / "README.md").read_text(encoding="utf-8") == "preserve me\n"
    assert (
        output / "data/chunk-000/episode_000000.parquet"
    ).read_bytes() == source_data
    assert (
        output / "videos/chunk-000/observation.images.side/episode_000000.mp4"
    ).read_bytes() == source_side
    with av.open(
        str(output / "videos/chunk-000/observation.images.top/episode_000000.mp4")
    ) as container:
        output_frames = [
            frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)
        ]
    assert len(output_frames) == len(source_frames)
    assert len(output_frames) == 4  # v2 videos contain exactly the episode frames.
    assert json.loads((output / "meta/stats.json").read_text())[
        "observation.images.top"
    ]["count"] == [4]
    stats = json.loads((output / "meta/stats.json").read_text())[
        "observation.images.top"
    ]
    assert 0 <= stats["min"][0][0][0] <= stats["max"][0][0][0] <= 1
    assert (
        settings.nas_root / f"manifests/segmentation/{export_job['id']}.json"
    ).is_file()


def test_export_fails_closed_for_modified_preview_and_existing_output(
    tmp_path: Path,
) -> None:
    settings, database, _, dataset = _registered(tmp_path)
    profile, preview_job, _ = _preview(settings, database, dataset)
    job, _ = database.create_job(
        kind="segmentation.export",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={},
        idempotency_key="stale-export",
    )
    database.claim_job(job["id"], worker_id="cpu", lease_seconds=120)
    (settings.nas_root / "derived/existing").mkdir()
    with pytest.raises(SegmentationError, match="already exists"):
        export_preview(
            database,
            settings,
            job_id=job["id"],
            worker_id="cpu",
            preview_id=preview_job["id"],
            output_name="existing",
        )

    mask = settings.jobs_root / f"segmentation/{preview_job['id']}/replace/000000.png"
    mask.write_bytes(mask.read_bytes() + b"changed")

    with pytest.raises(SegmentationError, match="stale"):
        export_preview(
            database,
            settings,
            job_id=job["id"],
            worker_id="cpu",
            preview_id=preview_job["id"],
            output_name="stale",
        )


def test_v3_export_preserves_shared_shard_offsets_and_unselected_episode(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    source = settings.nas_root / "raw/lab/v3"
    _write_v3(source)
    info_path = source / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["observation.images.top"] = {
        "dtype": "video",
        "shape": [16, 16, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.fps": 10,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "has_audio": False,
        },
    }
    info_path.write_text(json.dumps(info), encoding="utf-8")
    metadata_path = source / "meta/episodes/chunk-000/file-000.parquet"
    metadata = pd.read_parquet(metadata_path)
    metadata["videos/observation.images.top/chunk_index"] = [0, 0]
    metadata["videos/observation.images.top/file_index"] = [0, 0]
    metadata["videos/observation.images.top/from_timestamp"] = [0.0, 0.4]
    metadata["videos/observation.images.top/to_timestamp"] = [0.4, 0.8]
    for stat_name in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"):
        metadata[f"stats/observation.images.top/{stat_name}"] = pd.Series(
            [
                [[[0.25]], [[0.25]], [[0.25]]],
                [[[0.75]], [[0.75]], [[0.75]]],
            ],
            dtype=object,
        )
    metadata["stats/observation.images.top/count"] = pd.Series([[4], [4]], dtype=object)
    metadata.to_parquet(metadata_path, index=False)
    video_path = source / "videos/observation.images.top/chunk-000/file-000.mp4"
    _write_video(
        video_path,
        [(0, 0, 180)] * 4
        + [(20, 180, 20), (40, 160, 40), (60, 140, 60), (80, 120, 80)],
    )
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
    dataset["fingerprint"] = dataset_content_fingerprint(source)
    profile, preview_job, _ = _preview(settings, database, dataset)
    data_bytes = (source / "data/chunk-000/file-000.parquet").read_bytes()
    with av.open(str(video_path)) as container:
        original = [
            frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)
        ]
    export_job, _ = database.create_job(
        kind="segmentation.export",
        queue_name="io",
        profile_id=profile["id"],
        payload={},
        idempotency_key="v3-export",
    )
    database.claim_job(export_job["id"], worker_id="cpu", lease_seconds=120)

    export_preview(
        database,
        settings,
        job_id=export_job["id"],
        worker_id="cpu",
        preview_id=preview_job["id"],
        output_name="v3-segmented",
        recompute_statistics=True,
    )

    output = settings.nas_root / "derived/v3-segmented"
    assert (output / "data/chunk-000/file-000.parquet").read_bytes() == data_bytes
    output_metadata = pd.read_parquet(
        output / "meta/episodes/chunk-000/file-000.parquet"
    )
    assert output_metadata["dataset_from_index"].tolist() == [0, 4]
    assert output_metadata["dataset_to_index"].tolist() == [4, 8]
    assert [
        int(np.asarray(value).reshape(-1)[0])
        for value in output_metadata["stats/observation.images.top/count"]
    ] == [4, 4]
    unchanged = output_metadata.loc[1, "stats/observation.images.top/mean"]
    assert _nested_scalar(unchanged) == pytest.approx(0.75)
    changed = output_metadata.loc[0, "stats/observation.images.top/mean"]
    assert 0 <= _nested_scalar(changed) <= 1
    with av.open(
        str(output / "videos/observation.images.top/chunk-000/file-000.mp4")
    ) as container:
        rendered = [
            frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)
        ]
    assert all(
        np.array_equal(rendered[index], original[index]) for index in range(4, 8)
    )
    assert json.loads((output / "meta/stats.json").read_text())[
        "observation.images.top"
    ]["count"] == [8]


def test_export_retry_recovers_same_job_publish_and_rejects_corruption(
    tmp_path: Path, monkeypatch
) -> None:
    settings, database, _, dataset = _registered(tmp_path)
    profile, preview_job, _ = _preview(settings, database, dataset)
    export_job, _ = database.create_job(
        kind="segmentation.export",
        queue_name="io",
        profile_id=profile["id"],
        payload={},
        idempotency_key="recovery-export",
    )
    database.claim_job(export_job["id"], worker_id="cpu", lease_seconds=120)
    original_scan = database.begin_dataset_scan
    monkeypatch.setattr(
        database,
        "begin_dataset_scan",
        lambda storage_area: (_ for _ in ()).throw(RuntimeError("crash after rename")),
    )
    with pytest.raises(RuntimeError, match="crash after rename"):
        export_preview(
            database,
            settings,
            job_id=export_job["id"],
            worker_id="cpu",
            preview_id=preview_job["id"],
            output_name="recoverable",
        )
    monkeypatch.setattr(database, "begin_dataset_scan", original_scan)

    recovered = export_preview(
        database,
        settings,
        job_id=export_job["id"],
        worker_id="cpu",
        preview_id=preview_job["id"],
        output_name="recoverable",
    )
    assert recovered["relative_path"] == "recoverable"
    readme = settings.nas_root / "derived/recoverable/README.md"
    readme.write_text("corrupt\n", encoding="utf-8")
    with pytest.raises(SegmentationError, match="modified"):
        export_preview(
            database,
            settings,
            job_id=export_job["id"],
            worker_id="cpu",
            preview_id=preview_job["id"],
            output_name="recoverable",
        )
