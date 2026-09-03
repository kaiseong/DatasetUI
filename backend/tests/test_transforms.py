from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datasetui.config import Settings
from datasetui.database import Database, RecipeRevisionMismatchError
from datasetui.datasets import inspect_dataset
from datasetui.transforms import materialize_curation_recipe
from datasetui.transforms import _replace_language_columns, _slice_video
from datasetui.transform_errors import CurationTransformError


def _settings(tmp_path: Path) -> Settings:
    nas = tmp_path / "nas"
    for name in ("raw", "derived", "exports", "manifests"):
        (nas / name).mkdir(parents=True)
    for name in ("cache", "staging", "jobs"):
        (tmp_path / name).mkdir()
    return Settings(
        database_path=tmp_path / "registry.sqlite3",
        redis_url="redis://unused",
        allowed_origins=("https://example.test",),
        nas_root=nas,
        cache_root=tmp_path / "cache",
        staging_root=tmp_path / "staging",
        jobs_root=tmp_path / "jobs",
    )


def _write_v21(root: Path) -> None:
    info = {
        "codebase_version": "v2.1",
        "robot_type": "rby1",
        "total_episodes": 2,
        "total_frames": 20,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": 10,
        "splits": {"train": "0:2"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [1], "names": ["joint_0"]},
            "observation.state": {
                "dtype": "float32",
                "shape": [1],
                "names": ["joint_0"],
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "pick"}) + "\n", encoding="utf-8"
    )
    episode_lines = []
    for episode in range(2):
        values = np.array([0, 0, 0, 1, 2, 3, 3, 3, 3, 3], dtype=np.float32)
        frame = pd.DataFrame(
            {
                "action": [[value] for value in values],
                "observation.state": [[value] for value in values],
                "timestamp": np.arange(10) / 10,
                "frame_index": np.arange(10),
                "episode_index": episode,
                "index": np.arange(episode * 10, episode * 10 + 10),
                "task_index": 0,
            }
        )
        frame.to_parquet(
            root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet",
            index=False,
        )
        episode_lines.append(
            json.dumps({"episode_index": episode, "tasks": ["pick"], "length": 10})
        )
    (root / "meta" / "episodes.jsonl").write_text(
        "\n".join(episode_lines) + "\n", encoding="utf-8"
    )


def test_materialize_train_eval_is_immutable_trimmed_and_reusable(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    source = settings.nas_root / "raw" / "lab" / "pick"
    _write_v21(source)
    database = Database(settings.database_path)
    database.initialize()
    candidate = inspect_dataset(
        area_root=settings.nas_root / "raw",
        storage_area="raw",
        relative_path="lab/pick",
    )
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw",
        records=[candidate.as_record()],
        scan_generation=generation,
    )
    dataset = database.list_datasets()[0]
    profile = database.create_profile("Researcher")
    database.update_episode_flags(
        dataset_id=dataset["id"],
        profile_id=profile["id"],
        expected_revision=0,
        changes=[{"episode_index": 1, "flagged": True}],
    )
    recipe = database.create_curation_recipe(
        dataset_id=dataset["id"],
        profile_id=profile["id"],
        name="Train eval",
        selection_mode="all",
        operation="train_eval_split",
        trim_config={
            "enabled": True,
            "threshold": 0.1,
            "hold_time_s": 0.1,
            "margin_s": 0,
            "dimensions": ["joint_0"],
            "episode_overrides": {"1": {"start_frame": 2, "end_frame": 8}},
        },
    )
    snapshot = database.snapshot_curation_recipe(recipe["id"], profile_id=profile["id"])
    job, _ = database.create_job(
        kind="curation.materialize",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={"snapshot_id": snapshot["id"], "output_name": "pick-clean"},
        idempotency_key="run-1",
    )
    worker = "test-worker"
    database.claim_job(job["id"], worker_id=worker, lease_seconds=120)

    result = materialize_curation_recipe(
        database=database,
        settings=settings,
        payload=job["payload"],
        job_id=job["id"],
        worker_id=worker,
    )
    assert [item["role"] for item in result["outputs"]] == ["train", "eval"]
    assert result["outputs"][1]["frames"] == 6
    assert result["outputs"][1]["lineage"][0]["trim_method"] == "manual"
    assert json.loads((source / "meta" / "info.json").read_text())["total_frames"] == 20

    for output in result["outputs"]:
        root = settings.nas_root / "derived" / output["relative_path"]
        info = json.loads((root / "meta" / "info.json").read_text())
        assert info["total_episodes"] == 1
        data = pd.read_parquet(next((root / "data").rglob("*.parquet")))
        assert data["episode_index"].tolist() == [0] * len(data)
        assert data["frame_index"].tolist() == list(range(len(data)))
        assert data["index"].tolist() == list(range(len(data)))

    repeated = materialize_curation_recipe(
        database=database,
        settings=settings,
        payload=job["payload"],
        job_id=job["id"],
        worker_id=worker,
    )
    assert repeated["reused"] is True


def test_materialize_rejects_a_source_changed_after_recipe_snapshot(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    source = settings.nas_root / "raw" / "lab" / "changed"
    _write_v21(source)
    database = Database(settings.database_path)
    database.initialize()
    candidate = inspect_dataset(
        area_root=settings.nas_root / "raw",
        storage_area="raw",
        relative_path="lab/changed",
    )
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw",
        records=[candidate.as_record()],
        scan_generation=generation,
    )
    dataset = database.list_datasets()[0]
    profile = database.create_profile("Revision guard")
    recipe = database.create_curation_recipe(
        dataset_id=dataset["id"],
        profile_id=profile["id"],
        name="Immutable source",
        selection_mode="all",
    )
    snapshot = database.snapshot_curation_recipe(recipe["id"], profile_id=profile["id"])
    job, _ = database.create_job(
        kind="curation.materialize",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={"snapshot_id": snapshot["id"], "output_name": "changed-output"},
        idempotency_key="changed-source-run",
    )
    database.claim_job(job["id"], worker_id="guard-worker", lease_seconds=120)

    info_path = source / "meta" / "info.json"
    changed_info = json.loads(info_path.read_text(encoding="utf-8"))
    changed_info["total_frames"] = 999
    info_path.write_text(json.dumps(changed_info), encoding="utf-8")

    with pytest.raises(RecipeRevisionMismatchError):
        materialize_curation_recipe(
            database=database,
            settings=settings,
            payload=job["payload"],
            job_id=job["id"],
            worker_id="guard-worker",
        )
    assert list((settings.nas_root / "derived").iterdir()) == []


def test_exact_video_slice_decodes_and_reencodes_requested_frames(
    tmp_path: Path,
) -> None:
    import av

    source = tmp_path / "source.mp4"
    container = av.open(str(source), mode="w")
    stream = container.add_stream("libx264", rate=10)
    stream.width = 32
    stream.height = 24
    stream.pix_fmt = "yuv420p"
    for value in range(10):
        frame = av.VideoFrame.from_ndarray(
            np.full((24, 32, 3), value * 10, dtype=np.uint8), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    output = tmp_path / "trimmed.mp4"
    _slice_video(source, output, 2, 7, 10, 5)
    decoded = av.open(str(output))
    try:
        assert sum(1 for _ in decoded.decode(video=0)) == 5
    finally:
        decoded.close()


def test_materialize_v3_rebuilds_shards_and_episode_offsets(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    source = settings.nas_root / "raw" / "lab" / "v3-pick"
    (source / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (source / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "rby1",
        "total_episodes": 2,
        "total_frames": 8,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": 10,
        "splits": {"train": "0:2"},
        "features": {
            "action": {"dtype": "float32", "shape": [1], "names": ["joint_0"]},
            "observation.state": {
                "dtype": "float32",
                "shape": [1],
                "names": ["joint_0"],
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (source / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    pd.DataFrame([{"task_index": 0, "task": "pick"}]).to_parquet(
        source / "meta" / "tasks.parquet", index=False
    )
    pd.DataFrame(
        [
            {
                "episode_index": index,
                "tasks": ["pick"],
                "length": 4,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": index * 4,
                "dataset_to_index": index * 4 + 4,
            }
            for index in range(2)
        ]
    ).to_parquet(source / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pd.DataFrame(
        {
            "action": [[value] for value in range(8)],
            "observation.state": [[value] for value in range(8)],
            "timestamp": [0, 0.1, 0.2, 0.3] * 2,
            "frame_index": [0, 1, 2, 3] * 2,
            "episode_index": [0] * 4 + [1] * 4,
            "index": range(8),
            "task_index": 0,
        }
    ).to_parquet(source / "data" / "chunk-000" / "file-000.parquet", index=False)

    database = Database(settings.database_path)
    database.initialize()
    candidate = inspect_dataset(
        area_root=settings.nas_root / "raw",
        storage_area="raw",
        relative_path="lab/v3-pick",
    )
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw", records=[candidate.as_record()], scan_generation=generation
    )
    dataset = database.list_datasets()[0]
    profile = database.create_profile("V3 runner")
    database.update_episode_flags(
        dataset_id=dataset["id"],
        profile_id=profile["id"],
        expected_revision=0,
        changes=[{"episode_index": 1, "flagged": True}],
    )
    recipe = database.create_curation_recipe(
        dataset_id=dataset["id"],
        profile_id=profile["id"],
        name="Flag only",
        selection_mode="flagged",
    )
    snapshot = database.snapshot_curation_recipe(recipe["id"], profile_id=profile["id"])
    job, _ = database.create_job(
        kind="curation.materialize",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={"snapshot_id": snapshot["id"], "output_name": "v3-clean"},
        idempotency_key="v3-run",
    )
    database.claim_job(job["id"], worker_id="v3-worker", lease_seconds=120)
    result = materialize_curation_recipe(
        database=database,
        settings=settings,
        payload=job["payload"],
        job_id=job["id"],
        worker_id="v3-worker",
    )
    output = settings.nas_root / "derived" / result["outputs"][0]["relative_path"]
    output_info = json.loads((output / "meta" / "info.json").read_text())
    assert output_info["codebase_version"] == "v3.0"
    assert output_info["total_episodes"] == 1
    metadata = pd.read_parquet(
        output / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).iloc[0]
    assert metadata["episode_index"] == 0
    assert metadata["dataset_from_index"] == 0
    assert metadata["dataset_to_index"] == 4


def test_materialize_applies_frozen_annotations_without_touching_source(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    source = settings.nas_root / "raw" / "lab" / "annotated"
    _write_v21(source)
    database = Database(settings.database_path)
    database.initialize()
    candidate = inspect_dataset(
        area_root=settings.nas_root / "raw",
        storage_area="raw",
        relative_path="lab/annotated",
    )
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw", records=[candidate.as_record()], scan_generation=generation
    )
    dataset = database.list_datasets()[0]
    profile = database.create_profile("Annotator")
    source_parquet = source / "data/chunk-000/episode_000001.parquet"
    source_digest = source_parquet.read_bytes()
    database.update_episode_flags(
        dataset_id=dataset["id"],
        profile_id=profile["id"],
        expected_revision=0,
        changes=[{"episode_index": 1, "flagged": True}],
    )
    database.replace_episode_annotations(
        dataset_id=dataset["id"],
        profile_id=profile["id"],
        episode_index=1,
        expected_revision=0,
        task_override="place the cup carefully",
        atoms=[
            {
                "role": "assistant",
                "content": "grasp the cup",
                "style": "subtask",
                "timestamp": 0.1,
                "camera": None,
                "tool_calls": None,
            },
            {
                "role": "assistant",
                "content": "discarded memory",
                "style": "memory",
                "timestamp": 0.1,
                "camera": None,
                "tool_calls": None,
            },
            {
                "role": "user",
                "content": "move more slowly",
                "style": "interjection",
                "timestamp": 0.4,
                "camera": None,
                "tool_calls": None,
            },
            {
                "role": "assistant",
                "content": None,
                "style": None,
                "timestamp": 0.5,
                "camera": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "say",
                            "arguments": {"text": "understood"},
                        },
                    }
                ],
            },
            {
                "role": "user",
                "content": "outside trim",
                "style": "interjection",
                "timestamp": 0.9,
                "camera": None,
                "tool_calls": None,
            },
        ],
    )
    recipe = database.create_curation_recipe(
        dataset_id=dataset["id"],
        profile_id=profile["id"],
        name="Annotated trim",
        selection_mode="flagged",
        trim_config={
            "enabled": True,
            "threshold": 0.1,
            "hold_time_s": 0.1,
            "margin_s": 0,
            "dimensions": ["joint_0"],
            "episode_overrides": {"1": {"start_frame": 2, "end_frame": 8}},
        },
        include_annotations=True,
    )
    snapshot = database.snapshot_curation_recipe(recipe["id"], profile_id=profile["id"])
    job, _ = database.create_job(
        kind="curation.materialize",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={"snapshot_id": snapshot["id"], "output_name": "annotated-clean"},
        idempotency_key="annotated-run",
    )
    database.claim_job(job["id"], worker_id="annotation-worker", lease_seconds=120)

    result = materialize_curation_recipe(
        database=database,
        settings=settings,
        payload=job["payload"],
        job_id=job["id"],
        worker_id="annotation-worker",
    )
    output = settings.nas_root / "derived" / result["outputs"][0]["relative_path"]
    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    data = pd.read_parquet(next((output / "data").rglob("*.parquet")))
    tasks = [
        json.loads(line)
        for line in (output / "meta/tasks.jsonl").read_text().splitlines()
    ]

    assert source_parquet.read_bytes() == source_digest
    assert tasks == [{"task": "place the cup carefully", "task_index": 0}]
    assert data["task_index"].tolist() == [0] * 6
    assert set(info["features"]) >= {"language_persistent", "language_events"}
    assert info["tools"][0]["function"]["name"] == "say"
    persistent = data["language_persistent"].iloc[0]
    assert len(persistent) == 1
    assert persistent[0]["style"] == "subtask"
    assert persistent[0]["timestamp"] == pytest.approx(0)
    event_rows = [list(rows) for rows in data["language_events"]]
    assert [len(rows) for rows in event_rows] == [0, 0, 1, 1, 0, 0]
    assert "timestamp" not in event_rows[2][0]
    lineage = result["outputs"][0]["lineage"][0]
    assert lineage["task_overridden"] is True
    assert lineage["persistent_annotations"] == 1
    assert lineage["event_annotations"] == 2


def test_vqa_annotations_require_and_land_on_an_available_camera_frame() -> None:
    source = pd.DataFrame(
        {
            "timestamp": [0.0, 0.1, 0.2],
            "frame_index": [0, 1, 2],
            "episode_index": [0, 0, 0],
        }
    )
    output = source.copy()
    atom = {
        "role": "assistant",
        "content": '{"label":"cube","count":1}',
        "style": "vqa",
        "timestamp": 0.11,
        "camera": "observation.images.top",
        "tool_calls": None,
    }

    persistent, events = _replace_language_columns(
        output,
        source_data=source,
        atoms=[atom],
        start=0,
        end=3,
        fps=10,
        video_keys=["observation.images.top"],
    )
    assert persistent == 0
    assert events == 1
    assert [len(rows) for rows in output["language_events"]] == [0, 1, 0]
    assert output["language_events"].iloc[1][0]["camera"] == "observation.images.top"

    with pytest.raises(CurationTransformError, match="unavailable camera"):
        _replace_language_columns(
            source.copy(),
            source_data=source,
            atoms=[atom],
            start=0,
            end=3,
            fps=10,
            video_keys=[],
        )
