from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from test_segmentation import _write_video
from test_transforms import _settings

from datasetui.augmentation.joint_offset import (
    DEFAULT_RANGES_DEG,
    JointOffsetAugmentationCreate,
    JointOffsetError,
    arm_dimensions,
    augment_joint_offsets,
    draw_offsets,
    validate_internal_payload,
)
from datasetui.database import Database
from datasetui.datasets import inspect_dataset

NAMES = (
    [f"right_arm_{i}" for i in range(7)]
    + [f"left_arm_{i}" for i in range(7)]
    + ["right_gripper_0", "left_gripper_0"]
)
EPISODES, FRAMES = 3, 5


def _values(episode: int) -> np.ndarray:
    base = np.arange(FRAMES * 16, dtype=np.float32).reshape(FRAMES, 16) / 100
    return base + episode


def _info(version: str, names=NAMES) -> dict:
    info = {
        "codebase_version": version,
        "robot_type": "rby1",
        "total_episodes": EPISODES,
        "total_frames": EPISODES * FRAMES,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": 10,
        "splits": {"train": f"0:{EPISODES}"},
        "features": {
            "action": {"dtype": "float32", "shape": [16], "names": names},
            "observation.state": {"dtype": "float32", "shape": [16], "names": names},
        },
    }
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        info["features"][key] = {
            "dtype": "float32" if key == "timestamp" else "int64",
            "shape": [1],
            "names": None,
        }
    return info


def _frame(episode: int) -> pd.DataFrame:
    values = _values(episode)
    return pd.DataFrame(
        {
            "action": list(values),
            "observation.state": list(values + 0.5),
            "timestamp": (np.arange(FRAMES) / 10).astype(np.float32),
            "frame_index": np.arange(FRAMES),
            "episode_index": [episode] * FRAMES,
            "index": np.arange(episode * FRAMES, (episode + 1) * FRAMES),
            "task_index": [0] * FRAMES,
        }
    )


def _write_v21(root: Path, names=NAMES) -> None:
    (root / "meta").mkdir(parents=True)
    info = _info("v2.1", names)
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    info["features"]["observation.images.front"] = {
        "dtype": "video",
        "shape": [16, 16, 3],
        "names": ["height", "width", "channel"],
        "info": {"video.fps": 10, "video.codec": "h264", "video.pix_fmt": "yuv420p"},
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "pick"}) + "\n", encoding="utf-8"
    )
    (root / "meta/episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": e, "length": FRAMES, "tasks": ["pick"]}) + "\n"
            for e in range(EPISODES)
        ),
        encoding="utf-8",
    )
    for episode in range(EPISODES):
        path = root / f"data/chunk-000/episode_{episode:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        _frame(episode).to_parquet(path, index=False)
        _write_video(
            root / f"videos/chunk-000/observation.images.front/episode_{episode:06d}.mp4",
            [(40 * episode, 80, 120)] * FRAMES,
        )


def _write_v3(root: Path) -> None:
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    info = _info("v3.0")
    info["total_chunks"] = 1
    info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    pd.DataFrame([{"task_index": 0, "task": "pick"}]).to_parquet(
        root / "meta/tasks.parquet", index=False
    )
    pd.DataFrame(
        [
            {
                "episode_index": e,
                "tasks": ["pick"],
                "length": FRAMES,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": e * FRAMES,
                "dataset_to_index": (e + 1) * FRAMES,
            }
            for e in range(EPISODES)
        ]
    ).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet", index=False)
    pd.concat([_frame(e) for e in range(EPISODES)], ignore_index=True).to_parquet(
        root / "data/chunk-000/file-000.parquet", index=False
    )


def _registered(tmp_path: Path, writer=_write_v21):
    settings = _settings(tmp_path)
    writer(settings.nas_root / "raw/lab/rby1")
    database = Database(settings.database_path)
    database.initialize()
    candidate = inspect_dataset(
        area_root=settings.nas_root / "raw", storage_area="raw", relative_path="lab/rby1"
    )
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw", records=[candidate.as_record()], scan_generation=generation
    )
    return settings, database, database.list_datasets()[0]


def _run(settings, database, dataset, **overrides):
    payload = {
        "source": {"id": dataset["id"], "fingerprint": dataset["fingerprint"]},
        "ranges_deg": list(DEFAULT_RANGES_DEG),
        "copies": 2,
        "seed": 7,
        "episode_indices": [0, 2],
        "output_name": "rby1-offset",
        **overrides,
    }
    payload = validate_internal_payload(payload)
    profile = database.create_profile(f"augment {len(database.list_profiles())}")
    job, _ = database.create_job(
        kind="augment.joint_offset",
        queue_name="cpu",
        profile_id=profile["id"],
        payload=payload,
        idempotency_key=payload["output_name"],
    )
    database.claim_job(job["id"], worker_id="w", lease_seconds=120)
    result = augment_joint_offsets(
        database=database, settings=settings, payload=payload, job_id=job["id"], worker_id="w"
    )
    return result, settings.nas_root / "derived" / payload["output_name"]


def _episodes(output: Path) -> dict[int, pd.DataFrame]:
    data = pd.concat(pd.read_parquet(p) for p in sorted((output / "data").rglob("*.parquet")))
    return {int(e): frame for e, frame in data.groupby("episode_index")}


@pytest.mark.parametrize("writer", [_write_v21, _write_v3])
def test_copies_shift_state_and_action_by_the_same_per_episode_arm_offset(tmp_path, writer):
    settings, database, dataset = _registered(tmp_path, writer)
    result, output = _run(settings, database, dataset)
    info = json.loads((output / "meta/info.json").read_text())
    meta = json.loads((output / "meta/joint_offset_augmentation.json").read_text())
    episodes = _episodes(output)

    # Originals first, then copy 1 and copy 2 of the selected episodes 0 and 2.
    assert info["total_episodes"] == result["output"]["episodes"] == 3 + 2 * 2
    assert [(e["source_episode_index"], e["copy"]) for e in meta["episodes"]] == [
        (0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (2, 2)
    ]
    assert meta["unit"] == "degree" and meta["seed"] == 7
    for item in meta["episodes"]:
        frame = episodes[item["output_episode_index"]]
        source = item["source_episode_index"]
        action = np.stack(frame["action"].to_list())
        state = np.stack(frame["observation.state"].to_list())
        assert action.dtype == np.float32
        if item["copy"] == 0:
            assert np.array_equal(action, _values(source))
            assert np.array_equal(state, _values(source) + 0.5)
            continue
        expected = np.zeros(16)
        expected[:7] = np.deg2rad(item["offsets_deg"]["right"])
        expected[7:14] = np.deg2rad(item["offsets_deg"]["left"])
        for values, base in ((action, _values(source)), (state, _values(source) + 0.5)):
            delta = values.astype(np.float64) - base.astype(np.float64)
            assert np.allclose(delta, expected, atol=1e-6)  # every frame, both features
        assert delta[:, 14:].max() == 0  # grippers untouched
        for arm in ("right", "left"):
            assert all(
                abs(value) <= bound
                for value, bound in zip(item["offsets_deg"][arm], DEFAULT_RANGES_DEG)
            )
        assert item["offsets_deg"]["right"] != item["offsets_deg"]["left"]
    # Index is contiguous across the whole output.
    assert sorted(
        pd.concat(episodes.values())["index"].tolist()
    ) == list(range(info["total_frames"]))


def test_statistics_are_recomputed_from_the_shifted_values(tmp_path):
    settings, database, dataset = _registered(tmp_path)
    result, output = _run(settings, database, dataset, ranges_deg=[10.0] * 7, copies=1)
    stats = json.loads((output / "meta/stats.json").read_text())
    action = np.concatenate(
        [np.stack(frame["action"].to_list()) for frame in _episodes(output).values()]
    )
    assert np.allclose(stats["action"]["mean"], action.astype(np.float64).mean(axis=0), atol=1e-6)
    assert np.allclose(stats["action"]["max"], action.max(axis=0), atol=1e-6)
    rows = [
        json.loads(line)
        for line in (output / "meta/episodes_stats.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 5
    assert result["statistics"]["source"] == "full-output-recompute"


def test_videos_are_copied_for_every_output_episode(tmp_path):
    settings, database, dataset = _registered(tmp_path)
    _, output = _run(settings, database, dataset, copies=1, episode_indices=[1])
    videos = sorted((output / "videos").rglob("*.mp4"))
    assert len(videos) == 4
    source = settings.nas_root / "raw/lab/rby1/videos/chunk-000/observation.images.front"
    copy = output / "videos/chunk-000/observation.images.front/episode_000003.mp4"
    assert copy.read_bytes() == (source / "episode_000001.mp4").read_bytes()


def test_same_seed_reproduces_and_other_seed_differs():
    first = draw_offsets(3, list(DEFAULT_RANGES_DEG), [0, 1], 2)
    assert first == draw_offsets(3, list(DEFAULT_RANGES_DEG), [0, 1], 2)
    assert first != draw_offsets(4, list(DEFAULT_RANGES_DEG), [0, 1], 2)
    assert [(d["copy"], d["source_episode_index"]) for d in first] == [
        (1, 0), (1, 1), (2, 0), (2, 1)
    ]


def test_datasets_without_arm_joint_names_are_rejected(tmp_path):
    with pytest.raises(JointOffsetError, match="right_arm_0"):
        arm_dimensions(_info("v2.1", names=[f"joint_{i}" for i in range(16)]))
    settings = _settings(tmp_path)
    _write_v21(settings.nas_root / "raw/lab/rby1", names=[f"joint_{i}" for i in range(16)])
    database = Database(settings.database_path)
    database.initialize()
    candidate = inspect_dataset(
        area_root=settings.nas_root / "raw", storage_area="raw", relative_path="lab/rby1"
    )
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw", records=[candidate.as_record()], scan_generation=generation
    )
    with pytest.raises(JointOffsetError):
        _run(settings, database, database.list_datasets()[0])
    assert not (settings.nas_root / "derived/rby1-offset").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"ranges_deg": [0.3] * 6},
        {"ranges_deg": [11.0] * 7},
        {"copies": 0},
        {"copies": 11},
        {"episode_indices": []},
        {"episode_indices": [1, 1]},
        {"output_name": "../escape"},
    ],
)
def test_request_bounds(change):
    base = {"profile_id": "p", "output_name": "out", "idempotency_key": "k"}
    with pytest.raises(ValueError):
        JointOffsetAugmentationCreate.model_validate({**base, **change})


def test_api_queues_one_cpu_job_idempotently_and_checks_episode_range(tmp_path):
    from fastapi import FastAPI

    from datasetui.api import create_router
    from datasetui.queueing import RecordingDispatcher

    settings, database, dataset = _registered(tmp_path)
    app = FastAPI()
    app.include_router(create_router(database, RecordingDispatcher()))
    client = TestClient(app)
    profile = database.create_profile("api augment")
    body = {
        "profile_id": profile["id"],
        "copies": 1,
        "seed": 1,
        "episode_indices": [2, 0],
        "output_name": "api-offset",
        "idempotency_key": "api-offset",
    }
    url = f"/api/v1/datasets/{dataset['id']}/joint-offset-augmentations"
    first = client.post(url, json=body)
    assert first.status_code == 202, first.text
    job = first.json()
    assert job["kind"] == "augment.joint_offset" and job["queue_name"] == "cpu"
    assert job["payload"]["episode_indices"] == [0, 2]
    assert job["payload"]["ranges_deg"] == list(DEFAULT_RANGES_DEG)
    assert job["payload"]["source"] == {"id": dataset["id"], "fingerprint": dataset["fingerprint"]}
    assert client.post(url, json=body).json()["id"] == job["id"]
    out_of_range = client.post(
        url, json={**body, "episode_indices": [3], "idempotency_key": "other"}
    )
    assert out_of_range.status_code == 422
