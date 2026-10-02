"""Isolated AV1 merge test; no production registry or dataset writes."""

import hashlib
import json
import tempfile
from pathlib import Path

import av
import numpy as np
import pandas as pd

import datasetui.curation.writer as curation_writer

from datasetui.config import Settings
from datasetui.database import Database
from datasetui.dataset_io.source import DatasetSource
from datasetui.datasets import inspect_dataset
from datasetui.merge import merge_datasets
from datasetui.validation import validate_dataset_root



def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode_fixture(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libaom-av1", rate=10)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.options = {"cpu-used": "8", "crf": "35"}
        for value in values:
            frame = av.VideoFrame.from_ndarray(
                np.full((32, 32, 3), value, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def fixture(root, version, number):
    (root / "meta").mkdir(parents=True)
    keys = ("observation.images.front", "observation.images.left")
    features = {
        key: {"dtype": "int64", "shape": [1]}
        for key in ("frame_index", "episode_index", "index", "task_index")
    }
    features.update(
        {
            "timestamp": {"dtype": "float32", "shape": [1]},
            "action": {"dtype": "float32", "shape": [1]},
            "observation.state": {"dtype": "float32", "shape": [1]},
            **{
                key: {
                    "dtype": "video",
                    "shape": [32, 32, 3],
                    "info": {
                        "video.codec": "av1",
                        "video.fps": 10,
                        "video.pix_fmt": "yuv420p",
                    },
                }
                for key in keys
            },
        }
    )
    v3 = version == "v3.0"
    info = {
        "codebase_version": version,
        "robot_type": "test",
        "fps": 10,
        "total_episodes": 2,
        "total_frames": 8,
        "total_tasks": 1,
        "chunks_size": 1000,
        "splits": {"train": "0:2"},
        "features": features,
        "data_path": (
            "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
            if v3
            else "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        ),
        "video_path": (
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
            if v3
            else "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    tasks = [{"task_index": 0, "task": f"task-{number}"}]
    frames, rows = [], []
    for episode in range(2):
        frame = pd.DataFrame(
            {
                "episode_index": np.full(4, episode, dtype=np.int64),
                "frame_index": np.arange(4, dtype=np.int64),
                "index": np.arange(episode * 4, episode * 4 + 4, dtype=np.int64),
                "task_index": np.zeros(4, dtype=np.int64),
                "timestamp": np.arange(4, dtype=np.float32) / 10,
                "action": [
                    np.array([number * 10 + episode + i], dtype=np.float32)
                    for i in range(4)
                ],
                "observation.state": [
                    np.array([number + i], dtype=np.float32) for i in range(4)
                ],
            }
        )
        frames.append(frame)
        row = {"episode_index": episode, "length": 4, "tasks": [f"task-{number}"]}
        if v3:
            row.update(
                {
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "dataset_from_index": episode * 4,
                    "dataset_to_index": episode * 4 + 4,
                }
            )
            for key in keys:
                row.update(
                    {
                        f"videos/{key}/chunk_index": 0,
                        f"videos/{key}/file_index": 0,
                        f"videos/{key}/from_timestamp": episode * 0.4,
                        f"videos/{key}/to_timestamp": (episode + 1) * 0.4,
                    }
                )
        else:
            data_path = root / f"data/chunk-000/episode_{episode:06d}.parquet"
            data_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(data_path, index=False)
            for camera, key in enumerate(keys):
                encode_fixture(
                    root / f"videos/chunk-000/{key}/episode_{episode:06d}.mp4",
                    [number * 60 + camera * 10 + episode * 20 + i for i in range(4)],
                )
        rows.append(row)
    if v3:
        data_path = root / "data/chunk-000/file-000.parquet"
        meta_path = root / "meta/episodes/chunk-000/file-000.parquet"
        data_path.parent.mkdir(parents=True)
        meta_path.parent.mkdir(parents=True)
        pd.concat(frames).to_parquet(data_path, index=False)
        pd.DataFrame(rows).to_parquet(meta_path, index=False)
        pd.DataFrame(tasks).to_parquet(root / "meta/tasks.parquet", index=False)
        for camera, key in enumerate(keys):
            encode_fixture(
                root / f"videos/{key}/chunk-000/file-000.mp4",
                [number * 60 + camera * 10 + i * 5 for i in range(8)],
            )
    else:
        for name, records in (("tasks", tasks), ("episodes", rows)):
            (root / f"meta/{name}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in records)
            )


for version in ("v2.1", "v3.0"):
    with tempfile.TemporaryDirectory(prefix="datasetui-av1-preserve-") as temporary:
        root = Path(temporary)
        nas = root / "nas"
        for area in ("raw", "derived", "exports", "manifests"):
            (nas / area).mkdir(parents=True)
        settings = Settings(
            database_path=root / "test.sqlite3",
            redis_url="redis://unused",
            allowed_origins=(),
            nas_root=nas,
            cache_root=root / "cache",
            staging_root=root / "staging",
            jobs_root=root / "jobs",
        )
        for number, name in enumerate(("first", "second")):
            fixture(nas / "raw" / name, version, number)
        before = {
            str(path): digest(path)
            for path in (nas / "raw").rglob("*")
            if path.is_file()
        }
        database = Database(settings.database_path)
        database.initialize()
        records = [
            inspect_dataset(
                area_root=nas / "raw", storage_area="raw", relative_path=name
            ).as_record()
            for name in ("first", "second")
        ]
        generation = database.begin_dataset_scan("raw")
        database.synchronize_datasets(
            storage_area="raw", records=records, scan_generation=generation
        )
        profile = database.create_profile("Isolated AV1 preservation smoke")
        sources = database.list_datasets()
        payload = {
            "sources": [
                {"id": row["id"], "fingerprint": row["fingerprint"]} for row in sources
            ],
            "output_name": "merged",
            "robot_type": "test",
        }
        job, _ = database.create_job(
            kind="datasets.merge",
            queue_name="cpu",
            profile_id=profile["id"],
            payload=payload,
            idempotency_key="isolated",
        )
        database.claim_job(job["id"], worker_id="smoke", lease_seconds=600)
        snapshots = []
        persist_progress = database.update_job_progress

        def capture_progress(job_id, *, worker_id, progress):
            persist_progress(job_id, worker_id=worker_id, progress=progress)
            snapshots.append(progress)

        database.update_job_progress = capture_progress

        def forbidden(*args, **kwargs):
            raise AssertionError("Merge must never enter the video transcoder")

        curation_writer.slice_video = forbidden
        result = merge_datasets(
            database=database,
            settings=settings,
            payload=payload,
            job_id=job["id"],
            worker_id="smoke",
        )
        output = nas / "derived/merged"
        merged = DatasetSource(
            output, json.loads((output / "meta/info.json").read_text())
        )
        assert result["output"]["episodes"] == 4
        assert result["video_policy"] == "preserve_source_files"
        assert merged.info["total_frames"] == 16
        for item in result["output"]["lineage"]:
            record = next(
                row for row in sources if row["id"] == item["source_dataset_id"]
            )
            source_root = nas / "raw" / record["relative_path"]
            source = DatasetSource(
                source_root, json.loads((source_root / "meta/info.json").read_text())
            )
            source_index, output_index = (
                item["source_episode_index"],
                item["output_episode_index"],
            )
            expected_data, expected_meta = source.episode(source_index)
            actual_data, actual_meta = merged.episode(output_index)
            assert np.array_equal(
                np.stack(expected_data.action), np.stack(actual_data.action)
            )
            assert np.array_equal(
                np.stack(expected_data["observation.state"]),
                np.stack(actual_data["observation.state"]),
            )
            assert np.array_equal(expected_data.timestamp, actual_data.timestamp)
            assert np.array_equal(expected_data.frame_index, actual_data.frame_index)
            assert (
                merged.tasks[int(actual_data.task_index.iloc[0])]
                == source.tasks[int(expected_data.task_index.iloc[0])]
            )
            for key in source.video_keys:
                assert merged.info["features"][key] == source.info["features"][key]
                src, start = source.video_source(source_index, key, expected_meta)
                dst, actual_start = merged.video_source(output_index, key, actual_meta)
                assert digest(src) == digest(dst), (src, dst)
                assert start == actual_start
                if version == "v3.0":
                    for suffix in ("from_timestamp", "to_timestamp"):
                        assert (
                            expected_meta[f"videos/{key}/{suffix}"]
                            == actual_meta[f"videos/{key}/{suffix}"]
                        )
                with av.open(str(dst)) as video:
                    assert video.streams.video[0].codec_context.name in (
                        "av1",
                        "libdav1d",
                    )
        assert len(list(output.rglob("*.mp4"))) == (4 if version == "v3.0" else 8)
        video_progress = [row for row in snapshots if row["stage"] == "video"]
        expected_bytes = sum(
            path.stat().st_size for path in (nas / "raw").rglob("*.mp4")
        )
        assert (
            video_progress[-1]["completed"]
            == video_progress[-1]["total"]
            == expected_bytes
        )
        assert all(row["unit"] == "bytes" for row in video_progress)
        report = validate_dataset_root(output, mode="full")
        assert report["passed"], report["issues"]
        assert before == {
            str(path): digest(path)
            for path in (nas / "raw").rglob("*")
            if path.is_file()
        }
        print(
            json.dumps(
                {
                    "version": version,
                    "episodes": 4,
                    "frames": 16,
                    "full_validation": "passed",
                    "video_bytes_identical": True,
                    "source_tree_unchanged": True,
                    "production_data_modified": False,
                }
            )
        )
