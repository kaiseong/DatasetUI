from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_transforms import _settings, _write_v21

from datasetui.database import Database
from datasetui.dataset_io.stats import write_stats
from datasetui.dataset_io.video import slice_video
from datasetui.datasets import inspect_dataset
from datasetui.job_progress import JobProgressReporter
from datasetui.merge.job import merge_datasets



class _ProgressStore:
    def __init__(self, error: Exception | None = None) -> None:
        self.events: list[dict] = []
        self.error = error

    def update_job_progress(self, job_id, *, worker_id, progress) -> None:
        if self.error is not None:
            raise self.error
        self.events.append(progress)


def test_merge_reporter_throttles_hot_updates_but_keeps_stage_and_end_events() -> None:
    store = _ProgressStore()
    now = [100.0]
    reporter = JobProgressReporter(
        store,
        job_id="job",
        worker_id="worker",
        output_name="merged",
        clock=lambda: now[0],
    )

    reporter({"stage": "read", "completed": 0, "total": 10, "unit": "episodes"})
    now[0] += 0.2
    reporter({"stage": "read", "completed": 1, "total": 10, "unit": "episodes"})
    now[0] += 0.9
    reporter({"stage": "read", "completed": 4, "total": 10, "unit": "episodes"})
    reporter({"stage": "write", "completed": 0, "total": 10, "unit": "episodes"})
    reporter({"stage": "write", "completed": 10, "total": 10, "unit": "episodes"})

    assert [(item["stage"], item["completed"]) for item in store.events] == [
        ("read", 0),
        ("read", 4),
        ("write", 0),
        ("write", 10),
    ]
    assert all(item["output_name"] == "merged" for item in store.events)
    assert store.events[1]["elapsed_seconds"] == 1.1


def test_merge_reporter_propagates_lease_storage_failure() -> None:
    reporter = JobProgressReporter(
        _ProgressStore(RuntimeError("lease lost")),
        job_id="job",
        worker_id="worker",
        output_name="merged",
    )

    with pytest.raises(RuntimeError, match="lease lost"):
        reporter({"stage": "read", "completed": 0, "total": 1, "unit": "episodes"})


def test_statistics_callback_reports_actual_feature_counts(tmp_path: Path) -> None:
    events: list[dict] = []
    frames = pd.DataFrame(
        {
            "action": [np.asarray([1.0]), np.asarray([2.0])],
            "timestamp": [0.0, 0.1],
        }
    )
    meta = tmp_path / "meta"
    meta.mkdir()

    write_stats(meta / "stats.json", [frames], on_progress=events.append)

    feature_events = [item for item in events if item["stage"] == "statistics"]
    assert feature_events[0]["completed"] == 0
    assert feature_events[0]["total"] == 2
    assert any(item["completed"] == 1 for item in feature_events)
    assert feature_events[-1]["completed"] == feature_events[-1]["total"] == 2


def test_video_callback_reports_frames_and_propagates_failure(tmp_path: Path) -> None:
    av = pytest.importorskip("av")
    source = tmp_path / "source.mp4"
    container = av.open(str(source), mode="w")
    stream = container.add_stream("libx264", rate=10)
    stream.width = 32
    stream.height = 24
    stream.pix_fmt = "yuv420p"
    for value in range(6):
        frame = av.VideoFrame.from_ndarray(
            np.full((24, 32, 3), value, dtype=np.uint8), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    events: list[dict] = []
    slice_video(
        source,
        tmp_path / "output.mp4",
        1,
        5,
        10.0,
        4,
        on_progress=events.append,
    )
    assert events[-1] == {
        "stage": "video",
        "completed": 8,
        "total": 8,
        "unit": "frame_operations",
        "current_item": "인코딩",
    }
    assert any(0 < item["completed"] < item["total"] for item in events)

    def lose_lease(progress: dict) -> None:
        raise RuntimeError("lease lost in video")

    with pytest.raises(RuntimeError, match="lease lost in video"):
        slice_video(
            source,
            tmp_path / "never-finished.mp4",
            1,
            5,
            10.0,
            4,
            on_progress=lose_lease,
        )


def test_small_merge_persists_measured_stage_sequence(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path)
    _write_v21(settings.nas_root / "raw/lab/first")
    _write_v21(settings.nas_root / "raw/lab/second")
    database = Database(settings.database_path)
    database.initialize()
    candidates = [
        inspect_dataset(
            area_root=settings.nas_root / "raw",
            storage_area="raw",
            relative_path=relative,
        ).as_record()
        for relative in ("lab/first", "lab/second")
    ]
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw", records=candidates, scan_generation=generation
    )
    datasets = database.list_datasets()
    profile = database.create_profile("Progress merge")
    payload = {
        "sources": [
            {"id": item["id"], "fingerprint": item["fingerprint"]} for item in datasets
        ],
        "output_name": "merged-progress",
        "robot_type": "rby1",
    }
    job, _ = database.create_job(
        kind="datasets.merge",
        queue_name="cpu",
        profile_id=profile["id"],
        payload=payload,
        idempotency_key="merge-progress",
    )
    database.claim_job(job["id"], worker_id="merge-worker", lease_seconds=120)
    persisted: list[dict] = []
    original = database.update_job_progress

    def capture(job_id, *, worker_id, progress):
        persisted.append(progress)
        original(job_id, worker_id=worker_id, progress=progress)

    monkeypatch.setattr(database, "update_job_progress", capture)

    result = merge_datasets(
        database=database,
        settings=settings,
        payload=payload,
        job_id=job["id"],
        worker_id="merge-worker",
    )

    assert result["output"]["episodes"] == 4
    stages = [item["stage"] for item in persisted]
    for stage in (
        "preparing",
        "read",
        "write",
        "statistics",
        "validate",
        "publish",
        "register",
        "complete",
    ):
        assert stage in stages
    assert persisted[-1]["stage"] == "complete"
    assert persisted[-1]["completed"] == persisted[-1]["total"] == 1
    assert persisted[-1]["output_name"] == "merged-progress"
    assert (
        json.loads(
            (settings.nas_root / "derived/merged-progress/meta/info.json").read_text()
        )["total_episodes"]
        == 4
    )
