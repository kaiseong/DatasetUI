from __future__ import annotations

import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from datasetui.database import (
    Database,
    JobCancellationConflictError,
    JobLeaseLostError,
)
from datasetui.job_cancellation import (
    JobCancellationRequested,
    _children,
    _identity,
    cancellation_monitor,
)
from datasetui.tasks import run_job
from test_curation_api import _register


def _running_job(database: Database, *, key: str = "running-cancel"):
    profile = database.create_profile(f"Profile {key}")
    job, _ = database.create_job(
        kind="phase2.smoke",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={},
        idempotency_key=key,
    )
    claimed = database.claim_job(job["id"], worker_id="worker", lease_seconds=120)
    assert claimed is not None
    return profile, job


def test_running_cancellation_is_requested_then_acknowledged(database: Database):
    profile, job = _running_job(database)

    requested = database.request_job_cancellation(job["id"], profile_id=profile["id"])
    repeated = database.request_job_cancellation(job["id"], profile_id=profile["id"])

    assert requested["status"] == repeated["status"] == "running"
    assert requested["cancellation_requested"] is True
    assert requested["cancellation_requested_at"] is not None
    with pytest.raises(JobCancellationRequested):
        database.assert_job_lease(job["id"], worker_id="worker")
    finished = database.complete_job_cancellation(job["id"], worker_id="worker")
    assert finished["status"] == "cancelled"
    assert finished["cancellation_requested"] is True
    assert [event["event_type"] for event in database.list_job_events(job["id"])] == [
        "queued",
        "running",
        "cancellation_requested",
        "cancelled",
    ]


def test_cancellation_and_finalization_guard_have_one_atomic_winner(
    database: Database,
):
    profile, job = _running_job(database, key="guard-race")
    barrier = threading.Barrier(2)

    def cancel():
        barrier.wait()
        try:
            database.request_job_cancellation(job["id"], profile_id=profile["id"])
            return "cancel"
        except JobCancellationConflictError as exc:
            assert exc.reason == "finalizing"
            return "guard-lost-cancel"

    def guard():
        barrier.wait()
        try:
            database.begin_job_finalization(job["id"], worker_id="worker")
            return "guard"
        except JobCancellationRequested:
            return "cancel-lost-guard"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(cancel), executor.submit(guard)]
        results = {future.result() for future in futures}

    assert results in (
        {"guard", "guard-lost-cancel"},
        {"cancel", "cancel-lost-guard"},
    )
    stored = database.get_job(job["id"])
    assert (stored["cancellation_requested_at"] is not None) != (
        stored["cancellation_guarded_at"] is not None
    )


def test_cancellation_and_success_have_one_atomic_winner(database: Database):
    profile, job = _running_job(database, key="success-race")
    barrier = threading.Barrier(2)

    def cancel():
        barrier.wait()
        try:
            database.request_job_cancellation(job["id"], profile_id=profile["id"])
            return "cancel"
        except JobCancellationConflictError:
            return "success-lost-cancel"

    def succeed():
        barrier.wait()
        try:
            database.succeed_job(job["id"], {"ok": True}, worker_id="worker")
            return "success"
        except JobCancellationRequested:
            return "cancel-lost-success"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(cancel), executor.submit(succeed)]
        results = {future.result() for future in futures}

    if results == {"cancel", "cancel-lost-success"}:
        database.complete_job_cancellation(job["id"], worker_id="worker")
        assert database.get_job(job["id"])["status"] == "cancelled"
    else:
        assert results == {"success", "success-lost-cancel"}
        assert database.get_job(job["id"])["status"] == "succeeded"


def test_stale_finalization_guard_cannot_publish_or_requeue(database: Database):
    _profile, job = _running_job(database, key="stale-guard")
    database.begin_job_finalization(job["id"], worker_id="worker")
    expired = (
        (datetime.now(timezone.utc) - timedelta(seconds=1))
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
    with database.connect() as connection:
        connection.execute(
            "UPDATE jobs SET lease_expires_at = ? WHERE id = ?",
            (expired, job["id"]),
        )

    with pytest.raises(JobLeaseLostError):
        database.begin_job_finalization(job["id"], worker_id="worker")
    assert database.requeue_expired_jobs() == []
    stored = database.get_job(job["id"])
    assert stored["status"] == "interrupted"
    assert stored["error_code"] == "finalization_outcome_uncertain"


def test_expired_requested_validation_becomes_cancelled_and_deletable(
    client, database: Database
):
    dataset = _register(database)
    profile = database.create_profile("Cancelled validation")
    job, _ = database.create_job(
        kind="datasets.validate",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={"dataset_id": dataset["id"], "mode": "full"},
        idempotency_key="cancel-validation",
    )
    database.record_validation_run(
        job_id=job["id"],
        dataset_id=dataset["id"],
        dataset_fingerprint=dataset["fingerprint"],
        mode="full",
    )
    database.claim_job(job["id"], worker_id="lost", lease_seconds=120)
    database.request_job_cancellation(job["id"], profile_id=profile["id"])
    expired = (
        (datetime.now(timezone.utc) - timedelta(seconds=1))
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
    with database.connect() as connection:
        connection.execute(
            "UPDATE jobs SET lease_expires_at = ? WHERE id = ?",
            (expired, job["id"]),
        )

    assert database.requeue_expired_jobs() == []
    assert database.get_job(job["id"])["status"] == "cancelled"
    assert (
        client.delete(
            f"/api/v1/datasets/{dataset['id']}/validations/{job['id']}"
        ).status_code
        == 204
    )


def test_actual_worker_cancels_long_subprocess_and_reaps_it(
    tmp_path: Path, monkeypatch
):
    database_path = tmp_path / "datasetui.sqlite3"
    monkeypatch.setenv("DATASETUI_DB_PATH", str(database_path))
    database = Database(database_path)
    database.initialize()
    profile = database.create_profile("Long subprocess")
    job, _ = database.create_job(
        kind="phase2.smoke",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={},
        idempotency_key="long-subprocess",
    )
    child: list[subprocess.Popen] = []

    def long_job(_kind, _payload):
        process = subprocess.Popen(["sleep", "30"])
        child.append(process)
        return {"returncode": process.wait()}

    monkeypatch.setattr("datasetui.tasks.run_registered_job", long_job)

    def request_when_running():
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if database.get_job(job["id"])["status"] == "running" and child:
                database.request_job_cancellation(job["id"], profile_id=profile["id"])
                return
            time.sleep(0.01)
        raise AssertionError("worker did not start the subprocess")

    requester = threading.Thread(target=request_when_running)
    requester.start()
    result = run_job(job["id"])
    requester.join(timeout=5)

    assert result == {"job_id": job["id"], "status": "cancelled", "claimed": True}
    assert child and child[0].poll() is not None
    assert database.get_job(job["id"])["status"] == "cancelled"


def test_observed_cancellation_does_not_depend_on_second_database_read():
    class FailsAfterObservation:
        calls = 0

        def is_job_cancellation_requested(self, _job_id, *, worker_id):
            self.calls += 1
            if self.calls == 1:
                return True
            raise sqlite3.OperationalError("transient read failure")

    database = FailsAfterObservation()

    with pytest.raises(JobCancellationRequested):
        with cancellation_monitor(
            database,
            job_id="observed-cancel",
            worker_id="worker",
            poll_seconds=0.01,
            child_grace_seconds=0,
        ):
            while True:
                time.sleep(0.01)

    assert database.calls == 1


def test_cancellation_reaps_child_spawned_during_termination(
    tmp_path: Path, monkeypatch
):
    database_path = tmp_path / "datasetui.sqlite3"
    ready_path = tmp_path / "ready"
    spawned_path = tmp_path / "spawned-pid"
    spawned_ready_path = tmp_path / "spawned-ready"
    monkeypatch.setenv("DATASETUI_DB_PATH", str(database_path))
    database = Database(database_path)
    database.initialize()
    profile = database.create_profile("Replacement subprocess")
    job, _ = database.create_job(
        kind="phase2.smoke",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={},
        idempotency_key="replacement-subprocess",
    )
    parent: list[subprocess.Popen] = []
    script = """
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ready, spawned, spawned_ready = map(Path, sys.argv[1:])

def ignore_term():
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

def terminate(_signum, _frame):
    # Ignore SIGTERM from exec onwards: cleanup signals every descendant at
    # once, and a child still starting up must not die before it is ready.
    child = subprocess.Popen([
        sys.executable,
        "-c",
        "import sys,time; from pathlib import Path; "
        "Path(sys.argv[1]).write_text('ready'); time.sleep(30)",
        str(spawned_ready),
    ], preexec_fn=ignore_term)
    while not spawned_ready.exists():
        time.sleep(0.01)
    spawned.write_text(str(child.pid))
    os._exit(0)

signal.signal(signal.SIGTERM, terminate)
ready.write_text(str(os.getpid()))
while True:
    time.sleep(0.1)
"""

    def replacement_job(_kind, _payload):
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                str(ready_path),
                str(spawned_path),
                str(spawned_ready_path),
            ]
        )
        parent.append(process)
        return {"returncode": process.wait()}

    monkeypatch.setattr("datasetui.tasks.run_registered_job", replacement_job)

    def request_when_ready():
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if ready_path.exists():
                database.request_job_cancellation(job["id"], profile_id=profile["id"])
                return
            time.sleep(0.01)
        raise AssertionError("replacement subprocess did not become ready")

    requester = threading.Thread(target=request_when_ready)
    requester.start()
    result = run_job(job["id"])
    requester.join(timeout=5)

    assert result["status"] == "cancelled"
    assert parent and parent[0].poll() is not None
    assert spawned_path.exists()
    spawned_pid = int(spawned_path.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and Path(f"/proc/{spawned_pid}").exists():
        time.sleep(0.05)
    assert not Path(f"/proc/{spawned_pid}").exists()


def test_failure_cancellation_completion_lease_race_returns_current_status(
    tmp_path: Path, monkeypatch
):
    database_path = tmp_path / "datasetui.sqlite3"
    monkeypatch.setenv("DATASETUI_DB_PATH", str(database_path))
    database = Database(database_path)
    database.initialize()
    profile = database.create_profile("Cancellation lease race")
    job, _ = database.create_job(
        kind="phase2.smoke",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={},
        idempotency_key="cancellation-lease-race",
    )

    def failing_job(_kind, _payload):
        raise RuntimeError("operation failed while cancellation arrived")

    def cancellation_wins(self, job_id, *_args, **_kwargs):
        with self.connect() as connection:
            connection.execute(
                "UPDATE jobs SET cancellation_requested_at = ? WHERE id = ?",
                ("2026-09-21T00:00:00.000Z", job_id),
            )
        raise JobCancellationRequested(job_id)

    original_complete = Database.complete_job_cancellation

    def completed_by_racer(self, job_id, *, worker_id):
        original_complete(self, job_id, worker_id=worker_id)
        raise JobLeaseLostError(job_id)

    monkeypatch.setattr("datasetui.tasks.run_registered_job", failing_job)
    monkeypatch.setattr(Database, "fail_job", cancellation_wins)
    monkeypatch.setattr(Database, "complete_job_cancellation", completed_by_racer)

    result = run_job(job["id"])

    assert result == {"job_id": job["id"], "status": "cancelled", "claimed": True}
    assert database.get_job(job["id"])["status"] == "cancelled"


def test_schema_13_adds_running_cancellation_columns(tmp_path: Path, monkeypatch):
    import datasetui.database as database_module

    path = tmp_path / "schema-13.sqlite3"
    migrations = database_module.MIGRATIONS
    monkeypatch.setattr(database_module, "MIGRATIONS", migrations[:13])
    Database(path).initialize()
    monkeypatch.setattr(database_module, "MIGRATIONS", migrations)
    Database(path).initialize()

    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
    assert {"cancellation_requested_at", "cancellation_guarded_at"} <= columns


def test_proc_identity_handles_spaces_and_collects_every_thread_child(tmp_path: Path):
    proc = tmp_path / "proc"
    process = proc / "123"
    process.mkdir(parents=True)
    tail = ["S", "1", *(["0"] * 17), "98765", "0"]
    (process / "stat").write_text(f"123 (ffmpeg worker name) {' '.join(tail)}")
    for task_id, child_id in (("123", "201"), ("124", "202")):
        task = process / "task" / task_id
        task.mkdir(parents=True)
        (task / "children").write_text(child_id)

    identity = _identity(123, proc)
    assert identity is not None and identity.start_time == "98765"
    assert set(_children(123, proc)) == {201, 202}
