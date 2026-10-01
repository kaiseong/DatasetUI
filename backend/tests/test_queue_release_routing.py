from types import SimpleNamespace
from unittest.mock import ANY

import pytest
import rq
from rq.job import Job, JobStatus

from datasetui.queueing import RQDispatcher


@pytest.fixture
def queue_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(RQDispatcher, "_connection", lambda self: object())
    monkeypatch.setattr(Job, "exists", lambda *args, **kwargs: False)

    class Queue:
        def __init__(self, name, **kwargs):
            self.name = name
            calls.append((name, kwargs))

        def enqueue(self, *args, **kwargs):
            calls.append((args, kwargs))
            return SimpleNamespace(id=kwargs["job_id"])

        def remove(self, rq_job_id, pipeline=None):
            calls.append(("remove", self.name, rq_job_id))

    monkeypatch.setattr(rq, "Queue", Queue)
    return calls


@pytest.mark.parametrize("logical", ["cpu", "io", "converter-v21"])
def test_new_jobs_use_release_queue_with_original_timeout(
    logical, queue_calls, monkeypatch
):
    monkeypatch.setenv("DATASETUI_QUEUE_SUFFIX", "relative-v1")
    dispatcher = RQDispatcher("redis://unused", 900, 86400)
    dispatcher.enqueue(job_id="new", queue_name=logical, rq_job_id="rq-new")
    assert queue_calls[0][0] == f"{logical}-relative-v1"
    assert queue_calls[0][1]["default_timeout"] == (86400 if logical == "io" else 900)


def test_existing_running_or_queued_job_is_not_moved(queue_calls, monkeypatch):
    monkeypatch.setattr(Job, "exists", lambda *args, **kwargs: True)
    for status in [JobStatus.STARTED, JobStatus.QUEUED]:
        existing = SimpleNamespace(get_status=lambda **kwargs: status)
        monkeypatch.setattr(Job, "fetch", lambda *args, **kwargs: existing)
        RQDispatcher("redis://unused", 900, 86400, "relative-v1").enqueue(
            job_id="legacy", queue_name="cpu", rq_job_id="rq-legacy"
        )
    assert queue_calls == []


def test_default_queue_and_explicit_timeout_are_preserved(queue_calls, monkeypatch):
    monkeypatch.delenv("DATASETUI_QUEUE_SUFFIX", raising=False)
    RQDispatcher("redis://unused", 900, 86400).enqueue(
        job_id="new", queue_name="cpu", rq_job_id="rq-new", job_timeout=86400
    )
    assert queue_calls[0][0] == "cpu"
    assert queue_calls[1][1]["job_timeout"] == 86400


def test_invalid_release_suffix_fails_before_connecting():
    with pytest.raises(ValueError, match="DATASETUI_QUEUE_SUFFIX"):
        RQDispatcher("redis://unused", 900, 86400, "bad:suffix")


def test_release_does_not_move_gpu_jobs_without_a_gpu_release_worker(queue_calls):
    RQDispatcher("redis://unused", 900, 86400, "relative-v1").enqueue(
        job_id="gpu", queue_name="gpu", rq_job_id="rq-gpu"
    )
    assert queue_calls[0][0] == "gpu"


def test_adaptive_cpu_override_keeps_other_release_queues(queue_calls, monkeypatch):
    monkeypatch.setenv("DATASETUI_CPU_QUEUE", "cpu-adaptive-v1")
    dispatcher = RQDispatcher("redis://unused", 900, 86400, "relative-v1")
    dispatcher.enqueue(job_id="cpu", queue_name="cpu", rq_job_id="rq-cpu")
    assert queue_calls[0][0] == "cpu-adaptive-v1"
    dispatcher.enqueue(job_id="io", queue_name="io", rq_job_id="rq-io")
    assert queue_calls[2][0] == "io-relative-v1"


def test_pending_removal_checks_legacy_and_release_queues_without_cancelling(
    queue_calls, monkeypatch
):
    monkeypatch.setenv("DATASETUI_QUEUE_SUFFIX", "relative-v1")

    RQDispatcher("redis://unused", 900, 86400).remove_pending(
        rq_job_id="rq-pending", queue_name="cpu"
    )

    assert queue_calls == [
        ("cpu", {"connection": ANY, "serializer": ANY}),
        ("remove", "cpu", "rq-pending"),
        (
            "cpu-relative-v1",
            {"connection": ANY, "serializer": ANY},
        ),
        ("remove", "cpu-relative-v1", "rq-pending"),
    ]


def test_pending_removal_uses_real_rq_queue_api(monkeypatch):
    """Exercise the installed Queue.remove signature, not a permissive mock."""
    calls = []
    connection = SimpleNamespace(
        lrem=lambda key, count, job_id: calls.append((key, count, job_id))
    )
    monkeypatch.setattr(RQDispatcher, "_connection", lambda self: connection)
    dispatcher = RQDispatcher(
        "redis://unused", 900, 86400, "relative-v1", "cpu-adaptive-v1"
    )
    dispatcher.remove_pending(rq_job_id="rq-pending", queue_name="cpu")
    assert calls == [
        ("rq:queue:cpu", 1, "rq-pending"),
        ("rq:queue:cpu-adaptive-v1", 1, "rq-pending"),
        ("rq:queue:cpu-relative-v1", 1, "rq-pending"),
    ]
