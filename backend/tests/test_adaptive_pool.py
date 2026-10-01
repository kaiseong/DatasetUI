from dataclasses import replace
from types import SimpleNamespace

import rq

from datasetui import adaptive_pool
from datasetui.adaptive_pool import AdaptivePool
from datasetui.config import Settings
from datasetui.resource_admission import GIB, PoolConfig, Resources


def make_pool(tmp_path, monkeypatch, *, external=1):
    resource = Resources(95, 100 * GIB, 128 * GIB, 32)
    sampler = SimpleNamespace(sample=lambda: resource)
    workers = [
        SimpleNamespace(name=f"legacy-{i}", queue_names=lambda: ["cpu"])
        for i in range(external)
    ]
    monkeypatch.setattr(rq.Worker, "all", lambda **kwargs: workers)
    monkeypatch.setattr(rq, "Queue", lambda *args, **kwargs: SimpleNamespace(count=1))
    monkeypatch.setattr(adaptive_pool, "job_counts", lambda path: (1, 3))
    calls = []

    def spawn(*args, **kwargs):
        child = SimpleNamespace(poll=lambda: None)
        calls.append((args, kwargs, child))
        return child

    settings = replace(
        Settings.from_env(), jobs_root=tmp_path, database_path=tmp_path / "db"
    )
    pool = AdaptivePool(settings, PoolConfig(), sampler=sampler, spawn=spawn)
    pool.lock_fd = 123
    return pool, calls, workers


def test_stability_cooldown_and_legacy_slot_limit(tmp_path, monkeypatch):
    pool, calls, _ = make_pool(tmp_path, monkeypatch)
    assert pool.tick(None, now=0)["decision"] == "warming_up"
    pool.tick(None, now=5)
    assert len(calls) == 1
    assert calls[0][0][0][1:4] == ["-m", "datasetui.adaptive_pool", "--run-one"]
    assert calls[0][1]["pass_fds"] == (123,)
    pool.tick(None, now=10)
    assert len(calls) == 1
    pool.tick(None, now=15)
    assert len(calls) == 2
    assert pool.tick(None, now=25)["decision"] == "limit"
    assert len(calls) == 2


def test_pressure_and_shutdown_do_not_signal_or_replace_running_children(
    tmp_path, monkeypatch
):
    pool, calls, _ = make_pool(tmp_path, monkeypatch)
    pool.tick(None, now=0)
    pool.tick(None, now=5)
    original = dict(pool.children)
    pool.sampler.sample = lambda: Resources(10, 20 * GIB, 128 * GIB, 32)
    assert pool.tick(None, now=15)["decision"] == "resource_wait"
    assert pool.children == original
    pool.stop_admissions()
    assert pool.tick(None, now=25)["decision"] == "unavailable"
    assert pool.children == original and len(calls) == 1


def test_orphaned_adaptive_worker_still_reserves_a_slot(tmp_path, monkeypatch):
    pool, calls, workers = make_pool(tmp_path, monkeypatch, external=3)
    workers[0].name = "datasetui-adaptive-old-controller"
    assert pool.tick(None, now=0)["decision"] == "limit"
    assert calls == []


def test_finished_children_release_slots(tmp_path, monkeypatch):
    pool, calls, _ = make_pool(tmp_path, monkeypatch, external=2)
    pool.tick(None, now=0)
    pool.tick(None, now=5)
    assert pool.tick(None, now=10)["decision"] == "limit"
    calls[0][2].poll = lambda: 0
    pool.tick(None, now=15)
    pool.tick(None, now=20)
    assert len(calls) == 2


def test_run_one_uses_public_json_rq_one_job_contract(monkeypatch):
    calls = {}

    class Worker:
        def __init__(self, queues, **kwargs):
            calls.update(queues=queues, **kwargs)

        def work(self, **kwargs):
            calls["work"] = kwargs

    monkeypatch.setattr(rq, "Worker", Worker)
    monkeypatch.setenv("DATASETUI_ADMITTED_CPUS", "0,1")
    monkeypatch.setattr(
        adaptive_pool.os, "sched_setaffinity", lambda pid, cpus: calls.update(cpus=cpus)
    )
    adaptive_pool.run_one("test-one", ["cpu-adaptive-v1"])
    assert calls["work"] == {"burst": True, "max_jobs": 1}
    assert calls["serializer"].__name__ == "JSONSerializer"
    assert calls["cpus"] == [0, 1]
