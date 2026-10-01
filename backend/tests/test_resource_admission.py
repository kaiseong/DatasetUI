from datetime import datetime, timedelta, timezone
import json

import pytest

from datasetui.resource_admission import (
    GIB,
    HostSampler,
    PoolConfig,
    Resources,
    STATUS_FILE,
    admission,
    empty_status,
    read_pool_status,
)


def decide(resources=None, **kwargs):
    return admission(
        PoolConfig(),
        resources or Resources(90, 90 * GIB, 100 * GIB, 32),
        **dict(
            active_jobs=0, external_workers=0, local_children=0, queued_jobs=1, **kwargs
        ),
    )


@pytest.mark.parametrize(
    "cpu,memory,expected",
    [
        (42.5, 38, "ready"),
        (42.49, 38, "resource_wait"),
        (90, 37, "resource_wait"),
        (29, 90, "resource_wait"),
        (90, 29, "resource_wait"),
    ],
)
def test_reserves_thirty_percent_after_new_job_budget(cpu, memory, expected):
    assert decide(Resources(cpu, memory * GIB, 100 * GIB, 32))[0] == expected


def test_iowait_and_container_memory_can_block_admission():
    assert (
        decide(Resources(90, 90 * GIB, 100 * GIB, 32, iowait=11))[0] == "resource_wait"
    )
    assert (
        decide(Resources(90, 90 * GIB, 100 * GIB, 32, pool_memory_available=7 * GIB))[0]
        == "resource_wait"
    )


def test_running_jobs_and_idle_legacy_consumers_reserve_slots():
    config = PoolConfig()
    resource = Resources(90, 90 * GIB, 100 * GIB, 32)
    for active, external, local in [(3, 0, 0), (1, 2, 1), (0, 3, 0)]:
        assert (
            admission(
                config,
                resource,
                active_jobs=active,
                external_workers=external,
                local_children=local,
                queued_jobs=1,
            )[0]
            == "limit"
        )


def test_empty_queue_does_not_launch_and_missing_sample_is_not_zero():
    assert (
        admission(
            PoolConfig(),
            None,
            active_jobs=0,
            external_workers=0,
            local_children=0,
            queued_jobs=0,
        )[0]
        == "no_jobs"
    )
    assert (
        admission(
            PoolConfig(),
            None,
            active_jobs=0,
            external_workers=0,
            local_children=0,
            queued_jobs=1,
        )[0]
        == "warming_up"
    )


def test_local_starting_jobs_reserve_resource_budgets():
    resource = Resources(54, 90 * GIB, 100 * GIB, 32)
    assert (
        admission(
            PoolConfig(),
            resource,
            active_jobs=0,
            external_workers=0,
            local_children=1,
            queued_jobs=1,
        )[0]
        == "resource_wait"
    )


def test_sampler_uses_cpu_deltas_available_memory_and_cgroup_limit(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (proc / "stat").write_text("cpu  10 0 10 80 0 0 0 0 5 0\ncpu0 10\ncpu1 10\n")
    (proc / "meminfo").write_text(
        "MemTotal: 100000 kB\nMemFree: 100 kB\nMemAvailable: 80000 kB\n"
    )
    (cgroup / "memory.max").write_text(str(10 * GIB))
    (cgroup / "memory.current").write_text(str(GIB))
    sampler = HostSampler(proc, cgroup)
    assert sampler.sample() is None
    (proc / "stat").write_text("cpu  15 0 15 160 10 0 0 0 8 0\ncpu0 20\ncpu1 20\n")
    resource = sampler.sample()
    assert resource.cpu_available == 80
    assert resource.iowait == 10
    assert resource.memory_percent == 80
    assert resource.cpu_count == 2
    assert resource.pool_memory_available == 9 * GIB


def test_status_missing_disabled_and_stale_are_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("DATASETUI_ADAPTIVE_ENABLED", "1")
    assert read_pool_status(tmp_path)["cpu_available_percent"] is None
    value = empty_status(enabled=True)
    value.update(
        healthy=True,
        sampled_at=datetime.now(timezone.utc).isoformat(),
        decision="ready",
    )
    (tmp_path / STATUS_FILE).write_text(json.dumps(value))
    assert read_pool_status(tmp_path)["healthy"] is True
    value["sampled_at"] = (
        datetime.now(timezone.utc) - timedelta(seconds=30)
    ).isoformat()
    (tmp_path / STATUS_FILE).write_text(json.dumps(value))
    assert read_pool_status(tmp_path)["healthy"] is False
    monkeypatch.setenv("DATASETUI_ADAPTIVE_ENABLED", "0")
    assert read_pool_status(tmp_path)["enabled"] is False


@pytest.mark.parametrize("value", ["0", "9"])
def test_config_rejects_invalid_parallel_limit(value, monkeypatch):
    monkeypatch.setenv("DATASETUI_PARALLEL_MAX_JOBS", value)
    with pytest.raises(ValueError):
        PoolConfig.from_env()


def test_config_disallows_gpu_and_reserve_less_than_thirty(monkeypatch):
    monkeypatch.setenv("DATASETUI_ADAPTIVE_QUEUES", "gpu")
    with pytest.raises(ValueError):
        PoolConfig.from_env()
    monkeypatch.delenv("DATASETUI_ADAPTIVE_QUEUES")
    monkeypatch.setenv("DATASETUI_RESOURCE_RESERVE_PERCENT", "29")
    with pytest.raises(ValueError):
        PoolConfig.from_env()
