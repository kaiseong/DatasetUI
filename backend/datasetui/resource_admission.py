"""Host resource sampling and conservative admission; never preempts work."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

GIB = 1024**3
STATUS_FILE = "adaptive-pool.json"


@dataclass(frozen=True)
class PoolConfig:
    max_jobs: int = 3
    minimum_available: float = 30.0
    job_cpus: int = 4
    job_memory_bytes: int = 8 * GIB
    max_iowait: float = 10.0
    poll_seconds: float = 5.0
    cooldown_seconds: float = 10.0
    stable_samples: int = 2
    queues: tuple[str, ...] = ("cpu", "cpu-relative-v1", "cpu-adaptive-v1")

    @classmethod
    def from_env(cls):
        config = cls(
            max_jobs=int(os.environ.get("DATASETUI_PARALLEL_MAX_JOBS", "3")),
            minimum_available=float(
                os.environ.get("DATASETUI_RESOURCE_RESERVE_PERCENT", "30")
            ),
            job_cpus=int(os.environ.get("DATASETUI_PARALLEL_JOB_CPUS", "4")),
            job_memory_bytes=int(
                os.environ.get("DATASETUI_PARALLEL_JOB_MEMORY_GIB", "8")
            )
            * GIB,
            queues=tuple(
                q.strip()
                for q in os.environ.get(
                    "DATASETUI_ADAPTIVE_QUEUES", "cpu,cpu-relative-v1,cpu-adaptive-v1"
                ).split(",")
                if q.strip()
            ),
        )
        if not 1 <= config.max_jobs <= 8 or not 30 <= config.minimum_available <= 90:
            raise ValueError(
                "Parallel jobs must be 1–8; resource reserve must be 30–90%"
            )
        if (
            not 1 <= config.job_cpus <= 32
            or not GIB <= config.job_memory_bytes <= 64 * GIB
        ):
            raise ValueError("Invalid per-job resource reservation")
        if not config.queues or any(not q.startswith("cpu") for q in config.queues):
            raise ValueError("Adaptive pool accepts CPU queues only")
        return config


@dataclass(frozen=True)
class Resources:
    cpu_available: float
    memory_available: int
    memory_total: int
    cpu_count: int
    iowait: float = 0.0
    pool_memory_available: int | None = None

    @property
    def memory_percent(self):
        return 100.0 * self.memory_available / self.memory_total


class HostSampler:
    def __init__(self, proc_root=Path("/proc"), cgroup_root=Path("/sys/fs/cgroup")):
        self.proc_root = proc_root
        self.cgroup_root = cgroup_root
        self.previous = None

    def sample(self) -> Resources | None:
        lines = (self.proc_root / "stat").read_text().splitlines()
        fields = [int(n) for n in lines[0].split()[1:9]]
        if len(fields) < 5 or not lines[0].startswith("cpu "):
            raise ValueError("Host CPU metrics unavailable")
        current = (sum(fields), fields[3], fields[4])
        previous, self.previous = self.previous, current
        if previous is None:
            return None
        delta = current[0] - previous[0]
        idle, iowait = current[1] - previous[1], current[2] - previous[2]
        if delta <= 0 or min(idle, iowait) < 0 or idle + iowait > delta:
            raise ValueError("Invalid host CPU sample")
        memory = {}
        for line in (self.proc_root / "meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable"}:
                memory[key] = int(value.split()[0]) * 1024
        total, available = memory["MemTotal"], memory["MemAvailable"]
        if not 0 < available <= total:
            raise ValueError("Invalid host memory sample")
        pool_available = None
        maximum = self.cgroup_root / "memory.max"
        if maximum.exists():
            limit = maximum.read_text().strip()
            if limit != "max":
                used = int((self.cgroup_root / "memory.current").read_text())
                pool_available = max(0, int(limit) - used)
        return Resources(
            cpu_available=100 * idle / delta,
            memory_available=available,
            memory_total=total,
            cpu_count=min(
                sum(line.startswith("cpu") and line[3:4].isdigit() for line in lines),
                len(os.sched_getaffinity(0)),
            ),
            iowait=100 * iowait / delta,
            pool_memory_available=pool_available,
        )


def admission(
    config, resources, *, active_jobs, external_workers, local_children, queued_jobs
):
    """Reserve idle legacy consumers too: they can dequeue without our lock."""
    if queued_jobs == 0:
        return "no_jobs", "실행 가능한 CPU 큐 작업이 없습니다."
    if max(active_jobs, external_workers + local_children) >= config.max_jobs:
        return (
            "limit",
            f"동시 실행 한도 {config.max_jobs}개에 도달했거나 기존 worker 자리를 예약 중입니다.",
        )
    if resources is None:
        return "warming_up", "서버 CPU·RAM 사용량을 측정하고 있습니다."
    if resources.cpu_count < 1 or resources.memory_total <= 0:
        return (
            "unavailable",
            "서버 자원 정보를 확인할 수 없어 새 작업을 시작하지 않습니다.",
        )
    # Reserve whole budgets for local jobs even before their allocation ramps up.
    budgets = local_children + 1
    projected_cpu = (
        resources.cpu_available - 100 * config.job_cpus * budgets / resources.cpu_count
    )
    projected_memory = (
        100
        * (resources.memory_available - config.job_memory_bytes * budgets)
        / resources.memory_total
    )
    pool_ok = resources.pool_memory_available is None or (
        resources.pool_memory_available
        >= config.job_memory_bytes * budgets + 256 * 1024**2
    )
    if (
        not all(
            math.isfinite(n)
            for n in (projected_cpu, projected_memory, resources.iowait)
        )
        or projected_cpu < config.minimum_available
        or projected_memory < config.minimum_available
        or resources.iowait > config.max_iowait
        or not pool_ok
    ):
        return (
            "resource_wait",
            "추가 작업의 CPU·RAM 예약량과 저장장치 부하를 고려해 자원 여유를 기다립니다. 실행 중인 작업은 유지합니다.",
        )
    return "ready", "CPU·RAM 여유가 충분해 다음 작업을 병렬 실행할 수 있습니다."


def empty_status(config=None, *, enabled=False):
    config = config or PoolConfig.from_env()
    return {
        "enabled": enabled,
        "healthy": False,
        "sampled_at": None,
        "cpu_available_percent": None,
        "memory_available_percent": None,
        "iowait_percent": None,
        "active_jobs": 0,
        "queued_jobs": 0,
        "max_parallel_jobs": config.max_jobs,
        "reserved_worker_slots": 0,
        "managed_running_jobs": 0,
        "minimum_available_percent": config.minimum_available,
        "decision": "unavailable",
        "message": "자동 병렬 실행 상태를 확인하고 있습니다.",
    }


def read_pool_status(jobs_root: Path):
    enabled = os.environ.get("DATASETUI_ADAPTIVE_ENABLED", "0") == "1"
    fallback = empty_status(enabled=enabled)
    if not enabled:
        fallback["message"] = "자동 병렬 실행이 설정되지 않았습니다."
        return fallback
    try:
        path = jobs_root / STATUS_FILE
        if path.is_symlink() or path.stat().st_size > 32768:
            return fallback
        value = json.loads(path.read_text())
        timestamp = datetime.fromisoformat(value["sampled_at"])
        age = (datetime.now(timezone.utc) - timestamp).total_seconds()
        if not 0 <= age <= 25 or set(value) != set(fallback):
            return fallback
        if value["decision"] not in {
            "ready",
            "no_jobs",
            "resource_wait",
            "limit",
            "warming_up",
            "unavailable",
        }:
            return fallback
        for key in (
            "cpu_available_percent",
            "memory_available_percent",
            "iowait_percent",
        ):
            if value[key] is not None and (
                isinstance(value[key], bool)
                or not isinstance(value[key], (int, float))
                or not math.isfinite(value[key])
                or not 0 <= value[key] <= 100
            ):
                return fallback
        for key in (
            "active_jobs",
            "max_parallel_jobs",
            "queued_jobs",
            "reserved_worker_slots",
            "managed_running_jobs",
        ):
            if type(value[key]) is not int or value[key] < 0:
                return fallback
        if (
            type(value["enabled"]) is not bool
            or type(value["healthy"]) is not bool
            or not isinstance(value["message"], str)
        ):
            return fallback
        return value
    except (OSError, ValueError, KeyError, TypeError):
        return fallback
