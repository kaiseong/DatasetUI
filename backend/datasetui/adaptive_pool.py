"""Resource-admitted one-job RQ workers. Never signals active job processes."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

from datasetui.config import Settings
from datasetui.resource_admission import (
    HostSampler,
    PoolConfig,
    STATUS_FILE,
    admission,
    empty_status,
)

LOG = logging.getLogger("datasetui.adaptive")


def run_one(name, queues):
    """Public RQ max_jobs=1 ensures every subsequent job gets new admission."""
    from redis import Redis
    from rq import Worker
    from rq.serializers import JSONSerializer

    cpus = [int(cpu) for cpu in os.environ["DATASETUI_ADMITTED_CPUS"].split(",")]
    os.sched_setaffinity(0, cpus)
    worker = Worker(
        queues,
        name=name,
        connection=Redis.from_url(Settings.from_env().redis_url),
        serializer=JSONSerializer,
    )
    worker.work(burst=True, max_jobs=1)


def job_counts(database_path):
    with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True, timeout=2) as db:
        rows = dict(
            db.execute(
                "SELECT status,COUNT(*) FROM jobs WHERE queue_name='cpu' AND status IN ('running','queued') GROUP BY status"
            )
        )
    return rows.get("running", 0), rows.get("queued", 0)


def write_status(root, status):
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".{STATUS_FILE}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(status, ensure_ascii=False))
    os.replace(temporary, root / STATUS_FILE)


class AdaptivePool:
    def __init__(self, settings, config, *, sampler=None, spawn=subprocess.Popen):
        self.settings, self.config = settings, config
        self.sampler = sampler or HostSampler()
        self.spawn = spawn
        self.children = {}
        self.stopping = False
        self.stable = 0
        self.last_start = float("-inf")
        self.rotation = 0
        self.lock_fd = None

    def stop_admissions(self, *_):
        self.stopping = True

    def tick(self, connection, now=None):
        from rq import Queue, Worker
        from rq.serializers import JSONSerializer

        now = time.monotonic() if now is None else now
        self.children = {
            name: child for name, child in self.children.items() if child.poll() is None
        }
        status = empty_status(self.config, enabled=True)
        status["sampled_at"] = datetime.now(timezone.utc).isoformat()
        resources = self.sampler.sample()
        active, queued = job_counts(self.settings.database_path)
        queues = [
            Queue(q, connection=connection, serializer=JSONSerializer)
            for q in self.config.queues
        ]
        pending = sum(queue.count for queue in queues)
        # Orphaned workers from an older controller count as external too.
        external = sum(
            bool(set(worker.queue_names()) & set(self.config.queues))
            for worker in Worker.all(connection=connection)
            if worker.name not in self.children
        )
        status.update(
            healthy=True,
            active_jobs=active,
            queued_jobs=queued,
            reserved_worker_slots=external,
            managed_running_jobs=len(self.children),
        )
        if resources is not None:
            status.update(
                cpu_available_percent=round(resources.cpu_available, 1),
                memory_available_percent=round(resources.memory_percent, 1),
                iowait_percent=round(resources.iowait, 1),
            )
        decision, message = admission(
            self.config,
            resources,
            active_jobs=active,
            external_workers=external,
            local_children=len(self.children),
            queued_jobs=pending,
        )
        if self.stopping:
            decision, message = (
                "unavailable",
                "새 작업 배정을 중지하고 실행 중인 작업의 완료를 기다립니다.",
            )
        self.stable = self.stable + 1 if decision == "ready" else 0
        if decision == "ready" and (
            self.stable < self.config.stable_samples
            or now - self.last_start < self.config.cooldown_seconds
        ):
            decision, message = (
                "warming_up",
                "자원 여유가 안정적으로 유지되는지 확인한 뒤 다음 작업을 시작합니다.",
            )
        elif decision == "ready":
            assert self.lock_fd is not None, "Admission requires singleton ownership"
            name = f"datasetui-adaptive-{uuid.uuid4().hex}"
            available = sorted(os.sched_getaffinity(0))
            offset = self.rotation * self.config.job_cpus % len(available)
            chosen = (available[offset:] + available[:offset])[: self.config.job_cpus]
            env = dict(
                os.environ,
                DATASETUI_ADMITTED_CPUS=",".join(map(str, chosen)),
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
            )
            ordered = (
                self.config.queues[self.rotation % len(self.config.queues) :]
                + self.config.queues[: self.rotation % len(self.config.queues)]
            )
            child = self.spawn(
                [
                    sys.executable,
                    "-m",
                    "datasetui.adaptive_pool",
                    "--run-one",
                    name,
                    *ordered,
                ],
                env=env,
                start_new_session=True,
                pass_fds=(self.lock_fd,),
            )
            self.children[name] = child
            self.rotation += 1
            self.last_start = now
            self.stable = 0
            status["managed_running_jobs"] = len(self.children)
            message = "자원 여유를 확인해 대기 작업 1개를 추가 배정했습니다."
            LOG.info(
                "admitted %s; existing active=%s external slots=%s",
                name,
                active,
                external,
            )
        status.update(decision=decision, message=message)
        write_status(self.settings.jobs_root, status)
        return status

    def run(self):
        from redis import Redis

        self.settings.jobs_root.mkdir(parents=True, exist_ok=True)
        # Shared file lock prevents a second pool from admitting at the same time.
        with (self.settings.jobs_root / "adaptive-pool.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.lock_fd = lock.fileno()
            connection = Redis.from_url(
                self.settings.redis_url, socket_timeout=3, socket_connect_timeout=3
            )
            signal.signal(signal.SIGTERM, self.stop_admissions)
            signal.signal(signal.SIGINT, self.stop_admissions)
            while not self.stopping or self.children:
                try:
                    self.tick(connection)
                except Exception:
                    LOG.exception("Admission paused; running workers left untouched")
                    self.stable = 0
                    status = empty_status(self.config, enabled=True)
                    status["sampled_at"] = datetime.now(timezone.utc).isoformat()
                    try:
                        write_status(self.settings.jobs_root, status)
                    except OSError:
                        LOG.exception("Cannot write pool status")
                time.sleep(self.config.poll_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1 and sys.argv[1] == "--run-one":
        run_one(sys.argv[2], sys.argv[3:])
    else:
        AdaptivePool(Settings.from_env(), PoolConfig.from_env()).run()
