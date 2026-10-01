from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
from typing import Protocol


class QueueDispatcher(Protocol):
    def enqueue(
        self,
        *,
        job_id: str,
        queue_name: str,
        rq_job_id: str,
        job_timeout: int | None = None,
    ) -> None: ...

    def remove_pending(self, *, rq_job_id: str, queue_name: str) -> None: ...

    def ping(self) -> bool: ...


@dataclass(slots=True)
class RQDispatcher:
    redis_url: str
    timeout_seconds: int
    io_timeout_seconds: int
    queue_suffix: str = field(
        default_factory=lambda: os.environ.get("DATASETUI_QUEUE_SUFFIX", "").strip()
    )
    cpu_queue: str = field(
        default_factory=lambda: os.environ.get("DATASETUI_CPU_QUEUE", "").strip()
    )

    def __post_init__(self) -> None:
        if self.queue_suffix and not re.fullmatch(
            r"[A-Za-z0-9_-]{1,64}", self.queue_suffix
        ):
            raise ValueError(
                "DATASETUI_QUEUE_SUFFIX must be 1–64 letters, digits, _ or -"
            )
        if self.cpu_queue and not re.fullmatch(
            r"cpu[A-Za-z0-9_-]{0,61}", self.cpu_queue
        ):
            raise ValueError("DATASETUI_CPU_QUEUE must be a CPU queue name")

    def _connection(self):
        from redis import Redis

        return Redis.from_url(self.redis_url)

    def enqueue(
        self,
        *,
        job_id: str,
        queue_name: str,
        rq_job_id: str,
        job_timeout: int | None = None,
    ) -> None:
        from rq import Queue
        from rq.job import Job, JobStatus
        from rq.serializers import JSONSerializer

        connection = self._connection()
        if Job.exists(rq_job_id, connection=connection):
            existing = Job.fetch(
                rq_job_id,
                connection=connection,
                serializer=JSONSerializer,
            )
            existing_status = existing.get_status(refresh=True)
            if existing_status in {
                JobStatus.FAILED,
                JobStatus.STOPPED,
                JobStatus.CANCELED,
                JobStatus.CREATED,
            }:
                raise RuntimeError(
                    f"Existing RQ job cannot resume from {existing_status.value}"
                )
            return

        queue = Queue(
            self.cpu_queue
            if queue_name == "cpu" and self.cpu_queue
            else f"{queue_name}-{self.queue_suffix}"
            if self.queue_suffix and queue_name in {"cpu", "io", "converter-v21"}
            else queue_name,
            connection=connection,
            serializer=JSONSerializer,
            default_timeout=(
                self.io_timeout_seconds
                if queue_name in {"io", "gpu"}
                else self.timeout_seconds
            ),
        )
        rq_job = queue.enqueue(
            "datasetui.tasks.run_job",
            job_id,
            job_id=rq_job_id,
            result_ttl=86400,
            failure_ttl=604800,
            **({"job_timeout": job_timeout} if job_timeout is not None else {}),
        )
        if rq_job.id != rq_job_id:
            raise RuntimeError("RQ returned an unexpected job ID")

    def ping(self) -> bool:
        return bool(self._connection().ping())

    def positions(self, jobs: list[dict]) -> dict[str, int]:
        from rq import Queue

        connection = self._connection()
        result = {}
        for kind in {job["queue_name"] for job in jobs}:
            name = (
                self.cpu_queue
                if kind == "cpu" and self.cpu_queue
                else f"{kind}-{self.queue_suffix}"
                if self.queue_suffix and kind in {"cpu", "io", "converter-v21"}
                else kind
            )
            positions = {
                rq_id: index + 1
                for index, rq_id in enumerate(
                    Queue(name, connection=connection).job_ids
                )
            }
            for job in jobs:
                if job["queue_name"] == kind and job["rq_job_id"] in positions:
                    result[job["id"]] = positions[job["rq_job_id"]]
        return result

    def remove_pending(self, *, rq_job_id: str, queue_name: str) -> None:
        """Remove a job ID from pending queues without cancelling a running job."""

        from rq import Queue
        from rq.serializers import JSONSerializer

        connection = self._connection()
        queue_names = [queue_name]
        if queue_name == "cpu" and self.cpu_queue:
            queue_names.append(self.cpu_queue)
        if self.queue_suffix and queue_name in {"cpu", "io", "converter-v21"}:
            queue_names.append(f"{queue_name}-{self.queue_suffix}")
        for origin in dict.fromkeys(queue_names):
            Queue(
                origin,
                connection=connection,
                serializer=JSONSerializer,
            ).remove(rq_job_id)


@dataclass(slots=True)
class RecordingDispatcher:
    """Small deterministic dispatcher used by API tests."""

    available: bool = True
    enqueued: list[tuple[str, str]] = field(default_factory=list, init=False)
    removed: list[tuple[str, str]] = field(default_factory=list, init=False)

    def enqueue(
        self,
        *,
        job_id: str,
        queue_name: str,
        rq_job_id: str,
        job_timeout: int | None = None,
    ) -> None:
        if not self.available:
            raise ConnectionError("queue unavailable")
        item = (job_id, queue_name)
        if item not in self.enqueued:
            self.enqueued.append(item)

    def ping(self) -> bool:
        return self.available

    def positions(self, jobs: list[dict]) -> dict[str, int]:
        result = {}
        by_id = {job["id"]: job for job in jobs}
        counts: dict[str, int] = {}
        for job_id, queue in self.enqueued:
            job = by_id.get(job_id)
            if job and job["status"] == "queued" and not job.get("wait_reason"):
                counts[queue] = counts.get(queue, 0) + 1
                result[job_id] = counts[queue]
        return result

    def remove_pending(self, *, rq_job_id: str, queue_name: str) -> None:
        if not self.available:
            raise ConnectionError("queue unavailable")
        self.removed.append((rq_job_id, queue_name))
