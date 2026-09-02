from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class QueueDispatcher(Protocol):
    def enqueue(self, *, job_id: str, queue_name: str, rq_job_id: str) -> None: ...

    def ping(self) -> bool: ...


@dataclass(slots=True)
class RQDispatcher:
    redis_url: str
    timeout_seconds: int

    def _connection(self):
        from redis import Redis

        return Redis.from_url(self.redis_url)

    def enqueue(self, *, job_id: str, queue_name: str, rq_job_id: str) -> None:
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
            queue_name,
            connection=connection,
            serializer=JSONSerializer,
            default_timeout=self.timeout_seconds,
        )
        rq_job = queue.enqueue(
            "datasetui.tasks.run_job",
            job_id,
            job_id=rq_job_id,
            result_ttl=86400,
            failure_ttl=604800,
        )
        if rq_job.id != rq_job_id:
            raise RuntimeError("RQ returned an unexpected job ID")

    def ping(self) -> bool:
        return bool(self._connection().ping())


@dataclass(slots=True)
class RecordingDispatcher:
    """Small deterministic dispatcher used by API tests."""

    available: bool = True
    enqueued: list[tuple[str, str]] = field(default_factory=list, init=False)

    def enqueue(self, *, job_id: str, queue_name: str, rq_job_id: str) -> None:
        if not self.available:
            raise ConnectionError("queue unavailable")
        item = (job_id, queue_name)
        if item not in self.enqueued:
            self.enqueued.append(item)

    def ping(self) -> bool:
        return self.available
