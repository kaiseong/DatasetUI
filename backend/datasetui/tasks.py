from __future__ import annotations

import logging
import os
import socket
import threading
import uuid
from contextlib import contextmanager
from collections.abc import Iterator

from datasetui.config import Settings
from datasetui.database import Database, JobLeaseLostError, RecipeRevisionMismatchError
from datasetui.datasets import DatasetRootUnavailableError
from datasetui.hf_errors import (
    HuggingFaceDatasetNotFoundError,
    HuggingFaceImportConflictError,
    HuggingFaceImportTooLargeError,
    HuggingFaceImportValidationError,
    HuggingFaceRevisionNotFoundError,
    HuggingFaceUnavailableError,
)
from datasetui.jobs import run_registered_job
from datasetui.transform_errors import CurationTransformError


logger = logging.getLogger("datasetui.worker")


def run_job(job_id: str) -> dict[str, object]:
    """RQ entrypoint. Redis carries only the opaque job ID."""

    settings = Settings.from_env()
    database = Database(settings.database_path)
    database.initialize()
    worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"
    job = database.claim_job(
        job_id,
        worker_id=worker_id,
        lease_seconds=settings.job_lease_seconds,
    )
    if job is None:
        current = database.get_job(job_id)
        return {"job_id": job_id, "status": current["status"], "claimed": False}

    try:
        with _heartbeat_lease(
            database,
            job_id=job_id,
            worker_id=worker_id,
            lease_seconds=settings.job_lease_seconds,
            heartbeat_seconds=settings.job_heartbeat_seconds,
        ):
            if job["kind"] in {"hf.import", "curation.materialize"}:
                result = run_registered_job(
                    job["kind"],
                    job["payload"],
                    job_id=job_id,
                    worker_id=worker_id,
                )
            else:
                result = run_registered_job(job["kind"], job["payload"])
            database.assert_job_lease(job_id, worker_id=worker_id)
    except Exception as exc:
        logger.exception("job %s failed", job_id)
        error_code, public_message = _public_failure(exc)
        try:
            database.fail_job(
                job_id,
                error_code,
                public_message,
                worker_id=worker_id,
            )
        except JobLeaseLostError:
            logger.warning("job %s lost its lease before failure was recorded", job_id)
        raise

    finished = database.succeed_job(job_id, result, worker_id=worker_id)
    return {"job_id": job_id, "status": finished["status"], "claimed": True}


@contextmanager
def _heartbeat_lease(
    database: Database,
    *,
    job_id: str,
    worker_id: str,
    lease_seconds: int,
    heartbeat_seconds: int,
) -> Iterator[None]:
    if lease_seconds < 3:
        raise ValueError("job lease must be at least three seconds")
    interval = max(1, min(heartbeat_seconds, lease_seconds // 3))
    stopped = threading.Event()
    lost = threading.Event()

    def heartbeat() -> None:
        while not stopped.wait(interval):
            try:
                if not database.heartbeat_job(
                    job_id,
                    worker_id=worker_id,
                    lease_seconds=lease_seconds,
                ):
                    lost.set()
                    return
            except Exception:
                logger.exception("job %s heartbeat failed", job_id)
                lost.set()
                return

    thread = threading.Thread(
        target=heartbeat,
        name=f"datasetui-heartbeat-{job_id}",
        daemon=True,
    )
    thread.start()
    try:
        yield
        if lost.is_set():
            raise JobLeaseLostError(job_id)
    finally:
        stopped.set()
        thread.join(timeout=max(1, interval + 1))


def _public_failure(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, DatasetRootUnavailableError):
        return "storage_unavailable", "Dataset storage is unavailable"
    if isinstance(exc, HuggingFaceImportTooLargeError):
        return "hf_import_too_large", "Dataset exceeds the import size limit"
    if isinstance(
        exc,
        (HuggingFaceDatasetNotFoundError, HuggingFaceRevisionNotFoundError),
    ):
        return "hf_source_not_found", "The selected Hugging Face revision was not found"
    if isinstance(exc, HuggingFaceUnavailableError):
        return "hf_unavailable", "Hugging Face is temporarily unavailable"
    if isinstance(exc, HuggingFaceImportConflictError):
        return "hf_revision_conflict", "The immutable dataset revision conflicts"
    if isinstance(exc, HuggingFaceImportValidationError):
        return "hf_validation_failed", "The downloaded dataset failed validation"
    if isinstance(exc, JobLeaseLostError):
        return "worker_lost", "The worker stopped responding"
    if isinstance(exc, RecipeRevisionMismatchError):
        return "source_revision_changed", "The source dataset revision changed"
    if isinstance(exc, CurationTransformError):
        return "curation_failed", str(exc)
    return "job_failed", "The job could not be completed"
