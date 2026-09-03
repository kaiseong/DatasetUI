from __future__ import annotations

import logging

from datasetui.config import Settings
from datasetui.database import Database
from datasetui.datasets import DatasetRootUnavailableError
from datasetui.jobs import run_registered_job


logger = logging.getLogger("datasetui.worker")


def run_job(job_id: str) -> dict[str, object]:
    """RQ entrypoint. Redis carries only the opaque job ID."""

    settings = Settings.from_env()
    database = Database(settings.database_path)
    database.initialize()
    job = database.claim_job(job_id)
    if job is None:
        current = database.get_job(job_id)
        return {"job_id": job_id, "status": current["status"], "claimed": False}

    try:
        result = run_registered_job(job["kind"], job["payload"])
    except Exception as exc:
        logger.exception("job %s failed", job_id)
        error_code, public_message = _public_failure(exc)
        database.fail_job(job_id, error_code, public_message)
        raise

    finished = database.succeed_job(job_id, result)
    return {"job_id": job_id, "status": finished["status"], "claimed": True}


def _public_failure(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, DatasetRootUnavailableError):
        return "storage_unavailable", "Dataset storage is unavailable"
    return "job_failed", "The job could not be completed"
