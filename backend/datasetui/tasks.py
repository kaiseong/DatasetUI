from __future__ import annotations

import logging

from datasetui.config import Settings
from datasetui.database import Database
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
        database.fail_job(job_id, type(exc).__name__, str(exc))
        raise

    finished = database.succeed_job(job_id, result)
    return {"job_id": job_id, "status": finished["status"], "claimed": True}
