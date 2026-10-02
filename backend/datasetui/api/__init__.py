"""HTTP API (/api/v1): one module per area, each exposing register(router, ctx).

  system       System health, host resources and delivery capabilities.
  profiles     Researcher profiles.
  jobs         Job listing, creation, cancellation and events.
  huggingface  Hugging Face discovery, revisions, imports and deletion.
  trash        Recoverable dataset trash: list, move to trash, restore, empty.
  datasets     Dataset registry: list, rename, read files, details.
  curation     Episode flags, annotations and curation recipes (runs, snapshots).
  processing   Merge, validation runs and v2.1 conversion jobs.
  delivery     Export-gated deliveries: NAS, Hugging Face, PC (key or password).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from datasetui.api import (
    curation,
    datasets,
    delivery,
    huggingface,
    jobs,
    processing,
    profiles,
    system,
    trash,
)
from datasetui.api.context import RouterContext
from datasetui.api.huggingface import _current_pointer_matches, _hf_dataset_status
from datasetui.config import Settings
from datasetui.database import Database, DatasetNotReadyError
from datasetui.delivery.workflow import dispatch_registered_job, reconcile_deliveries
from datasetui.huggingface import HuggingFaceGateway
from datasetui.queueing import QueueDispatcher

logger = logging.getLogger("datasetui.api")


def create_router(
    database: Database,
    dispatcher: QueueDispatcher,
    *,
    settings: Settings | None = None,
    hf_gateway: HuggingFaceGateway | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/v1")
    settings = settings or Settings.from_env()
    hf_gateway = hf_gateway or HuggingFaceGateway(settings.hf_read_token)

    def dispatch_job(job: dict[str, Any]) -> dict[str, Any]:
        return dispatch_registered_job(database, dispatcher, settings, job)

    def dispatch_library_job(job: dict[str, Any]) -> dict[str, Any]:
        try:
            return dispatch_job(job)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail={"message": "Job queue unavailable", "job_id": job["id"]},
            ) from exc

    def recover_expired_jobs() -> None:
        for job_id in database.requeue_expired_jobs():
            try:
                dispatch_job(database.get_job(job_id))
            except Exception:
                logger.exception("failed to redispatch expired job %s", job_id)
                database.record_dispatch_error(job_id, "Unable to dispatch job")
        reconcile_deliveries(database, dispatcher, settings)

    def dataset_job_payload(dataset_id: str) -> dict[str, str]:
        dataset = database.get_dataset(dataset_id)
        if not dataset["available"] or dataset["readiness"] != "ready":
            raise DatasetNotReadyError(dataset_id)
        return {
            "dataset_id": dataset["id"],
            "fingerprint": dataset["fingerprint"],
            "storage_area": dataset["storage_area"],
            "relative_path": dataset["relative_path"],
        }

    ctx = RouterContext(
        database=database,
        dispatcher=dispatcher,
        settings=settings,
        hf_gateway=hf_gateway,
        dispatch_job=dispatch_job,
        dispatch_library_job=dispatch_library_job,
        recover_expired_jobs=recover_expired_jobs,
        dataset_job_payload=dataset_job_payload,
    )
    for module in (system, profiles, jobs, huggingface, trash, datasets, curation, processing, delivery):
        module.register(router, ctx)
    return router


__all__ = ["create_router", "_current_pointer_matches", "_hf_dataset_status"]
