from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response, status

from datasetui.database import (
    Database,
    DuplicateProfileNameError,
    IdempotencyConflictError,
    JobNotFoundError,
    ProfileNotFoundError,
)
from datasetui.jobs import queue_for_kind, validate_job_payload
from datasetui.models import (
    Job,
    JobCreate,
    JobEvent,
    JobStatus,
    Profile,
    ProfileCreate,
    ProfileUpdate,
    SystemHealth,
)
from datasetui.queueing import QueueDispatcher


logger = logging.getLogger("datasetui.api")


def create_router(database: Database, dispatcher: QueueDispatcher) -> APIRouter:
    router = APIRouter(prefix="/api/v1")

    @router.get("/system/health", response_model=SystemHealth)
    def system_health(response: Response) -> dict[str, Any]:
        database_ok = False
        queue_ok = False
        try:
            database_ok = database.ping()
        except Exception:
            database_ok = False
        try:
            queue_ok = dispatcher.ping()
        except Exception:
            queue_ok = False
        if not database_ok or not queue_ok:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "ok": database_ok and queue_ok,
            "service": "datasetui-workbench",
            "database": "ok" if database_ok else "error",
            "queue": "ok" if queue_ok else "error",
            "schema_versions": database.schema_versions() if database_ok else [],
        }

    @router.get("/profiles", response_model=list[Profile])
    def list_profiles(include_archived: bool = False) -> list[dict[str, Any]]:
        return database.list_profiles(include_archived=include_archived)

    @router.post(
        "/profiles", response_model=Profile, status_code=status.HTTP_201_CREATED
    )
    def create_profile(payload: ProfileCreate) -> dict[str, Any]:
        try:
            return database.create_profile(payload.name)
        except DuplicateProfileNameError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A profile with this name already exists",
            ) from exc

    @router.get("/profiles/{profile_id}", response_model=Profile)
    def get_profile(profile_id: str) -> dict[str, Any]:
        try:
            return database.get_profile(profile_id)
        except ProfileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Profile not found") from exc

    @router.patch("/profiles/{profile_id}", response_model=Profile)
    def update_profile(profile_id: str, payload: ProfileUpdate) -> dict[str, Any]:
        try:
            return database.update_profile(
                profile_id,
                name=payload.name,
                archived=payload.archived,
            )
        except ProfileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Profile not found") from exc
        except DuplicateProfileNameError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A profile with this name already exists",
            ) from exc

    @router.get("/jobs", response_model=list[Job])
    def list_jobs(
        profile_id: str | None = None,
        job_status: JobStatus | None = Query(default=None, alias="status"),
        limit: int = Query(default=100, ge=1, le=200),
    ) -> list[dict[str, Any]]:
        return database.list_jobs(
            profile_id=profile_id,
            status=job_status,
            limit=limit,
        )

    @router.post("/jobs", response_model=Job, status_code=status.HTTP_202_ACCEPTED)
    def create_job(payload: JobCreate, response: Response) -> dict[str, Any]:
        queue_name = queue_for_kind(payload.kind)
        if queue_name is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Unsupported job kind: {payload.kind}",
            )
        try:
            validated_payload = validate_job_payload(payload.kind, payload.payload)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc
        try:
            job, created = database.create_job(
                kind=payload.kind,
                queue_name=queue_name,
                profile_id=payload.profile_id,
                payload=validated_payload,
                idempotency_key=payload.idempotency_key,
            )
        except ProfileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Profile not found") from exc
        except IdempotencyConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        if not created and job["status"] != "queued":
            response.status_code = status.HTTP_200_OK
            return job
        rq_job_id = job["rq_job_id"] or f"datasetui-{job['id']}"
        if job["rq_job_id"] is None:
            job = database.mark_enqueued(job["id"], rq_job_id)
        try:
            dispatcher.enqueue(
                job_id=job["id"],
                queue_name=queue_name,
                rq_job_id=rq_job_id,
            )
            job = database.clear_dispatch_error(job["id"])
            if not created:
                response.status_code = status.HTTP_200_OK
            return job
        except Exception as exc:
            logger.exception("failed to dispatch job %s", job["id"])
            job = database.record_dispatch_error(job["id"], "Unable to dispatch job")
            if job["status"] != "queued":
                return job
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"message": "Job queue unavailable", "job_id": job["id"]},
            ) from exc

    @router.get("/jobs/{job_id}", response_model=Job)
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            return database.get_job(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc

    @router.get("/jobs/{job_id}/events", response_model=list[JobEvent])
    def list_job_events(job_id: str) -> list[dict[str, Any]]:
        try:
            return database.list_job_events(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc

    return router
