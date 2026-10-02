"""Job listing, creation, cancellation and events."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response, status

from datasetui.api.context import RouterContext
from datasetui.database import (
    DatasetTrashConflictError,
    IdempotencyConflictError,
    JobCancellationConflictError,
    JobNotFoundError,
    JobOwnershipError,
    ProfileNotFoundError,
)
from datasetui.delivery.workflow import reconcile_deliveries
from datasetui.jobs import queue_for_kind, validate_job_payload
from datasetui.models import Job, JobCancel, JobCreate, JobEvent, JobStatus

logger = logging.getLogger("datasetui.api")


def register(router: APIRouter, ctx: RouterContext) -> None:
    database = ctx.database
    dispatcher = ctx.dispatcher
    settings = ctx.settings
    dispatch_job = ctx.dispatch_job
    recover_expired_jobs = ctx.recover_expired_jobs

    @router.get("/jobs", response_model=list[Job])
    def list_jobs(
        profile_id: str | None = None,
        job_status: JobStatus | None = Query(default=None, alias="status"),
        kind: str | None = Query(default=None, min_length=1, max_length=80),
        limit: int = Query(default=100, ge=1, le=200),
        active_only: bool = False,
        offset: int = Query(default=0, ge=0),
    ) -> list[dict[str, Any]]:
        recover_expired_jobs()
        jobs = database.list_jobs(
            profile_id=profile_id,
            status=job_status,
            kind=kind,
            limit=limit,
            active_only=active_only,
            offset=offset,
        )
        try:
            positions = (
                dispatcher.positions(jobs) if hasattr(dispatcher, "positions") else {}
            )
            for job in jobs:
                if job["status"] == "queued" and not job.get("wait_reason"):
                    job["queue_position"] = positions.get(job["id"])
        except Exception:
            pass  # A disconnected queue must never produce an invented rank.
        return jobs

    @router.post("/jobs", response_model=Job, status_code=status.HTTP_202_ACCEPTED)
    def create_job(payload: JobCreate, response: Response) -> dict[str, Any]:
        if payload.kind in {
            "hf.delete",
            "datasets.empty_trash",
            "datasets.delivery_preflight",
            "datasets.copy_pc_password",
        }:
            raise HTTPException(
                status_code=422, detail="Use the dedicated operation endpoint"
            )
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
        except DatasetTrashConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": exc.code,
                    "message": "휴지통에 보존된 데이터셋 경로와 충돌합니다.",
                },
            ) from exc
        if not created and job["status"] != "queued":
            response.status_code = status.HTTP_200_OK
            return job
        try:
            job = dispatch_job(job)
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

    @router.post("/jobs/{job_id}/cancel", response_model=Job)
    def cancel_job(job_id: str, payload: JobCancel) -> dict[str, Any]:
        try:
            job = database.request_job_cancellation(
                job_id, profile_id=payload.profile_id
            )
        except JobNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail="작업을 찾을 수 없습니다."
            ) from exc
        except JobOwnershipError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="이 프로필의 작업만 취소할 수 있습니다.",
            ) from exc
        except JobCancellationConflictError as exc:
            message = (
                "결과 게시를 시작한 작업은 안전하게 취소할 수 없습니다."
                if exc.reason == "finalizing"
                else f"{exc.job_status} 상태의 작업은 취소할 수 없습니다."
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=message,
            ) from exc

        if job["status"] == "cancelled" and job["rq_job_id"] is not None:
            try:
                dispatcher.remove_pending(
                    rq_job_id=job["rq_job_id"],
                    queue_name=job["queue_name"],
                )
            except Exception:
                logger.warning(
                    "failed to remove cancelled pending RQ job %s",
                    job["rq_job_id"],
                    exc_info=True,
                )
        reconcile_deliveries(database, dispatcher, settings)
        return job

    @router.get("/jobs/{job_id}/events", response_model=list[JobEvent])
    def list_job_events(job_id: str) -> list[dict[str, Any]]:
        try:
            return database.list_job_events(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc
