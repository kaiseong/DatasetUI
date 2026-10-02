"""Merge, validation runs and v2.1 conversion jobs."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Response, status

from datasetui.api.context import RouterContext
from datasetui.database import (
    DatasetNotFoundError,
    DatasetNotReadyError,
    IdempotencyConflictError,
    ProfileNotFoundError,
    ValidationRunActiveError,
    ValidationRunNotFoundError,
)
from datasetui.models import (
    DatasetConversionCreate,
    DatasetMergeCreate,
    DatasetValidationCreate,
    Job,
    ValidationRun,
)

logger = logging.getLogger("datasetui.api")


def register(router: APIRouter, ctx: RouterContext) -> None:
    database = ctx.database
    dispatch_job = ctx.dispatch_job
    recover_expired_jobs = ctx.recover_expired_jobs
    dataset_job_payload = ctx.dataset_job_payload

    @router.post(
        "/datasets/merge",
        response_model=Job,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def create_dataset_merge(
        payload: DatasetMergeCreate, response: Response
    ) -> dict[str, Any]:
        internal_sources: list[dict[str, str]] = []
        try:
            for dataset_id in payload.dataset_ids:
                dataset = database.get_dataset(dataset_id)
                if not dataset["available"] or dataset["readiness"] != "ready":
                    raise DatasetNotReadyError(dataset_id)
                internal_sources.append(
                    {"id": dataset["id"], "fingerprint": dataset["fingerprint"]}
                )
            internal_payload = {
                "sources": internal_sources,
                "output_name": payload.output_name,
                "robot_type": payload.robot_type,
            }
            job, created = database.create_job(
                kind="datasets.merge",
                queue_name="cpu",
                profile_id=payload.profile_id,
                payload=internal_payload,
                idempotency_key=payload.idempotency_key,
            )
            if not created and job["status"] != "queued":
                response.status_code = status.HTTP_200_OK
                return job
            job = dispatch_job(job)
            if not created:
                response.status_code = status.HTTP_200_OK
            return job
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Dataset not found") from exc
        except ProfileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Profile not found") from exc
        except DatasetNotReadyError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Every merge source must be ready",
            ) from exc
        except IdempotencyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("failed to dispatch dataset merge")
            if "job" in locals():
                database.record_dispatch_error(job["id"], "Unable to dispatch job")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Job queue unavailable",
            ) from exc

    @router.get(
        "/datasets/{dataset_id}/validations", response_model=list[ValidationRun]
    )
    def list_dataset_validations(dataset_id: str) -> list[dict[str, Any]]:
        try:
            recover_expired_jobs()
            return database.list_validation_runs(dataset_id)
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Dataset not found") from exc

    @router.delete(
        "/datasets/{dataset_id}/validations/{job_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    def delete_dataset_validation(dataset_id: str, job_id: str) -> Response:
        try:
            database.delete_validation_run(dataset_id, job_id)
        except ValidationRunNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail="검사 기록을 찾을 수 없습니다."
            ) from exc
        except ValidationRunActiveError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="진행 중인 검사는 삭제할 수 없습니다.",
            ) from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @router.post(
        "/datasets/{dataset_id}/validations",
        response_model=Job,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def create_dataset_validation(
        dataset_id: str, payload: DatasetValidationCreate, response: Response
    ) -> dict[str, Any]:
        try:
            internal_payload = {
                **dataset_job_payload(dataset_id),
                "mode": payload.mode,
            }
            job, created = database.create_job(
                kind="datasets.validate",
                queue_name="cpu",
                profile_id=payload.profile_id,
                payload=internal_payload,
                idempotency_key=payload.idempotency_key,
            )
            if created:
                database.record_validation_run(
                    job_id=job["id"],
                    dataset_id=dataset_id,
                    dataset_fingerprint=internal_payload["fingerprint"],
                    mode=payload.mode,
                )
            if not created and job["status"] != "queued":
                response.status_code = status.HTTP_200_OK
                return job
            job = dispatch_job(job)
            if not created:
                response.status_code = status.HTTP_200_OK
            return job
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Dataset not found") from exc
        except ProfileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Profile not found") from exc
        except DatasetNotReadyError as exc:
            raise HTTPException(status_code=409, detail="Dataset is not ready") from exc
        except IdempotencyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("failed to dispatch dataset validation")
            if "job" in locals():
                database.record_dispatch_error(job["id"], "Unable to dispatch job")
            raise HTTPException(
                status_code=503, detail="Job queue unavailable"
            ) from exc

    @router.post(
        "/datasets/{dataset_id}/conversions/v2.1",
        response_model=Job,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def create_v21_conversion(
        dataset_id: str, payload: DatasetConversionCreate, response: Response
    ) -> dict[str, Any]:
        try:
            dataset = database.get_dataset(dataset_id)
            if dataset["codebase_version"] != "v3.0":
                raise ValueError("Only v3.0 datasets can be converted to v2.1")
            internal_payload = {
                **dataset_job_payload(dataset_id),
                "output_name": payload.output_name,
            }
            job, created = database.create_job(
                kind="datasets.convert_v21",
                queue_name="converter-v21",
                profile_id=payload.profile_id,
                payload=internal_payload,
                idempotency_key=payload.idempotency_key,
            )
            if not created and job["status"] != "queued":
                response.status_code = status.HTTP_200_OK
                return job
            job = dispatch_job(job)
            if not created:
                response.status_code = status.HTTP_200_OK
            return job
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Dataset not found") from exc
        except ProfileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Profile not found") from exc
        except DatasetNotReadyError as exc:
            raise HTTPException(status_code=409, detail="Dataset is not ready") from exc
        except (IdempotencyConflictError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("failed to dispatch v2.1 conversion")
            if "job" in locals():
                database.record_dispatch_error(job["id"], "Unable to dispatch job")
            raise HTTPException(
                status_code=503, detail="Job queue unavailable"
            ) from exc
