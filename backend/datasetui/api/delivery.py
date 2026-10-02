"""Export-gated deliveries: NAS, Hugging Face, PC (key or password)."""

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
)
from datasetui.delivery.workflow import create_delivery_workflow
from datasetui.models import (
    HuggingFaceDeliveryCreate,
    Job,
    NasDeliveryCreate,
    PcKeyDeliveryCreate,
    PcPasswordDeliveryCreate,
)

logger = logging.getLogger("datasetui.api")


def register(router: APIRouter, ctx: RouterContext) -> None:
    database = ctx.database
    dispatcher = ctx.dispatcher
    settings = ctx.settings
    dataset_job_payload = ctx.dataset_job_payload

    def create_delivery_job(
        *,
        dataset_id: str,
        profile_id: str,
        kind: str,
        extra: dict[str, Any],
        idempotency_key: str,
        response: Response,
    ) -> dict[str, Any]:
        internal_payload = {**dataset_job_payload(dataset_id), **extra}
        job, created = create_delivery_workflow(
            database,
            dispatcher,
            settings,
            kind=kind,
            profile_id=profile_id,
            payload=internal_payload,
            idempotency_key=idempotency_key,
        )
        if not created and job["status"] != "queued":
            response.status_code = status.HTTP_200_OK
            return job
        if not created:
            response.status_code = status.HTTP_200_OK
        return job

    @router.post(
        "/datasets/{dataset_id}/deliveries/nas",
        response_model=Job,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def create_nas_delivery(
        dataset_id: str, payload: NasDeliveryCreate, response: Response
    ) -> dict[str, Any]:
        try:
            return create_delivery_job(
                dataset_id=dataset_id,
                profile_id=payload.profile_id,
                kind="datasets.export_nas",
                extra={"output_name": payload.output_name},
                idempotency_key=payload.idempotency_key,
                response=response,
            )
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Dataset not found") from exc
        except ProfileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Profile not found") from exc
        except DatasetNotReadyError as exc:
            raise HTTPException(status_code=409, detail="Dataset is not ready") from exc
        except IdempotencyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("failed to dispatch NAS delivery")
            raise HTTPException(
                status_code=503, detail="Job queue unavailable"
            ) from exc

    @router.post(
        "/datasets/{dataset_id}/deliveries/huggingface",
        response_model=Job,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def create_hf_delivery(
        dataset_id: str, payload: HuggingFaceDeliveryCreate, response: Response
    ) -> dict[str, Any]:
        try:
            return create_delivery_job(
                dataset_id=dataset_id,
                profile_id=payload.profile_id,
                kind="datasets.upload_hf",
                extra={
                    "repo_name": payload.repo_name,
                    "visibility": payload.visibility,
                },
                idempotency_key=payload.idempotency_key,
                response=response,
            )
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Dataset not found") from exc
        except ProfileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Profile not found") from exc
        except DatasetNotReadyError as exc:
            raise HTTPException(status_code=409, detail="Dataset is not ready") from exc
        except IdempotencyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("failed to dispatch Hugging Face delivery")
            raise HTTPException(
                status_code=503, detail="Job queue unavailable"
            ) from exc

    @router.post(
        "/datasets/{dataset_id}/deliveries/pc/key",
        response_model=Job,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def create_pc_key_delivery(
        dataset_id: str, payload: PcKeyDeliveryCreate, response: Response
    ) -> dict[str, Any]:
        try:
            return create_delivery_job(
                dataset_id=dataset_id,
                profile_id=payload.profile_id,
                kind="datasets.copy_pc_key",
                extra={
                    "host": payload.host,
                    "port": payload.port,
                    "username": payload.username,
                    "destination": payload.destination,
                },
                idempotency_key=payload.idempotency_key,
                response=response,
            )
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Dataset not found") from exc
        except ProfileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Profile not found") from exc
        except DatasetNotReadyError as exc:
            raise HTTPException(status_code=409, detail="Dataset is not ready") from exc
        except IdempotencyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("failed to dispatch PC delivery")
            raise HTTPException(
                status_code=503, detail="Job queue unavailable"
            ) from exc

    @router.post(
        "/datasets/{dataset_id}/deliveries/pc/password",
        response_model=Job,
        status_code=202,
    )
    def create_pc_password_delivery(
        dataset_id: str, payload: PcPasswordDeliveryCreate, response: Response
    ) -> dict[str, Any]:
        from datasetui.credential_store import CredentialUnavailableError

        if not settings.credential_redis_url:
            raise HTTPException(
                status_code=409, detail="임시 비밀번호 저장소를 먼저 설정하세요."
            )
        try:
            job, created = create_delivery_workflow(
                database,
                dispatcher,
                settings,
                kind="datasets.copy_pc_password",
                profile_id=payload.profile_id,
                payload={
                    **dataset_job_payload(dataset_id),
                    "host": payload.host,
                    "port": payload.port,
                    "username": payload.username,
                    "destination": payload.destination,
                },
                idempotency_key=payload.idempotency_key,
                password=payload.password,
            )
            if not created:
                response.status_code = 200
            return job
        except (DatasetNotFoundError, ProfileNotFoundError) as exc:
            raise HTTPException(
                status_code=404, detail="Dataset or profile not found"
            ) from exc
        except (DatasetNotReadyError, IdempotencyConflictError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except CredentialUnavailableError as exc:
            raise HTTPException(
                status_code=503, detail="임시 비밀번호 저장소 설정을 확인하세요."
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="전송 대기열 또는 임시 비밀번호 저장소에 연결할 수 없습니다.",
            ) from exc
