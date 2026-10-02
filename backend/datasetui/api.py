from __future__ import annotations

import json
import logging
import os
import stat as stat_module
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse

from datasetui.config import Settings
from datasetui.database import (
    AnnotationRevisionConflictError,
    Database,
    DatasetNameConflictError,
    DatasetNotFoundError,
    DatasetNotReadyError,
    DatasetTrashConflictError,
    DuplicateProfileNameError,
    DuplicateRecipeNameError,
    FlagRevisionConflictError,
    IdempotencyConflictError,
    JobCancellationConflictError,
    JobNotFoundError,
    JobOwnershipError,
    ProfileNotFoundError,
    RecipeNotFoundError,
    RecipeRevisionMismatchError,
    ValidationRunActiveError,
    ValidationRunNotFoundError,
)
from datasetui.dataset_files import (
    DatasetFilePathError,
    DatasetFileUnavailableError,
    DatasetRangeError,
    iter_open_file,
    open_dataset_file,
    parse_byte_range,
)
from datasetui.dataset_trash import (
    DatasetTrashPathError,
    dataset_location_identity,
    dataset_trash_locations,
    identity_matches,
    move_dataset_to_trash,
    registered_dataset_identity,
    restore_dataset_from_trash,
)
from datasetui.delivery.workflow import (
    create_delivery_workflow,
    dispatch_registered_job,
    reconcile_deliveries,
)
from datasetui.hf_errors import (
    HuggingFaceDatasetNotFoundError,
    HuggingFaceRevisionNotFoundError,
    HuggingFaceUnavailableError,
)
from datasetui.huggingface import HuggingFaceGateway, HF_NAMESPACE
from datasetui.jobs import queue_for_kind, validate_job_payload
from datasetui.models import (
    CurationRecipe,
    CurationRecipeCreate,
    CurationRecipeSnapshot,
    CurationRecipeSnapshotCreate,
    CurationRecipeUpdate,
    CurationRunCreate,
    Dataset,
    DatasetConversionCreate,
    DatasetMergeCreate,
    DatasetReadiness,
    DatasetTrashCreate,
    DatasetTrashEmptyCreate,
    DatasetTrashEntry,
    DatasetTrashRestore,
    DatasetUpdate,
    DatasetValidationCreate,
    EpisodeAnnotations,
    EpisodeAnnotationsPut,
    EpisodeFlagPatch,
    EpisodeFlags,
    HuggingFaceDataset,
    HuggingFaceDeleteCreate,
    HuggingFaceDeliveryCreate,
    HuggingFaceImportCreate,
    HuggingFaceRevision,
    Job,
    JobCancel,
    JobCreate,
    JobEvent,
    JobStatus,
    NasDeliveryCreate,
    PcKeyDeliveryCreate,
    PcPasswordDeliveryCreate,
    Profile,
    ProfileCreate,
    ProfileUpdate,
    StorageArea,
    SystemHealth,
    ValidationRun,
)
from datasetui.queueing import QueueDispatcher



logger = logging.getLogger("datasetui.api")


class DatasetFileStreamingResponse(StreamingResponse):
    def __init__(self, descriptor: int, *args: Any, **kwargs: Any) -> None:
        self._dataset_file_descriptor = descriptor
        super().__init__(*args, **kwargs)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            os.close(self._dataset_file_descriptor)


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

    @router.get("/system/resources")
    def resource_status() -> dict[str, Any]:
        from datasetui.resource_admission import read_pool_status

        return read_pool_status(settings.jobs_root)

    @router.get("/delivery/capabilities")
    def delivery_capabilities() -> dict[str, bool | str]:
        return {
            "hf_upload_configured": bool(
                settings.hf_write_token or settings.hf_upload_configured
            ),
            "hf_namespace": HF_NAMESPACE,
            "hf_delete_configured": bool(
                settings.hf_write_token or settings.hf_upload_configured
            ),
            "pc_password_configured": bool(settings.credential_redis_url),
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

    @router.get("/hf/datasets", response_model=list[HuggingFaceDataset])
    def list_huggingface_datasets(
        q: str | None = Query(default=None, max_length=100),
        limit: int = Query(default=100, ge=1, le=200),
    ) -> list[dict[str, Any]]:
        recover_expired_jobs()
        try:
            remote_datasets = hf_gateway.list_datasets(query=q, limit=limit)
        except HuggingFaceUnavailableError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
            ) from exc

        sources = {item["repo_id"]: item for item in database.list_hf_sources()}
        latest_jobs: dict[str, dict[str, Any]] = {}
        for job in database.list_jobs(limit=200):
            if job["kind"] != "hf.import":
                continue
            repo_id = job["payload"].get("repo_id")
            if isinstance(repo_id, str) and repo_id not in latest_jobs:
                latest_jobs[repo_id] = job

        return [
            {
                **remote,
                "current_commit_sha": (
                    sources.get(remote["repo_id"], {}).get("current_commit_sha")
                ),
                "status": _hf_dataset_status(
                    remote=remote,
                    source=sources.get(remote["repo_id"]),
                    latest_job=latest_jobs.get(remote["repo_id"]),
                    raw_root=settings.nas_root / "raw",
                ),
            }
            for remote in remote_datasets
        ]

    @router.get(
        "/hf/datasets/{dataset_name}/revisions",
        response_model=list[HuggingFaceRevision],
    )
    def list_huggingface_revisions(dataset_name: str) -> list[dict[str, str]]:
        try:
            return hf_gateway.list_revisions(dataset_name)
        except HuggingFaceDatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, HuggingFaceRevisionNotFoundError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc
        except HuggingFaceUnavailableError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
            ) from exc

    @router.post(
        "/hf/imports",
        response_model=Job,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def create_huggingface_import(
        payload: HuggingFaceImportCreate,
        response: Response,
    ) -> dict[str, Any]:
        try:
            resolved = hf_gateway.resolve_revision(
                payload.dataset_name,
                payload.requested_revision,
            )
        except HuggingFaceDatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except HuggingFaceRevisionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except HuggingFaceUnavailableError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
            ) from exc
        if resolved["commit_sha"] != payload.commit_sha:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The selected revision changed. Reload revisions and try again.",
            )
        total_bytes = resolved.get("total_bytes")
        if total_bytes is not None and total_bytes > settings.hf_import_max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Dataset exceeds the configured import size limit",
            )

        try:
            job, created = database.create_hf_import_job(
                profile_id=payload.profile_id,
                repo_id=f"{HF_NAMESPACE}/{payload.dataset_name}",
                dataset_name=payload.dataset_name,
                requested_revision=payload.requested_revision,
                commit_sha=payload.commit_sha,
                expected_file_count=resolved["file_count"],
                expected_total_bytes=total_bytes,
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
            logger.exception("failed to dispatch Hugging Face import %s", job["id"])
            job = database.record_dispatch_error(job["id"], "Unable to dispatch job")
            if job["status"] != "queued":
                return job
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"message": "Job queue unavailable", "job_id": job["id"]},
            ) from exc

    @router.post(
        "/hf/datasets/{dataset_name}/delete", response_model=Job, status_code=202
    )
    def delete_hf_dataset(
        dataset_name: str, payload: HuggingFaceDeleteCreate, response: Response
    ):
        from datasetui.huggingface import validate_dataset_name

        try:
            name = validate_dataset_name(dataset_name)
            if payload.expected_repo_id != f"{HF_NAMESPACE}/{name}":
                raise ValueError("HF confirmation target mismatch")
            if not (settings.hf_write_token or settings.hf_upload_configured):
                raise HTTPException(
                    status_code=409, detail="HF 삭제 권한이 설정되지 않았습니다."
                )
            job, created = database.create_job(
                kind="hf.delete",
                queue_name="io",
                profile_id=payload.profile_id,
                payload={
                    "repo_id": payload.expected_repo_id,
                    "dataset_name": name,
                    "expected_commit_sha": payload.expected_commit_sha,
                },
                idempotency_key=payload.idempotency_key,
            )
            if not created:
                response.status_code = 200
            return dispatch_library_job(job)
        except (ValueError, IdempotencyConflictError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ProfileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Profile not found") from exc

    @router.post("/dataset-trash/empty", response_model=Job, status_code=202)
    def empty_dataset_trash(payload: DatasetTrashEmptyCreate, response: Response):
        from datasetui.library_operations import (
            reserve_trash_items,
            validate_library_operation,
        )

        try:
            internal = validate_library_operation(
                "datasets.empty_trash",
                {"items": [item.model_dump() for item in payload.items]},
            )
            job, created = database.create_job(
                kind="datasets.empty_trash",
                queue_name="io",
                profile_id=payload.profile_id,
                payload=internal,
                idempotency_key=payload.idempotency_key,
            )
            if job["status"] == "queued":
                reserve_trash_items(database, job)
            if not created:
                response.status_code = 200
            return dispatch_library_job(job)
        except (ValueError, IdempotencyConflictError, DatasetTrashConflictError) as exc:
            raise HTTPException(
                status_code=409,
                detail="휴지통 상태가 변경됐거나 다른 작업이 진행 중입니다.",
            ) from exc
        except ProfileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Profile not found") from exc

    @router.get("/datasets", response_model=list[Dataset])
    def list_datasets(
        storage_area: StorageArea | None = None,
        readiness: DatasetReadiness | None = None,
        include_missing: bool = False,
        limit: int = Query(default=100, ge=1, le=500),
        sort: Literal["name_asc", "name_desc", "newest", "oldest"] = "name_asc",
        q: str | None = Query(default=None, max_length=160),
        offset: int = Query(default=0, ge=0),
    ) -> list[dict[str, Any]]:
        return database.list_datasets(
            storage_area=storage_area,
            readiness=readiness,
            include_missing=include_missing,
            limit=limit,
            sort=sort,
            query=q,
            offset=offset,
        )

    def public_trash(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "dataset": record["dataset"],
            "original_relative_path": record["original_relative_path"],
            "trashed_at": record["trashed_at"],
            "state": record["state"],
            "requested_by_profile_id": record["requested_by_profile_id"],
        }

    def trash_conflict(code: str) -> HTTPException:
        messages = {
            "confirmation_mismatch": "데이터셋 이름 또는 내용 지문이 변경되었습니다.",
            "dataset_in_use": "진행 중이거나 대기 중인 작업이 이 데이터셋을 사용합니다.",
            "dataset_already_trashed": "이미 휴지통에 있는 데이터셋입니다.",
            "trash_path_conflict": "안전하게 이동할 수 없는 데이터셋 경로입니다.",
            "restore_path_occupied": "원래 경로가 이미 사용 중이라 복원할 수 없습니다.",
            "trash_recovery_required": "중단된 휴지통 작업을 관리자 확인 후 복구해야 합니다.",
            "trash_operation_in_progress": "휴지통 작업이 진행 중입니다. 중단된 작업이면 1분 후 다시 시도하세요.",
        }
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": code, "message": messages[code]},
        )

    def stale_trash_operation(record: dict[str, Any]) -> bool:
        updated = datetime.fromisoformat(record["updated_at"].replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - updated).total_seconds() >= 60

    def trash_location_state(record: dict[str, Any]) -> tuple[bool, bool]:
        try:
            return dataset_trash_locations(settings.nas_root, record)
        except DatasetTrashPathError as exc:
            database.mark_dataset_trash_recovery_required(record["dataset_id"])
            raise DatasetTrashConflictError("trash_recovery_required") from exc

    @router.get("/dataset-trash", response_model=list[DatasetTrashEntry])
    def list_dataset_trash(
        profile_id: str, limit: int = Query(default=100, ge=1, le=500)
    ) -> list[dict[str, Any]]:
        try:
            return [
                public_trash(record)
                for record in database.list_dataset_trash(
                    profile_id=profile_id, limit=limit
                )
            ]
        except ProfileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "profile_not_found",
                    "message": "프로필을 찾을 수 없습니다.",
                },
            ) from exc

    @router.post("/datasets/{dataset_id}/trash", response_model=DatasetTrashEntry)
    def trash_dataset(dataset_id: str, payload: DatasetTrashCreate) -> dict[str, Any]:
        try:
            try:
                existing = database.get_dataset_trash(dataset_id)
            except DatasetNotFoundError:
                existing = None
            if existing is None:
                current_dataset = database.get_dataset(dataset_id)
                record = database.prepare_dataset_trash(
                    dataset_id,
                    profile_id=payload.profile_id,
                    expected_name=payload.expected_name,
                    expected_fingerprint=payload.expected_fingerprint,
                )
                try:
                    source_device, source_inode = registered_dataset_identity(
                        settings.nas_root,
                        storage_area=current_dataset["storage_area"],
                        relative_path=current_dataset["relative_path"],
                    )
                except DatasetTrashPathError:
                    database.abort_dataset_trash(dataset_id)
                    raise
                record = database.record_dataset_trash_source_identity(
                    dataset_id, source_device=source_device, source_inode=source_inode
                )
            else:
                database.get_profile(payload.profile_id)
                if (
                    existing["dataset"]["name"] != payload.expected_name
                    or existing["dataset"]["fingerprint"] != payload.expected_fingerprint
                ):
                    raise DatasetTrashConflictError("confirmation_mismatch")
                if existing["state"] != "moving":
                    raise DatasetTrashConflictError("dataset_already_trashed")
                if existing["requested_by_profile_id"] != payload.profile_id:
                    raise DatasetTrashConflictError("dataset_already_trashed")
                if not stale_trash_operation(existing):
                    raise DatasetTrashConflictError("trash_operation_in_progress")
                original_exists, trash_exists = trash_location_state(existing)
                if trash_exists and not original_exists:
                    if not identity_matches(
                        existing,
                        dataset_location_identity(settings.nas_root, existing, trashed=True),
                    ):
                        database.mark_dataset_trash_recovery_required(dataset_id)
                        raise DatasetTrashConflictError("trash_recovery_required")
                    return public_trash(database.finalize_dataset_trash(dataset_id))
                if original_exists and not trash_exists:
                    current_identity = dataset_location_identity(
                        settings.nas_root, existing, trashed=False
                    )
                    if existing.get("source_device") is None:
                        record = database.record_dataset_trash_source_identity(
                            dataset_id,
                            source_device=current_identity[0],
                            source_inode=current_identity[1],
                        )
                    elif identity_matches(existing, current_identity):
                        record = existing
                    else:
                        database.mark_dataset_trash_recovery_required(dataset_id)
                        raise DatasetTrashConflictError("trash_recovery_required")
                else:
                    database.mark_dataset_trash_recovery_required(dataset_id)
                    raise DatasetTrashConflictError("trash_recovery_required")

            try:
                move_dataset_to_trash(
                    settings.nas_root,
                    record,
                    registered_locations=database.dataset_locations_for_trash(
                        dataset_id
                    ),
                )
            except DatasetTrashPathError as exc:
                original_exists, trash_exists = trash_location_state(record)
                if exc.code == "trash_recovery_required":
                    database.mark_dataset_trash_recovery_required(dataset_id)
                elif original_exists and not trash_exists:
                    database.abort_dataset_trash(dataset_id)
                else:
                    database.mark_dataset_trash_recovery_required(dataset_id)
                raise DatasetTrashConflictError(exc.code) from exc
            return public_trash(database.finalize_dataset_trash(dataset_id))
        except (DatasetNotFoundError, ProfileNotFoundError) as exc:
            code = (
                "profile_not_found"
                if isinstance(exc, ProfileNotFoundError)
                else "dataset_not_found"
            )
            raise HTTPException(
                status_code=404,
                detail={"code": code, "message": "대상을 찾을 수 없습니다."},
            ) from exc
        except DatasetTrashConflictError as exc:
            raise trash_conflict(exc.code) from exc
        except DatasetTrashPathError as exc:
            raise trash_conflict(exc.code) from exc

    @router.post("/dataset-trash/{dataset_id}/restore", response_model=Dataset)
    def restore_dataset(
        dataset_id: str, payload: DatasetTrashRestore
    ) -> dict[str, Any]:
        try:
            try:
                record = database.prepare_dataset_restore(
                    dataset_id,
                    profile_id=payload.profile_id,
                    expected_fingerprint=payload.expected_fingerprint,
                )
            except DatasetTrashConflictError as exc:
                existing = database.get_dataset_trash(dataset_id)
                if (
                    exc.code != "trash_recovery_required"
                    or existing["state"] != "restoring"
                ):
                    raise
                if existing["dataset"]["fingerprint"] != payload.expected_fingerprint:
                    raise DatasetTrashConflictError("confirmation_mismatch")
                if existing["requested_by_profile_id"] != payload.profile_id:
                    raise
                if not stale_trash_operation(existing):
                    raise DatasetTrashConflictError("trash_operation_in_progress")
                original_exists, trash_exists = trash_location_state(existing)
                if original_exists and not trash_exists:
                    if not identity_matches(
                        existing,
                        dataset_location_identity(settings.nas_root, existing, trashed=False),
                    ):
                        database.mark_dataset_trash_recovery_required(dataset_id)
                        raise DatasetTrashConflictError("trash_recovery_required")
                    return database.finalize_dataset_restore(dataset_id)
                if trash_exists and not original_exists:
                    if not identity_matches(
                        existing,
                        dataset_location_identity(settings.nas_root, existing, trashed=True),
                    ):
                        database.mark_dataset_trash_recovery_required(dataset_id)
                        raise DatasetTrashConflictError("trash_recovery_required")
                    record = database.return_dataset_restore_to_trash(dataset_id)
                    record = database.prepare_dataset_restore(
                        dataset_id,
                        profile_id=payload.profile_id,
                        expected_fingerprint=payload.expected_fingerprint,
                    )
                else:
                    database.mark_dataset_trash_recovery_required(dataset_id)
                    raise DatasetTrashConflictError("trash_recovery_required")
            try:
                restore_dataset_from_trash(settings.nas_root, record)
            except DatasetTrashPathError as exc:
                if exc.code == "restore_path_occupied":
                    try:
                        retained_identity = dataset_location_identity(
                            settings.nas_root, record, trashed=True
                        )
                    except DatasetTrashPathError:
                        database.mark_dataset_trash_recovery_required(dataset_id)
                        raise DatasetTrashConflictError(
                            "trash_recovery_required"
                        ) from exc
                    if identity_matches(record, retained_identity):
                        database.return_dataset_restore_to_trash(dataset_id)
                        raise DatasetTrashConflictError(exc.code) from exc
                    database.mark_dataset_trash_recovery_required(dataset_id)
                    raise DatasetTrashConflictError("trash_recovery_required") from exc
                original_exists, trash_exists = trash_location_state(record)
                if exc.code == "trash_recovery_required":
                    database.mark_dataset_trash_recovery_required(dataset_id)
                elif trash_exists and not original_exists:
                    database.return_dataset_restore_to_trash(dataset_id)
                else:
                    database.mark_dataset_trash_recovery_required(dataset_id)
                raise DatasetTrashConflictError(exc.code) from exc
            return database.finalize_dataset_restore(dataset_id)
        except (DatasetNotFoundError, ProfileNotFoundError) as exc:
            code = (
                "profile_not_found"
                if isinstance(exc, ProfileNotFoundError)
                else "dataset_not_found"
            )
            raise HTTPException(
                status_code=404,
                detail={"code": code, "message": "대상을 찾을 수 없습니다."},
            ) from exc
        except DatasetTrashConflictError as exc:
            raise trash_conflict(exc.code) from exc

    @router.patch("/datasets/{dataset_id}", response_model=Dataset)
    def update_dataset(dataset_id: str, payload: DatasetUpdate) -> dict[str, Any]:
        try:
            return database.update_dataset_name(
                dataset_id,
                name=payload.name,
                expected_name=payload.expected_name,
            )
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Dataset not found") from exc
        except DatasetNameConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="데이터셋 이름이 다른 곳에서 변경되었습니다. 새로고침 후 다시 시도해 주세요.",
            ) from exc

    @router.get("/datasets/{dataset_id}/flags", response_model=EpisodeFlags)
    def get_episode_flags(dataset_id: str, profile_id: str) -> dict[str, Any]:
        try:
            return database.get_episode_flags(
                dataset_id=dataset_id, profile_id=profile_id
            )
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Dataset not found") from exc
        except ProfileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Profile not found") from exc
        except DatasetNotReadyError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Dataset is not ready for curation",
            ) from exc

    @router.patch("/datasets/{dataset_id}/flags", response_model=EpisodeFlags)
    def update_episode_flags(
        dataset_id: str, payload: EpisodeFlagPatch
    ) -> dict[str, Any]:
        try:
            return database.update_episode_flags(
                dataset_id=dataset_id,
                profile_id=payload.profile_id,
                expected_revision=payload.expected_revision,
                changes=[change.model_dump() for change in payload.changes],
            )
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Dataset not found") from exc
        except ProfileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Profile not found") from exc
        except DatasetNotReadyError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Dataset is not ready for curation",
            ) from exc
        except FlagRevisionConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Flags changed in another session. Reload and try again.",
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc

    @router.get(
        "/datasets/{dataset_id}/annotations/{episode_index}",
        response_model=EpisodeAnnotations,
    )
    def get_episode_annotations(
        dataset_id: str, episode_index: int, profile_id: str
    ) -> dict[str, Any]:
        try:
            return database.get_episode_annotations(
                dataset_id=dataset_id,
                profile_id=profile_id,
                episode_index=episode_index,
            )
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Dataset not found") from exc
        except ProfileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Profile not found") from exc
        except DatasetNotReadyError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Dataset is not ready for annotation",
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc

    @router.put(
        "/datasets/{dataset_id}/annotations/{episode_index}",
        response_model=EpisodeAnnotations,
    )
    def replace_episode_annotations(
        dataset_id: str, episode_index: int, payload: EpisodeAnnotationsPut
    ) -> dict[str, Any]:
        try:
            return database.replace_episode_annotations(
                dataset_id=dataset_id,
                profile_id=payload.profile_id,
                episode_index=episode_index,
                expected_revision=payload.expected_revision,
                task_override=payload.task_override,
                atoms=[atom.model_dump(mode="json") for atom in payload.atoms],
            )
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Dataset not found") from exc
        except ProfileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Profile not found") from exc
        except DatasetNotReadyError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Dataset is not ready for annotation",
            ) from exc
        except AnnotationRevisionConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Annotations changed in another session. Reload and try again.",
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc

    @router.get("/datasets/{dataset_id}/recipes", response_model=list[CurationRecipe])
    def list_curation_recipes(
        dataset_id: str,
        profile_id: str,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        try:
            return database.list_curation_recipes(
                dataset_id=dataset_id,
                profile_id=profile_id,
                include_archived=include_archived,
            )
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Dataset not found") from exc
        except ProfileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Profile not found") from exc
        except DatasetNotReadyError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Dataset is not ready for curation",
            ) from exc

    @router.post(
        "/datasets/{dataset_id}/recipes",
        response_model=CurationRecipe,
        status_code=status.HTTP_201_CREATED,
    )
    def create_curation_recipe(
        dataset_id: str, payload: CurationRecipeCreate
    ) -> dict[str, Any]:
        try:
            return database.create_curation_recipe(
                dataset_id=dataset_id,
                profile_id=payload.profile_id,
                name=payload.name,
                selection_mode=payload.selection_mode,
                operation=payload.operation,
                trim_config=payload.trim_config.model_dump(mode="json"),
                include_annotations=payload.include_annotations,
                relative_action=payload.relative_action.model_dump(mode="json"),
                split_config=payload.split_config.model_dump(mode="json"),
            )
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Dataset not found") from exc
        except ProfileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Profile not found") from exc
        except DatasetNotReadyError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Dataset is not ready for curation",
            ) from exc
        except DuplicateRecipeNameError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A recipe with this name already exists",
            ) from exc

    @router.patch("/recipes/{recipe_id}", response_model=CurationRecipe)
    def update_curation_recipe(
        recipe_id: str, payload: CurationRecipeUpdate
    ) -> dict[str, Any]:
        try:
            return database.update_curation_recipe(
                recipe_id,
                profile_id=payload.profile_id,
                name=payload.name,
                selection_mode=payload.selection_mode,
                operation=payload.operation,
                trim_config=(
                    payload.trim_config.model_dump(mode="json")
                    if payload.trim_config is not None
                    else None
                ),
                include_annotations=payload.include_annotations,
                relative_action=(
                    payload.relative_action.model_dump(mode="json")
                    if payload.relative_action is not None
                    else None
                ),
                split_config=(
                    payload.split_config.model_dump(mode="json")
                    if payload.split_config is not None
                    else None
                ),
                archived=payload.archived,
            )
        except (RecipeNotFoundError, ProfileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail="Recipe not found") from exc
        except DuplicateRecipeNameError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A recipe with this name already exists",
            ) from exc
        except (DatasetNotReadyError, RecipeRevisionMismatchError) as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The recipe belongs to a different dataset revision",
            ) from exc

    @router.post(
        "/recipes/{recipe_id}/runs",
        response_model=Job,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def run_curation_recipe(
        recipe_id: str, payload: CurationRunCreate, response: Response
    ) -> dict[str, Any]:
        def matching_existing_job(existing: dict[str, Any]) -> dict[str, Any]:
            if existing["kind"] != "curation.materialize":
                raise IdempotencyConflictError(
                    "idempotency key is already bound to a different request"
                )
            existing_snapshot = database.get_curation_snapshot(
                existing["payload"].get("snapshot_id", "")
            )
            if (
                existing_snapshot["recipe_id"] != recipe_id
                or existing["payload"].get("output_name") != payload.output_name
            ):
                raise IdempotencyConflictError(
                    "idempotency key is already bound to a different request"
                )
            return existing

        try:
            existing = database.get_job_for_idempotency(
                payload.profile_id, payload.idempotency_key
            )
            if existing is not None:
                existing = matching_existing_job(existing)
                if existing["status"] == "queued":
                    existing = dispatch_job(existing)
                response.status_code = status.HTTP_200_OK
                return existing
            snapshot = database.snapshot_curation_recipe(
                recipe_id, profile_id=payload.profile_id
            )
            internal_payload = {
                "snapshot_id": snapshot["id"],
                "output_name": payload.output_name,
            }
            try:
                job, created = database.create_job(
                    kind="curation.materialize",
                    queue_name="cpu",
                    profile_id=payload.profile_id,
                    payload=internal_payload,
                    idempotency_key=payload.idempotency_key,
                )
            except IdempotencyConflictError:
                raced_job = database.get_job_for_idempotency(
                    payload.profile_id, payload.idempotency_key
                )
                if raced_job is None:
                    raise
                job = matching_existing_job(raced_job)
                created = False
            if created:
                database.record_curation_run(
                    job_id=job["id"],
                    snapshot_id=snapshot["id"],
                    output_name=payload.output_name,
                )
            elif job["status"] != "queued":
                response.status_code = status.HTTP_200_OK
                return job
            job = dispatch_job(job)
            if not created:
                response.status_code = status.HTTP_200_OK
            return job
        except (RecipeNotFoundError, ProfileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail="Recipe not found") from exc
        except (DatasetNotReadyError, RecipeRevisionMismatchError) as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The recipe belongs to a different dataset revision",
            ) from exc
        except IdempotencyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("failed to dispatch curation run")
            if "job" in locals():
                database.record_dispatch_error(job["id"], "Unable to dispatch job")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Job queue unavailable",
            ) from exc

    @router.post(
        "/recipes/{recipe_id}/snapshots",
        response_model=CurationRecipeSnapshot,
        status_code=status.HTTP_201_CREATED,
    )
    def snapshot_curation_recipe(
        recipe_id: str, payload: CurationRecipeSnapshotCreate
    ) -> dict[str, Any]:
        try:
            return database.snapshot_curation_recipe(
                recipe_id, profile_id=payload.profile_id
            )
        except (RecipeNotFoundError, ProfileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail="Recipe not found") from exc
        except (DatasetNotReadyError, RecipeRevisionMismatchError) as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The recipe belongs to a different dataset revision",
            ) from exc

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

    @router.api_route(
        "/datasets/{dataset_id}/files/{file_path:path}",
        methods=["GET", "HEAD"],
        response_class=StreamingResponse,
    )
    def read_dataset_file(
        dataset_id: str,
        file_path: str,
        request: Request,
        range_header: str | None = Header(default=None, alias="Range"),
    ) -> Response:
        try:
            dataset = database.get_dataset(dataset_id)
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Dataset not found") from exc
        if not dataset["available"]:
            raise HTTPException(status_code=404, detail="Dataset not found")
        if dataset["readiness"] != "ready":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Dataset is not ready for Viewer",
            )

        try:
            opened = open_dataset_file(
                nas_root=settings.nas_root,
                storage_area=dataset["storage_area"],
                dataset_relative_path=dataset["relative_path"],
                requested_path=file_path,
            )
        except DatasetFilePathError as exc:
            raise HTTPException(
                status_code=400, detail="Invalid dataset file path"
            ) from exc
        except DatasetFileUnavailableError as exc:
            raise HTTPException(
                status_code=404, detail="Dataset file not found"
            ) from exc

        try:
            requested_range = parse_byte_range(range_header, opened.size)
        except DatasetRangeError:
            os.close(opened.descriptor)
            return Response(
                status_code=status.HTTP_416_RANGE_NOT_SATISFIABLE,
                headers={
                    "Accept-Ranges": "bytes",
                    "Content-Range": f"bytes */{opened.size}",
                    "Cache-Control": "private, no-store",
                },
            )

        selected_start = requested_range.start if requested_range else 0
        selected_length = requested_range.length if requested_range else opened.size
        headers = {
            "Accept-Ranges": "bytes",
            "Cache-Control": "private, no-store",
            "Content-Length": str(selected_length),
            "X-Content-Type-Options": "nosniff",
        }
        response_status = status.HTTP_206_PARTIAL_CONTENT if requested_range else 200
        if requested_range:
            headers["Content-Range"] = (
                f"bytes {requested_range.start}-{requested_range.end}/{opened.size}"
            )

        if request.method == "HEAD":
            os.close(opened.descriptor)
            return Response(
                status_code=response_status,
                media_type=opened.content_type,
                headers=headers,
            )

        return DatasetFileStreamingResponse(
            opened.descriptor,
            iter_open_file(
                opened.descriptor,
                start=selected_start,
                length=selected_length,
            ),
            status_code=response_status,
            media_type=opened.content_type,
            headers=headers,
        )

    @router.get("/datasets/{dataset_id}", response_model=Dataset)
    def get_dataset(dataset_id: str) -> dict[str, Any]:
        try:
            return database.get_dataset(dataset_id)
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Dataset not found") from exc

    return router


def _hf_dataset_status(
    *,
    remote: dict[str, Any],
    source: dict[str, Any] | None,
    latest_job: dict[str, Any] | None,
    raw_root: Path,
) -> str:
    if latest_job and latest_job["status"] in {"queued", "running"}:
        return "queued" if latest_job["status"] == "queued" else "downloading"
    if source and source["pointer_confirmed"] and source["current_commit_sha"]:
        relative_path = source.get("relative_path")
        revision_path = (
            raw_root / relative_path if isinstance(relative_path, str) else None
        )
        if (
            revision_path is None
            or revision_path.is_symlink()
            or not revision_path.is_dir()
            or not _current_pointer_matches(source=source, raw_root=raw_root)
        ):
            return "incomplete"
        return (
            "ready"
            if source["current_commit_sha"] == remote["latest_commit_sha"]
            else "update_available"
        )
    if latest_job and latest_job["status"] == "failed":
        return "validation_failed"
    if source:
        return "incomplete"
    return "not_downloaded"


def _current_pointer_matches(*, source: dict[str, Any], raw_root: Path) -> bool:
    repo_id = source.get("repo_id")
    if not isinstance(repo_id, str):
        return False
    namespace, separator, dataset_name = repo_id.partition("/")
    if separator != "/" or namespace != HF_NAMESPACE:
        return False

    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    directory_fd: int | None = None
    try:
        directory_fd = os.open(raw_root, directory_flags)
        for component in ("hf", HF_NAMESPACE, dataset_name):
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        pointer_fd = os.open("current.json", file_flags, dir_fd=directory_fd)
        try:
            pointer_stat = os.fstat(pointer_fd)
            if (
                not stat_module.S_ISREG(pointer_stat.st_mode)
                or pointer_stat.st_size > 16384
            ):
                return False
            chunks: list[bytes] = []
            remaining = 16385
            while remaining:
                chunk = os.read(pointer_fd, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            if len(raw) > 16384:
                return False
        finally:
            os.close(pointer_fd)
    except (OSError, ValueError):
        return False
    finally:
        if directory_fd is not None:
            os.close(directory_fd)

    try:
        pointer = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(pointer, dict)
        and pointer.get("schema_version") == 1
        and pointer.get("repo_id") == repo_id
        and pointer.get("commit_sha") == source.get("current_commit_sha")
        and pointer.get("generation") == source.get("current_generation")
        and pointer.get("manifest_sha256") == source.get("manifest_sha256")
    )
