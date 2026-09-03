from __future__ import annotations

import json
import logging
import os
import stat as stat_module
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse

from datasetui.database import (
    AnnotationRevisionConflictError,
    Database,
    DatasetNotReadyError,
    DatasetNotFoundError,
    DuplicateRecipeNameError,
    DuplicateProfileNameError,
    FlagRevisionConflictError,
    IdempotencyConflictError,
    JobNotFoundError,
    ProfileNotFoundError,
    RecipeNotFoundError,
    RecipeRevisionMismatchError,
)
from datasetui.dataset_files import (
    DatasetFilePathError,
    DatasetFileUnavailableError,
    DatasetRangeError,
    iter_open_file,
    open_dataset_file,
    parse_byte_range,
)
from datasetui.config import Settings
from datasetui.huggingface import (
    HF_NAMESPACE,
    HuggingFaceGateway,
)
from datasetui.hf_errors import (
    HuggingFaceDatasetNotFoundError,
    HuggingFaceRevisionNotFoundError,
    HuggingFaceUnavailableError,
)
from datasetui.jobs import queue_for_kind, validate_job_payload
from datasetui.models import (
    CurationRecipe,
    CurationRecipeCreate,
    CurationRunCreate,
    CurationRecipeSnapshot,
    CurationRecipeSnapshotCreate,
    CurationRecipeUpdate,
    Dataset,
    DatasetReadiness,
    EpisodeFlagPatch,
    EpisodeFlags,
    EpisodeAnnotations,
    EpisodeAnnotationsPut,
    Job,
    JobCreate,
    JobEvent,
    JobStatus,
    HuggingFaceDataset,
    HuggingFaceImportCreate,
    HuggingFaceRevision,
    Profile,
    ProfileCreate,
    ProfileUpdate,
    SystemHealth,
    StorageArea,
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
        rq_job_id = job["rq_job_id"] or database.rq_job_id_for(job["id"])
        if job["rq_job_id"] is None:
            job = database.mark_enqueued(job["id"], rq_job_id)
        dispatcher.enqueue(
            job_id=job["id"],
            queue_name=job["queue_name"],
            rq_job_id=rq_job_id,
        )
        return database.clear_dispatch_error(job["id"])

    def recover_expired_jobs() -> None:
        for job_id in database.requeue_expired_jobs():
            try:
                dispatch_job(database.get_job(job_id))
            except Exception:
                logger.exception("failed to redispatch expired job %s", job_id)
                database.record_dispatch_error(job_id, "Unable to dispatch job")

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
        recover_expired_jobs()
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

    @router.get("/datasets", response_model=list[Dataset])
    def list_datasets(
        storage_area: StorageArea | None = None,
        readiness: DatasetReadiness | None = None,
        include_missing: bool = False,
        limit: int = Query(default=100, ge=1, le=500),
    ) -> list[dict[str, Any]]:
        return database.list_datasets(
            storage_area=storage_area,
            readiness=readiness,
            include_missing=include_missing,
            limit=limit,
        )

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
