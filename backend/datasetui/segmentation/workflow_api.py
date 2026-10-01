"""HTTP routes for templates, batches and the object workspace."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from datasetui.config import Settings
from datasetui.content_integrity import ContentIntegrityError
from datasetui.database import (
    Database,
    DatasetNotFoundError,
    DatasetNotReadyError,
    IdempotencyConflictError,
    JobNotFoundError,
    ProfileNotFoundError,
    RecipeRevisionMismatchError,
)
from datasetui.delivery_workflow import dispatch_registered_job
from datasetui.models import Job
from datasetui.queueing import QueueDispatcher
from datasetui.segmentation import workflows as workflow
from datasetui.segmentation import workspace as workspace
from datasetui.segmentation.workflow_contract import (
    BatchApprove,
    BatchCreate,
    BatchExportCreate,
    BatchPrepare,
    BatchPreviewBind,
    ProfileRequest,
    TemplateSave,
)
from datasetui.segmentation.workspace import WorkspaceSave, WorkspaceScope
from datasetui.transform_errors import CurationTransformError


def create_segmentation_workflow_router(
    database: Database, dispatcher: QueueDispatcher, settings: Settings
) -> APIRouter:
    workflow.initialize_segmentation_workflows(database)
    workspace.initialize(database)
    router = APIRouter(prefix="/api/v1/segmentation")

    def invoke(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (
            DatasetNotFoundError,
            JobNotFoundError,
            ProfileNotFoundError,
            FileNotFoundError,
        ) as exc:
            raise HTTPException(
                404, "데이터셋, 템플릿 또는 작업을 찾을 수 없습니다."
            ) from exc
        except IdempotencyConflictError as exc:
            raise HTTPException(
                409, "같은 요청 키를 다른 작업에 사용할 수 없습니다."
            ) from exc
        except (
            ValueError,
            ContentIntegrityError,
            DatasetNotReadyError,
            RecipeRevisionMismatchError,
            CurationTransformError,
        ) as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.get("/workspace")
    def get_workspace(profile_id: UUID, dataset_id: UUID, episode_index: Annotated[int, Query(ge=0, le=999999)], video_key: Annotated[str, Query(min_length=1, max_length=240, pattern=r"^[A-Za-z0-9_.-]+$")]):
        return invoke(workspace.read_workspace, database, settings, WorkspaceScope(
            profile_id=profile_id, dataset_id=dataset_id,
            episode_index=episode_index, video_key=video_key))

    @router.put("/workspace")
    def put_workspace(payload: WorkspaceSave):
        return invoke(workspace.save_workspace, database, settings, payload)

    @router.get("/templates")
    def templates(profile_id: UUID):
        return {"templates": invoke(workflow.list_templates, database, str(profile_id))}

    @router.post("/templates", status_code=201)
    def save_template(payload: TemplateSave):
        return invoke(workflow.save_template, database, settings, payload)

    @router.patch("/templates/{template_id}")
    def update_template(template_id: UUID, payload: TemplateSave):
        return invoke(
            workflow.save_template, database, settings, payload, str(template_id)
        )

    @router.post("/templates/{template_id}/archive")
    def archive_template(template_id: UUID, payload: ProfileRequest):
        invoke(
            workflow.archive_template,
            database,
            str(template_id),
            str(payload.profile_id),
        )
        return {"archived": True}

    @router.get("/batches")
    def batches(profile_id: UUID):
        return {"batches": invoke(workflow.list_batches, database, str(profile_id))}

    @router.get("/batches/{batch_id}")
    def batch(batch_id: UUID, profile_id: UUID):
        return invoke(workflow.get_batch, database, str(batch_id), str(profile_id))

    @router.post("/batches/prepare", status_code=202, response_model=Job)
    def prepare_batch(payload: BatchPrepare):
        invoke(database.get_profile, str(payload.profile_id))
        invoke(
            workflow.get_template,
            database,
            str(payload.template_id),
            str(payload.profile_id),
        )

        def create():
            job, _ = database.create_job(
                kind="segmentation.batch_prepare",
                queue_name="io",
                profile_id=str(payload.profile_id),
                payload=payload.model_dump(mode="json"),
                idempotency_key=payload.idempotency_key,
            )
            dispatch_registered_job(database, dispatcher, settings, job)
            return database.get_job(job["id"])

        return invoke(create)

    @router.post("/batches", status_code=202)
    def create_batch(payload: BatchCreate):
        template = invoke(
            workflow.get_template,
            database,
            str(payload.template_id),
            str(payload.profile_id),
        )
        needs_model = any(
            camera["prompts"]
            for camera in template["cameras"]
            if camera["video_key"] in payload.video_keys
        )
        if needs_model and not (
            settings.sam3_checkpoint
            and settings.sam3_checkpoint.is_file()
            and len(settings.sam3_checkpoint_sha256) == 64
        ):
            raise HTTPException(
                503, "SAM 가중치·SHA256와 GPU 작업자를 설정해야 합니다."
            )
        return invoke(workflow.create_batch, database, dispatcher, settings, payload)

    @router.post("/batches/{batch_id}/retry")
    def retry(batch_id: UUID, payload: ProfileRequest):
        invoke(
            workflow.dispatch_batch,
            database,
            dispatcher,
            settings,
            str(batch_id),
            str(payload.profile_id),
        )
        return invoke(
            workflow.get_batch, database, str(batch_id), str(payload.profile_id)
        )

    @router.post("/batches/{batch_id}/items/{item_id}/preview")
    def bind(batch_id: UUID, item_id: UUID, payload: BatchPreviewBind):
        return invoke(
            workflow.bind_preview,
            database,
            settings,
            str(batch_id),
            str(item_id),
            str(payload.profile_id),
            str(payload.preview_id),
        )

    @router.post("/batches/{batch_id}/items/{item_id}/approve")
    def approve(batch_id: UUID, item_id: UUID, payload: BatchApprove):
        return invoke(
            workflow.approve_item,
            database,
            settings,
            str(batch_id),
            str(item_id),
            str(payload.profile_id),
            payload.recipe_hash,
        )

    @router.post("/batches/{batch_id}/items/{item_id}/invalidate")
    def invalidate(batch_id: UUID, item_id: UUID, payload: ProfileRequest):
        return invoke(
            workflow.invalidate_item,
            database,
            str(batch_id),
            str(item_id),
            str(payload.profile_id),
        )

    @router.post("/batches/{batch_id}/exports", status_code=202, response_model=Job)
    def export(batch_id: UUID, payload: BatchExportCreate):
        profile_id = str(payload.profile_id)
        invoke(workflow.get_batch, database, str(batch_id), profile_id)
        existing = database.get_job_for_idempotency(profile_id, payload.idempotency_key)
        if existing:
            expected = {
                "batch_id": str(batch_id),
                "profile_id": profile_id,
                "output_name": payload.output_name,
                "recompute_statistics": payload.recompute_statistics,
            }
            if existing["kind"] != "segmentation.batch_export" or any(
                existing["payload"].get(key) != value for key, value in expected.items()
            ):
                raise HTTPException(
                    409, "같은 요청 키를 다른 작업에 사용할 수 없습니다."
                )
            job = existing
        else:
            preview_ids, approval_hashes = invoke(
                workflow.verify_batch_approvals,
                database,
                settings,
                str(batch_id),
                profile_id,
            )
            output = settings.nas_root / "derived" / payload.output_name
            if output.exists() or output.is_symlink():
                raise HTTPException(
                    409, "같은 이름의 데이터셋이 있습니다. 다른 출력 이름을 입력하세요."
                )
            job, _ = invoke(
                database.create_job,
                kind="segmentation.batch_export",
                queue_name="io",
                profile_id=profile_id,
                idempotency_key=payload.idempotency_key,
                payload={
                    "profile_id": profile_id,
                    "batch_id": str(batch_id),
                    "output_name": payload.output_name,
                    "recompute_statistics": payload.recompute_statistics,
                    "preview_ids": preview_ids,
                    "approval_hashes": approval_hashes,
                },
            )
        try:
            if job["status"] == "queued":
                database.record_dispatch_error(job["id"], "Unable to dispatch job")
            return dispatch_registered_job(database, dispatcher, settings, job)
        except Exception as exc:
            raise HTTPException(
                503, "작업 대기열을 확인하세요. 같은 요청 키로 다시 시도할 수 있습니다."
            ) from exc

    return router
