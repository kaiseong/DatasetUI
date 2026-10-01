from __future__ import annotations

import hashlib
import secrets
import re
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response

from datasetui.config import Settings
from datasetui.content_integrity import (
    dataset_content_fingerprint,
    ContentIntegrityError,
)
from datasetui.database import (
    Database,
    DatasetNotFoundError,
    DatasetNotReadyError,
    JobNotFoundError,
    ProfileNotFoundError,
    IdempotencyConflictError,
    RecipeRevisionMismatchError,
)
from datasetui.queueing import QueueDispatcher
from datasetui.models import Job
from datasetui.segmentation_contract import (
    PreviewApprove,
    PreviewCreate,
    SegmentationExport,
    decode_background,
)
from datasetui.transform_errors import CurationTransformError
from datasetui.segmentation_sample import SampleCreate


def preview_directory(settings: Settings, preview_id: str) -> Path:
    preview_id = str(UUID(preview_id))
    root = settings.jobs_root / "segmentation"
    path = root / preview_id
    if root.is_symlink() or path.is_symlink():
        raise ValueError("미리보기 경로가 올바르지 않습니다.")
    return path


def verified_preview(
    database: Database,
    settings: Settings,
    preview_id: str,
    *,
    verify_source: bool = True,
):
    job = database.get_job(preview_id)
    if job["kind"] != "segmentation.preview" or job["status"] != "succeeded":
        raise ValueError("완료된 미리보기가 필요합니다.")
    result = job["result"]
    path = preview_directory(settings, preview_id)
    if not result or result.get("artifact_fingerprint") != dataset_content_fingerprint(
        path, reuse_file_digests=True
    ):
        raise ValueError("미리보기 파일이 변경되었습니다. 다시 생성하세요.")
    provenance = result.get("model_provenance") or {}
    if (
        provenance.get("engine") == "sam3.1-multiplex"
        and "candidates" not in provenance
    ):
        raise ValueError(
            "객체 선택 이전 버전의 미리보기입니다. 새 미리보기를 생성하세요."
        )
    from datasetui.segmentation import load_source

    if verify_source:
        load_source(
            database,
            settings,
            job["payload"]["spec"]["dataset_id"],
            result["fingerprint"],
        )
    from datasetui.segmentation_pending import effective_preview_job
    return effective_preview_job(job), result


def verify_approval(
    database: Database,
    settings: Settings,
    *,
    preview_id: str,
    profile_id: str,
    approval_token: str,
    verify_source: bool = True,
):
    job, result = verified_preview(
        database, settings, preview_id, verify_source=verify_source
    )
    if result.get("selection_required"):
        raise ValueError("객체 후보 선택이 완료되지 않았습니다.")
    if result.get("review_blocked"):
        raise ValueError(
            "영상 전체에서 대상 영역이 비어 있습니다. 프롬프트를 보정하세요."
        )
    if job["profile_id"] != profile_id:
        raise ValueError("미리보기를 만든 프로필로 승인하세요.")
    with database.connect() as connection:
        approval = connection.execute(
            "SELECT * FROM segmentation_approvals WHERE preview_id = ? AND profile_id = ?",
            (preview_id, profile_id),
        ).fetchone()
    token_hash = hashlib.sha256(approval_token.encode()).hexdigest()
    if (
        approval is None
        or not secrets.compare_digest(approval["token_hash"], token_hash)
        or approval["artifact_fingerprint"] != result["artifact_fingerprint"]
        or approval["recipe_hash"] != result["recipe_hash"]
    ):
        raise ValueError("현재 미리보기를 승인한 뒤 다시 시도하세요.")
    return job, result


def create_segmentation_router(
    database: Database, dispatcher: QueueDispatcher, settings: Settings
) -> APIRouter:
    router = APIRouter(prefix="/api/v1/segmentation")
    with database.connect() as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS segmentation_approvals (
            preview_id TEXT NOT NULL REFERENCES jobs(id),
            profile_id TEXT NOT NULL REFERENCES profiles(id),
            recipe_hash TEXT NOT NULL, artifact_fingerprint TEXT NOT NULL,
            token_hash TEXT NOT NULL, PRIMARY KEY(preview_id, profile_id))""")

    def handle_error(exc: Exception):
        if isinstance(
            exc,
            (
                DatasetNotFoundError,
                JobNotFoundError,
                ProfileNotFoundError,
                FileNotFoundError,
            ),
        ):
            raise HTTPException(404, "데이터셋 또는 작업을 찾을 수 없습니다.") from exc
        if isinstance(
            exc,
            (
                ValueError,
                ContentIntegrityError,
                DatasetNotReadyError,
                RecipeRevisionMismatchError,
                CurationTransformError,
            ),
        ):
            raise HTTPException(409, str(exc)) from exc
        raise exc

    def submit(kind: str, profile_id: str, key: str, payload: dict, queue: str):
        job = None
        try:
            job, _ = database.create_job(
                kind=kind,
                queue_name=queue,
                profile_id=profile_id,
                payload=payload,
                idempotency_key=key,
            )
            if job["status"] == "queued":
                rq_id = job["rq_job_id"] or database.rq_job_id_for(job["id"])
                if job["rq_job_id"] is None:
                    database.mark_enqueued(job["id"], rq_id)
                dispatcher.enqueue(job_id=job["id"], queue_name=queue, rq_job_id=rq_id)
                database.clear_dispatch_error(job["id"])
            return database.get_job(job["id"])
        except IdempotencyConflictError as exc:
            raise HTTPException(
                409, "같은 요청 키를 다른 작업에 사용할 수 없습니다."
            ) from exc
        except ProfileNotFoundError as exc:
            handle_error(exc)
        except Exception as exc:
            if job is not None:
                database.record_dispatch_error(job["id"], "Unable to dispatch job")
            raise HTTPException(
                503, "작업 대기열을 확인하세요. 같은 요청 키로 다시 시도할 수 있습니다."
            ) from exc

    @router.get("/capabilities")
    def capabilities():
        configured = bool(
            settings.sam3_checkpoint
            and settings.sam3_checkpoint.is_file()
            and len(settings.sam3_checkpoint_sha256) == 64
        )
        return {
            "configured": configured,
            "model": "SAM 3.1",
            "message": (
                "모델 경로가 설정되었습니다. GPU 작업자가 가중치와 실행 환경을 확인합니다."
                if configured
                else "SAM 3.1 가중치·SHA256와 GPU 작업자를 설정해야 합니다."
            ),
        }

    @router.get("/datasets/{dataset_id}/catalog")
    def catalog(dataset_id: UUID):
        from datasetui.segmentation_catalog import dataset_catalog
        try:
            return dataset_catalog(database, settings, str(dataset_id))
        except Exception as exc:
            handle_error(exc)

    @router.get("/datasets/{dataset_id}/selection")
    def selection(dataset_id: UUID, episode_index: int = Query(ge=0),
                  video_key: str = Query(min_length=1, max_length=240)):
        from datasetui.segmentation_catalog import selected_scope
        try:
            return selected_scope(database, settings, str(dataset_id), episode_index, video_key)
        except Exception as exc:
            handle_error(exc)

    @router.get("/datasets/{dataset_id}/scope")
    def scope(dataset_id: UUID):
        from datasetui.segmentation import dataset_scope
        from datasetui.segmentation_frames import create_frame_snapshot

        try:
            result = dataset_scope(database, settings, str(dataset_id))
            result["frame_token"] = create_frame_snapshot(
                database, settings, str(dataset_id), result
            )
            return result
        except Exception as exc:
            handle_error(exc)

    @router.get("/datasets/{dataset_id}/frame")
    def frame(
        dataset_id: UUID,
        frame_token: UUID,
        episode_index: int = Query(ge=0),
        video_key: str = Query(min_length=1, max_length=240),
        frame_index: int = Query(ge=0, le=999_999),
    ):
        from datasetui.segmentation_frames import read_snapshot_frame

        try:
            content = read_snapshot_frame(
                database,
                settings,
                str(dataset_id),
                str(frame_token),
                episode_index,
                video_key,
                frame_index,
            )
            return Response(
                content=content,
                media_type="image/png",
                headers={"Cache-Control": "no-store"},
            )
        except Exception as exc:
            handle_error(exc)

    @router.post("/samples", status_code=202, response_model=Job)
    def sample(payload: SampleCreate):
        try:
            database.get_profile(str(payload.profile_id))
            database.get_dataset(str(payload.spec.dataset_id))
        except Exception as exc:
            handle_error(exc)
        if payload.spec.prompts and not capabilities()["configured"]:
            raise HTTPException(503, capabilities()["message"])
        return submit("segmentation.sample", str(payload.profile_id), payload.idempotency_key,
                      {"spec": payload.spec.model_dump(mode="json")},
                      "gpu" if payload.spec.prompts else "io")

    @router.get("/samples/{sample_id}/artifacts/{name}")
    def sample_artifact(sample_id: UUID, name: str, profile_id: UUID):
        from datasetui.segmentation_frames import verified_file_sha256
        try:
            job = database.get_job(str(sample_id))
            if (job["kind"] != "segmentation.sample" or job["status"] != "succeeded"
                    or job["profile_id"] != str(profile_id)):
                raise ValueError("완료된 본인 샘플만 조회할 수 있습니다.")
            result = job["result"] or {}
            if not re.fullmatch(r"(?:original|mask|composite|candidate-[0-9]+-[0-9]+)\.png", name):
                raise ValueError("Invalid sample artifact")
            root = settings.jobs_root / "segmentation-samples"
            path = root / str(sample_id) / name
            if root.is_symlink() or path.parent.is_symlink() or verified_file_sha256(path) != result.get("artifacts", {}).get(name):
                raise ValueError("Sample artifact changed")
            return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})
        except Exception as exc:
            handle_error(exc)

    @router.post("/previews", status_code=202, response_model=Job)
    def preview(payload: PreviewCreate):
        from datasetui.segmentation import load_source

        queue = (
            "io"
            if payload.spec.source_preview_id or not payload.spec.prompts
            else "gpu"
        )

        existing = database.get_job_for_idempotency(
            str(payload.profile_id), payload.idempotency_key
        )
        if existing is not None:
            return submit(
                "segmentation.preview",
                str(payload.profile_id),
                payload.idempotency_key,
                {"spec": payload.spec.model_dump(mode="json")},
                queue,
            )
        if (
            payload.spec.prompts
            and not payload.spec.source_preview_id
            and not capabilities()["configured"]
        ):
            raise HTTPException(503, capabilities()["message"])
        try:
            database.get_profile(str(payload.profile_id))
            if payload.spec.fingerprint:
                load_source(database, settings, str(payload.spec.dataset_id), payload.spec.fingerprint)
            else:
                # Selected-file verification and full hashing run in the worker.
                database.get_dataset(str(payload.spec.dataset_id))
            if payload.spec.render_mode == "image":
                decode_background(payload.spec.background_base64)
            if payload.spec.source_preview_id:
                previous, _ = verified_preview(
                    database, settings, str(payload.spec.source_preview_id)
                )
                if previous["profile_id"] != str(payload.profile_id):
                    raise ValueError("미리보기를 만든 프로필로 마스크를 재사용하세요.")
        except Exception as exc:
            handle_error(exc)
        return submit(
            "segmentation.preview",
            str(payload.profile_id),
            payload.idempotency_key,
            {"spec": payload.spec.model_dump(mode="json")},
            queue,
        )

    @router.get("/previews/{preview_id}")
    def preview_details(preview_id: UUID, profile_id: UUID):
        try:
            # Display only: the artifacts are verified, the source is rechecked
            # by approval-dependent export jobs rather than on every page load.
            job, result = verified_preview(
                database, settings, str(preview_id), verify_source=False
            )
            if job["profile_id"] != str(profile_id):
                raise ValueError("미리보기를 만든 프로필로 조회하세요.")
            return {"spec": job["payload"]["spec"], "result": result}
        except Exception as exc:
            handle_error(exc)

    @router.get("/previews/{preview_id}/artifacts/{name}")
    def artifact(preview_id: UUID, name: str):
        from datasetui.segmentation_frames import verified_file_sha256

        if name not in {
            "original.mp4",
            "composite.mp4",
            "mask.mp4",
        } and not re.fullmatch(r"candidate-[0-9]+-[0-9]+\.png", name):
            raise HTTPException(404, "미리보기 파일을 찾을 수 없습니다.")
        try:
            job = database.get_job(str(preview_id))
            if job["kind"] != "segmentation.preview" or job["status"] != "succeeded":
                raise ValueError("미리보기가 아직 완료되지 않았습니다.")
            path = preview_directory(settings, str(preview_id)) / name
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError(name)
            expected = (job.get("result") or {}).get("artifact_hashes", {}).get(name)
            if not expected or not secrets.compare_digest(
                verified_file_sha256(path), expected
            ):
                raise ValueError(
                    "미리보기 파일이 변경되었습니다. 새 미리보기를 생성하세요."
                )
            return FileResponse(
                path,
                media_type="image/png" if name.endswith(".png") else "video/mp4",
                headers={"Cache-Control": "no-store"},
            )
        except Exception as exc:
            handle_error(exc)

    @router.post("/previews/{preview_id}/approve")
    def approve(preview_id: UUID, payload: PreviewApprove):
        try:
            database.get_profile(str(payload.profile_id))
            # Export re-verifies the complete source before reading it.
            job, result = verified_preview(
                database, settings, str(preview_id), verify_source=False
            )
            if result.get("selection_required"):
                raise ValueError(
                    "텍스트로 찾은 객체 후보를 선택하고 미리보기를 다시 생성하세요."
                )
            if result.get("review_blocked"):
                raise ValueError(
                    "영상 전체에서 대상 영역이 비어 있습니다. 프롬프트를 보정하세요."
                )
            if (
                job["profile_id"] != str(payload.profile_id)
                or result["recipe_hash"] != payload.recipe_hash
            ):
                raise ValueError("현재 프로필과 미리보기 설정을 확인하세요.")
            token = secrets.token_hex(32)
            with database.connect() as connection:
                connection.execute(
                    """INSERT INTO segmentation_approvals
                    VALUES (?, ?, ?, ?, ?) ON CONFLICT(preview_id, profile_id)
                    DO UPDATE SET recipe_hash=excluded.recipe_hash,
                    artifact_fingerprint=excluded.artifact_fingerprint,
                    token_hash=excluded.token_hash""",
                    (
                        str(preview_id),
                        str(payload.profile_id),
                        result["recipe_hash"],
                        result["artifact_fingerprint"],
                        hashlib.sha256(token.encode()).hexdigest(),
                    ),
                )
            return {"approval_token": token}
        except Exception as exc:
            handle_error(exc)

    @router.post("/exports", status_code=202, response_model=Job)
    def export(payload: SegmentationExport):
        existing = database.get_job_for_idempotency(
            str(payload.profile_id), payload.idempotency_key
        )
        if existing is not None:
            return submit(
                "segmentation.export",
                str(payload.profile_id),
                payload.idempotency_key,
                payload.model_dump(mode="json", exclude={"idempotency_key"}),
                "io",
            )
        try:
            selections = payload.previews or [payload]
            identities = set()
            for selected in selections:
                job, result = verify_approval(
                    database,
                    settings,
                    preview_id=str(selected.preview_id),
                    profile_id=str(payload.profile_id),
                    approval_token=selected.approval_token,
                    verify_source=False,
                )
                identities.add(
                    (str(job["payload"]["spec"]["dataset_id"]), result["fingerprint"])
                )
            if len(identities) != 1:
                raise ValueError("같은 원본 데이터셋의 미리보기만 함께 출력하세요.")
            from datasetui.segmentation import load_source

            # One source verification for the whole export, not one per preview.
            load_source(database, settings, *next(iter(identities)))
            path = settings.nas_root / "derived" / payload.output_name
            if path.exists() or path.is_symlink():
                raise ValueError(
                    "같은 이름의 데이터셋이 있습니다. 다른 출력 이름을 입력하세요."
                )
        except Exception as exc:
            handle_error(exc)
        return submit(
            "segmentation.export",
            str(payload.profile_id),
            payload.idempotency_key,
            payload.model_dump(mode="json", exclude={"idempotency_key"}),
            "io",
        )

    return router
