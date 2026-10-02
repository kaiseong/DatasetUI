"""Hugging Face discovery, revisions, imports and deletion."""

from __future__ import annotations

import json
import logging
import os
import stat as stat_module
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response, status

from datasetui.api.context import RouterContext
from datasetui.database import (
    DatasetTrashConflictError,
    IdempotencyConflictError,
    ProfileNotFoundError,
)
from datasetui.hf_errors import (
    HuggingFaceDatasetNotFoundError,
    HuggingFaceRevisionNotFoundError,
    HuggingFaceUnavailableError,
)
from datasetui.huggingface import HF_NAMESPACE
from datasetui.models import (
    HuggingFaceDataset,
    HuggingFaceDeleteCreate,
    HuggingFaceImportCreate,
    HuggingFaceRevision,
    Job,
)

logger = logging.getLogger("datasetui.api")


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


def register(router: APIRouter, ctx: RouterContext) -> None:
    database = ctx.database
    settings = ctx.settings
    hf_gateway = ctx.hf_gateway
    dispatch_job = ctx.dispatch_job
    dispatch_library_job = ctx.dispatch_library_job
    recover_expired_jobs = ctx.recover_expired_jobs

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
