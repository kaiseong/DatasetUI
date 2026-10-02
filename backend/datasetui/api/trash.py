"""Recoverable dataset trash: list, move to trash, restore, empty."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response, status

from datasetui.api.context import RouterContext
from datasetui.database import (
    DatasetNotFoundError,
    DatasetTrashConflictError,
    IdempotencyConflictError,
    ProfileNotFoundError,
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
from datasetui.models import (
    Dataset,
    DatasetTrashCreate,
    DatasetTrashEmptyCreate,
    DatasetTrashEntry,
    DatasetTrashRestore,
    Job,
)


def register(router: APIRouter, ctx: RouterContext) -> None:
    database = ctx.database
    settings = ctx.settings
    dispatch_library_job = ctx.dispatch_library_job

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
