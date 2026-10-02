"""Dataset registry: list, rename, read files, details."""

from __future__ import annotations

import os
from typing import Any, Literal

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse

from datasetui.api.context import RouterContext
from datasetui.database import DatasetNameConflictError, DatasetNotFoundError
from datasetui.dataset_files import (
    DatasetFilePathError,
    DatasetFileUnavailableError,
    DatasetRangeError,
    iter_open_file,
    open_dataset_file,
    parse_byte_range,
)
from datasetui.models import Dataset, DatasetReadiness, DatasetUpdate, StorageArea


class DatasetFileStreamingResponse(StreamingResponse):
    def __init__(self, descriptor: int, *args: Any, **kwargs: Any) -> None:
        self._dataset_file_descriptor = descriptor
        super().__init__(*args, **kwargs)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            os.close(self._dataset_file_descriptor)


def register(router: APIRouter, ctx: RouterContext) -> None:
    database = ctx.database
    settings = ctx.settings

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
