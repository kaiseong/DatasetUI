from __future__ import annotations

from collections.abc import Sequence
from contextlib import asynccontextmanager
import asyncio
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from datasetui.api import create_router
from datasetui.config import Settings
from datasetui.database import Database
from datasetui.queueing import QueueDispatcher
from datasetui.segmentation.api import create_segmentation_router
from datasetui.segmentation.workflow_api import create_segmentation_workflow_router


def create_application(
    *,
    database: Database,
    dispatcher: QueueDispatcher,
    allowed_origins: Sequence[str],
    legacy_app: FastAPI,
    settings: Settings | None = None,
) -> FastAPI:
    configured = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(_application):
        from datasetui.delivery_workflow import recover_workflows
        from datasetui.segmentation.workflows import recover_segmentation_batches

        async def reconcile():
            while True:
                try:
                    await asyncio.to_thread(
                        recover_workflows, database, dispatcher, configured
                    )
                except Exception:
                    logging.getLogger(__name__).warning(
                        "workflow reconciliation delayed"
                    )
                try:
                    await asyncio.to_thread(
                        recover_segmentation_batches, database, dispatcher, configured
                    )
                except Exception:
                    logging.getLogger(__name__).warning("segmentation reconciliation delayed")
                await asyncio.sleep(5)

        task = asyncio.create_task(reconcile())
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    application = FastAPI(title="DatasetUI backend", lifespan=lifespan)

    @application.middleware("http")
    async def protect_workbench_origin(request: Request, call_next):
        if request.url.path.startswith("/api/v1/"):
            origin = request.headers.get("origin")
            if origin and origin not in allowed_origins:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Origin is not allowed"},
                )
        return await call_next(request)

    application.state.datasetui_database = database
    application.state.datasetui_dispatcher = dispatcher
    application.include_router(create_router(database, dispatcher, settings=settings))
    application.include_router(
        create_segmentation_router(
            database, dispatcher, settings or Settings.from_env()
        )
    )
    application.include_router(
        create_segmentation_workflow_router(database, dispatcher, configured)
    )
    application.mount("", legacy_app)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(allowed_origins),
        allow_methods=["GET", "HEAD", "POST", "PATCH"],
        allow_headers=["Content-Type", "Range"],
    )
    return application
