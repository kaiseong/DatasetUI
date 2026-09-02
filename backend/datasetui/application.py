from __future__ import annotations

from collections.abc import Sequence

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from datasetui.api import create_router
from datasetui.database import Database
from datasetui.queueing import QueueDispatcher


def create_application(
    *,
    database: Database,
    dispatcher: QueueDispatcher,
    allowed_origins: Sequence[str],
    legacy_app: FastAPI,
) -> FastAPI:
    application = FastAPI(title="DatasetUI backend")

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
    application.include_router(create_router(database, dispatcher))
    application.mount("", legacy_app)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(allowed_origins),
        allow_methods=["GET", "POST", "PATCH"],
        allow_headers=["Content-Type"],
    )
    return application
