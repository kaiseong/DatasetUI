"""System health, host resources and delivery capabilities."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response, status

from datasetui.api.context import RouterContext
from datasetui.huggingface import HF_NAMESPACE
from datasetui.models import SystemHealth


def register(router: APIRouter, ctx: RouterContext) -> None:
    database = ctx.database
    dispatcher = ctx.dispatcher
    settings = ctx.settings

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
