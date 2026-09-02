from __future__ import annotations

from collections.abc import Callable
from typing import Any

from datasetui.config import Settings
from datasetui.database import Database
from datasetui.datasets import scan_storage_area


JobHandler = Callable[[dict[str, Any]], dict[str, Any]]


def _phase2_smoke(payload: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True}


def _validate_empty_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload:
        raise ValueError("phase2.smoke does not accept payload fields")
    return {}


def _validate_dataset_scan_payload(payload: dict[str, Any]) -> dict[str, Any]:
    unexpected = set(payload) - {"storage_areas"}
    if unexpected:
        raise ValueError("datasets.scan accepts only the storage_areas field")
    areas = payload.get("storage_areas", ["raw", "derived"])
    if not isinstance(areas, list) or not areas:
        raise ValueError("storage_areas must be a non-empty list")
    if any(area not in {"raw", "derived"} for area in areas):
        raise ValueError("storage_areas may contain only raw or derived")
    if len(areas) != len(set(areas)):
        raise ValueError("storage_areas cannot contain duplicates")
    return {"storage_areas": areas}


def _scan_datasets(payload: dict[str, Any]) -> dict[str, Any]:
    settings = Settings.from_env()
    database = Database(settings.database_path)
    database.initialize()
    summaries: dict[str, dict[str, int]] = {}
    for area in payload["storage_areas"]:
        scan_generation = database.begin_dataset_scan(area)
        records = scan_storage_area(
            settings.nas_root,
            area,
            max_depth=settings.dataset_scan_max_depth,
        )
        summaries[area] = database.synchronize_datasets(
            storage_area=area,
            records=records,
            scan_generation=scan_generation,
        )
    return {"storage_areas": summaries}


JOB_HANDLERS: dict[str, tuple[str, JobHandler]] = {
    "phase2.smoke": ("cpu", _phase2_smoke),
    "datasets.scan": ("io", _scan_datasets),
}


def queue_for_kind(kind: str) -> str | None:
    registered = JOB_HANDLERS.get(kind)
    return registered[0] if registered else None


def validate_job_payload(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    if kind == "phase2.smoke":
        return _validate_empty_payload(payload)
    if kind == "datasets.scan":
        return _validate_dataset_scan_payload(payload)
    raise ValueError(f"Unsupported job kind: {kind}")


def run_registered_job(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    registered = JOB_HANDLERS.get(kind)
    if registered is None:
        raise ValueError(f"Unsupported job kind: {kind}")
    return registered[1](payload)
