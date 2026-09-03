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


def _validate_hf_import_payload(payload: dict[str, Any]) -> dict[str, Any]:
    from datasetui.huggingface import (
        HF_NAMESPACE,
        validate_commit_sha,
        validate_dataset_name,
        validate_revision,
    )

    allowed = {
        "repo_id",
        "dataset_name",
        "requested_revision",
        "commit_sha",
        "generation",
        "expected_file_count",
        "expected_total_bytes",
    }
    if set(payload) != allowed:
        raise ValueError("invalid internal Hugging Face import payload")
    dataset_name = validate_dataset_name(payload["dataset_name"])
    if payload["repo_id"] != f"{HF_NAMESPACE}/{dataset_name}":
        raise ValueError("invalid Hugging Face repository namespace")
    requested_revision = validate_revision(payload["requested_revision"])
    commit_sha = validate_commit_sha(payload["commit_sha"])
    generation = payload["generation"]
    file_count = payload["expected_file_count"]
    total_bytes = payload["expected_total_bytes"]
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 1
    ):
        raise ValueError("invalid import generation")
    if (
        isinstance(file_count, bool)
        or not isinstance(file_count, int)
        or file_count < 0
    ):
        raise ValueError("invalid expected file count")
    if total_bytes is not None and (
        isinstance(total_bytes, bool)
        or not isinstance(total_bytes, int)
        or total_bytes < 0
    ):
        raise ValueError("invalid expected byte count")
    return {
        "repo_id": payload["repo_id"],
        "dataset_name": dataset_name,
        "requested_revision": requested_revision,
        "commit_sha": commit_sha,
        "generation": generation,
        "expected_file_count": file_count,
        "expected_total_bytes": total_bytes,
    }


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


def run_registered_job(
    kind: str,
    payload: dict[str, Any],
    *,
    job_id: str | None = None,
    worker_id: str | None = None,
) -> dict[str, Any]:
    if kind == "hf.import":
        from datasetui.huggingface import import_huggingface_dataset

        if job_id is None or worker_id is None:
            raise ValueError("Hugging Face imports require worker ownership")
        settings = Settings.from_env()
        database = Database(settings.database_path)
        database.initialize()
        return import_huggingface_dataset(
            database=database,
            settings=settings,
            payload=_validate_hf_import_payload(payload),
            job_id=job_id,
            worker_id=worker_id,
        )
    registered = JOB_HANDLERS.get(kind)
    if registered is None:
        raise ValueError(f"Unsupported job kind: {kind}")
    return registered[1](payload)
