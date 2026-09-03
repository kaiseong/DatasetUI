from __future__ import annotations

from collections.abc import Callable
from typing import Any
import re

from datasetui.config import Settings
from datasetui.database import Database
from datasetui.datasets import scan_storage_area


JobHandler = Callable[[dict[str, Any]], dict[str, Any]]
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
OUTPUT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$")


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


def _validate_curation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != {"snapshot_id", "output_name"}:
        raise ValueError("invalid internal curation payload")
    snapshot_id = payload["snapshot_id"]
    output_name = payload["output_name"]
    if not isinstance(snapshot_id, str) or not UUID_PATTERN.fullmatch(snapshot_id):
        raise ValueError("invalid curation snapshot")
    if (
        not isinstance(output_name, str)
        or not OUTPUT_NAME_PATTERN.fullmatch(output_name)
        or ".." in output_name
    ):
        raise ValueError("invalid curation output name")
    return {"snapshot_id": snapshot_id, "output_name": output_name}


def _validate_merge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != {"sources", "output_name", "robot_type"}:
        raise ValueError("invalid internal merge payload")
    sources = payload["sources"]
    if not isinstance(sources, list) or not 2 <= len(sources) <= 50:
        raise ValueError("merge requires 2 to 50 sources")
    normalized = []
    for source in sources:
        if not isinstance(source, dict) or set(source) != {"id", "fingerprint"}:
            raise ValueError("invalid internal merge source")
        if not isinstance(source["id"], str) or not UUID_PATTERN.fullmatch(source["id"]):
            raise ValueError("invalid internal merge source")
        fingerprint = source["fingerprint"]
        if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise ValueError("invalid internal merge fingerprint")
        normalized.append({"id": source["id"], "fingerprint": fingerprint})
    output_name = payload["output_name"]
    robot_type = payload["robot_type"]
    if not isinstance(output_name, str) or not OUTPUT_NAME_PATTERN.fullmatch(output_name) or ".." in output_name:
        raise ValueError("invalid merge output name")
    if not isinstance(robot_type, str) or not robot_type.strip() or len(robot_type) > 120:
        raise ValueError("invalid merge robot type")
    return {"sources": normalized, "output_name": output_name, "robot_type": robot_type.strip()}


def _validate_dataset_job_payload(
    payload: dict[str, Any], *, conversion: bool = False
) -> dict[str, Any]:
    expected = {"dataset_id", "fingerprint", "storage_area", "relative_path"}
    if conversion:
        expected.add("output_name")
    else:
        expected.add("mode")
    if set(payload) != expected:
        raise ValueError("invalid internal dataset job payload")
    if not isinstance(payload["dataset_id"], str) or not UUID_PATTERN.fullmatch(payload["dataset_id"]):
        raise ValueError("invalid internal dataset ID")
    if not isinstance(payload["fingerprint"], str) or not re.fullmatch(r"[0-9a-f]{64}", payload["fingerprint"]):
        raise ValueError("invalid internal dataset fingerprint")
    if payload["storage_area"] not in {"raw", "derived"}:
        raise ValueError("invalid internal storage area")
    relative = payload["relative_path"]
    if not isinstance(relative, str) or relative.startswith("/") or any(part in {"", ".", ".."} for part in relative.split("/")):
        raise ValueError("invalid internal dataset path")
    if conversion:
        output = payload["output_name"]
        if not isinstance(output, str) or not OUTPUT_NAME_PATTERN.fullmatch(output) or ".." in output:
            raise ValueError("invalid conversion output name")
    elif payload["mode"] not in {"quick", "full", "export_gate"}:
        raise ValueError("invalid validation mode")
    return dict(payload)


def _validate_delivery_payload(payload: dict[str, Any], kind: str) -> dict[str, Any]:
    base = {"dataset_id", "fingerprint", "storage_area", "relative_path"}
    extras = {
        "datasets.export_nas": {"output_name"},
        "datasets.upload_hf": {"repo_name", "visibility"},
        "datasets.copy_pc_key": {"host", "port", "username", "destination"},
    }[kind]
    if set(payload) != base | extras:
        raise ValueError("invalid internal delivery payload")
    core = _validate_dataset_job_payload(
        {**{key: payload[key] for key in base}, "mode": "quick"}
    )
    core.pop("mode")
    result = {**core}
    if kind == "datasets.export_nas":
        name = payload["output_name"]
        if not isinstance(name, str) or not OUTPUT_NAME_PATTERN.fullmatch(name) or ".." in name:
            raise ValueError("invalid NAS export name")
        result["output_name"] = name
    elif kind == "datasets.upload_hf":
        name = payload["repo_name"]
        if not isinstance(name, str) or not OUTPUT_NAME_PATTERN.fullmatch(name) or ".." in name:
            raise ValueError("invalid Hugging Face repository name")
        if payload["visibility"] not in {"private", "public"}:
            raise ValueError("invalid Hugging Face visibility")
        result.update(repo_name=name, visibility=payload["visibility"])
    else:
        if not isinstance(payload["host"], str) or len(payload["host"]) > 64:
            raise ValueError("invalid PC host")
        if isinstance(payload["port"], bool) or not isinstance(payload["port"], int) or not 1 <= payload["port"] <= 65535:
            raise ValueError("invalid PC port")
        if not isinstance(payload["username"], str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", payload["username"]):
            raise ValueError("invalid PC username")
        if not isinstance(payload["destination"], str) or not payload["destination"].startswith("~/"):
            raise ValueError("invalid PC destination")
        result.update(
            host=payload["host"],
            port=payload["port"],
            username=payload["username"],
            destination=payload["destination"],
        )
    return result


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
    if kind == "curation.materialize":
        return _validate_curation_payload(payload)
    if kind == "datasets.merge":
        return _validate_merge_payload(payload)
    if kind == "datasets.validate":
        return _validate_dataset_job_payload(payload)
    if kind == "datasets.convert_v21":
        return _validate_dataset_job_payload(payload, conversion=True)
    if kind in {"datasets.export_nas", "datasets.upload_hf", "datasets.copy_pc_key"}:
        return _validate_delivery_payload(payload, kind)
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
    if kind == "curation.materialize":
        from datasetui.transforms import materialize_curation_recipe

        if job_id is None or worker_id is None:
            raise ValueError("curation materialization requires worker ownership")
        settings = Settings.from_env()
        database = Database(settings.database_path)
        database.initialize()
        return materialize_curation_recipe(
            database=database,
            settings=settings,
            payload=_validate_curation_payload(payload),
            job_id=job_id,
            worker_id=worker_id,
        )
    if kind == "datasets.merge":
        from datasetui.merge import merge_datasets

        if job_id is None or worker_id is None:
            raise ValueError("dataset merge requires worker ownership")
        settings = Settings.from_env()
        database = Database(settings.database_path)
        database.initialize()
        return merge_datasets(
            database=database,
            settings=settings,
            payload=_validate_merge_payload(payload),
            job_id=job_id,
            worker_id=worker_id,
        )
    if kind == "datasets.validate":
        from datasetui.validation import validate_registered_dataset

        settings = Settings.from_env()
        database = Database(settings.database_path)
        database.initialize()
        return validate_registered_dataset(
            database=database,
            settings=settings,
            payload=_validate_dataset_job_payload(payload),
        )
    if kind == "datasets.convert_v21":
        from datasetui.conversion import convert_dataset_to_v21

        if job_id is None or worker_id is None:
            raise ValueError("dataset conversion requires worker ownership")
        settings = Settings.from_env()
        database = Database(settings.database_path)
        database.initialize()
        return convert_dataset_to_v21(
            database=database,
            settings=settings,
            payload=_validate_dataset_job_payload(payload, conversion=True),
            job_id=job_id,
            worker_id=worker_id,
        )
    if kind in {"datasets.export_nas", "datasets.upload_hf", "datasets.copy_pc_key"}:
        from datasetui.delivery import (
            copy_to_pc_with_key,
            export_to_nas,
            upload_to_huggingface,
        )

        if job_id is None or worker_id is None:
            raise ValueError("delivery requires worker ownership")
        settings = Settings.from_env()
        database = Database(settings.database_path)
        database.initialize()
        safe_payload = _validate_delivery_payload(payload, kind)
        if kind == "datasets.export_nas":
            return export_to_nas(
                database=database,
                settings=settings,
                payload=safe_payload,
                job_id=job_id,
                worker_id=worker_id,
            )
        if kind == "datasets.upload_hf":
            return upload_to_huggingface(
                database=database,
                settings=settings,
                payload=safe_payload,
                job_id=job_id,
                worker_id=worker_id,
            )
        return copy_to_pc_with_key(
            database=database, settings=settings, payload=safe_payload
        )
    registered = JOB_HANDLERS.get(kind)
    if registered is None:
        raise ValueError(f"Unsupported job kind: {kind}")
    return registered[1](payload)
