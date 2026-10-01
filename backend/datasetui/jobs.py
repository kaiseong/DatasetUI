from __future__ import annotations

from collections.abc import Callable
from typing import Any
import re

from datasetui.config import Settings
from datasetui.database import Database
from datasetui.datasets import scan_storage_area
from datasetui.job_progress import JobProgressReporter


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


def _scan_datasets(
    payload: dict[str, Any],
    *,
    job_id: str | None = None,
    worker_id: str | None = None,
) -> dict[str, Any]:
    settings = Settings.from_env()
    database = Database(settings.database_path)
    database.initialize()
    progress = (
        JobProgressReporter(database, job_id=job_id, worker_id=worker_id)
        if job_id is not None and worker_id is not None
        else None
    )
    summaries: dict[str, dict[str, int]] = {}
    areas = payload["storage_areas"]
    for area_index, area in enumerate(areas):
        if progress is not None:
            progress(
                {
                    "stage": "scan",
                    "completed": area_index,
                    "total": len(areas),
                    "unit": "items",
                    "current_item": f"{area} 저장 영역 탐색 중",
                    "_force": True,
                }
            )
        scan_generation = database.begin_dataset_scan(area)
        records = scan_storage_area(
            settings.nas_root,
            area,
            max_depth=settings.dataset_scan_max_depth,
            on_discovered=(
                lambda count, area=area: progress(
                    {
                        "stage": "scan",
                        "completed": count,
                        "total": 0,
                        "unit": "items",
                        "current_item": f"{area}: 데이터셋 {count}개 발견",
                    }
                )
                if progress is not None
                else None
            ),
        )
        summaries[area] = database.synchronize_datasets(
            storage_area=area,
            records=records,
            scan_generation=scan_generation,
        )
        if progress is not None:
            progress(
                {
                    "stage": "scan",
                    "completed": area_index + 1,
                    "total": len(areas),
                    "unit": "items",
                    "current_item": f"{area}: 데이터셋 {len(records)}개 발견",
                    "_force": True,
                }
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
        if not isinstance(source["id"], str) or not UUID_PATTERN.fullmatch(
            source["id"]
        ):
            raise ValueError("invalid internal merge source")
        fingerprint = source["fingerprint"]
        if not isinstance(fingerprint, str) or not re.fullmatch(
            r"[0-9a-f]{64}", fingerprint
        ):
            raise ValueError("invalid internal merge fingerprint")
        normalized.append({"id": source["id"], "fingerprint": fingerprint})
    output_name = payload["output_name"]
    robot_type = payload["robot_type"]
    if (
        not isinstance(output_name, str)
        or not OUTPUT_NAME_PATTERN.fullmatch(output_name)
        or ".." in output_name
    ):
        raise ValueError("invalid merge output name")
    if (
        not isinstance(robot_type, str)
        or not robot_type.strip()
        or len(robot_type) > 120
    ):
        raise ValueError("invalid merge robot type")
    return {
        "sources": normalized,
        "output_name": output_name,
        "robot_type": robot_type.strip(),
    }


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
    if not isinstance(payload["dataset_id"], str) or not UUID_PATTERN.fullmatch(
        payload["dataset_id"]
    ):
        raise ValueError("invalid internal dataset ID")
    if not isinstance(payload["fingerprint"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", payload["fingerprint"]
    ):
        raise ValueError("invalid internal dataset fingerprint")
    if payload["storage_area"] not in {"raw", "derived"}:
        raise ValueError("invalid internal storage area")
    relative = payload["relative_path"]
    if (
        not isinstance(relative, str)
        or relative.startswith("/")
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise ValueError("invalid internal dataset path")
    if conversion:
        output = payload["output_name"]
        if (
            not isinstance(output, str)
            or not OUTPUT_NAME_PATTERN.fullmatch(output)
            or ".." in output
        ):
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
        "datasets.copy_pc_password": {"host", "port", "username", "destination"},
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
        if (
            not isinstance(name, str)
            or not OUTPUT_NAME_PATTERN.fullmatch(name)
            or ".." in name
        ):
            raise ValueError("invalid NAS export name")
        result["output_name"] = name
    elif kind == "datasets.upload_hf":
        from datasetui.huggingface import validate_dataset_name

        name = payload["repo_name"]
        if not isinstance(name, str):
            raise ValueError("invalid Hugging Face repository name")
        try:
            name = validate_dataset_name(name)
        except ValueError as exc:
            raise ValueError("invalid Hugging Face repository name") from exc
        if payload["visibility"] not in {"private", "public"}:
            raise ValueError("invalid Hugging Face visibility")
        result.update(repo_name=name, visibility=payload["visibility"])
    else:
        if not isinstance(payload["host"], str) or len(payload["host"]) > 64:
            raise ValueError("invalid PC host")
        if (
            isinstance(payload["port"], bool)
            or not isinstance(payload["port"], int)
            or not 1 <= payload["port"] <= 65535
        ):
            raise ValueError("invalid PC port")
        if not isinstance(payload["username"], str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_.-]*", payload["username"]
        ):
            raise ValueError("invalid PC username")
        if not isinstance(payload["destination"], str) or not payload[
            "destination"
        ].startswith("~/"):
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
    if kind == "datasets.delivery_preflight":
        return "cpu"
    if kind in {"hf.delete", "datasets.empty_trash", "datasets.copy_pc_password"}:
        return "io"
    registered = JOB_HANDLERS.get(kind)
    return registered[0] if registered else None


def validate_job_payload(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    if kind == "segmentation.batch_prepare":
        from datasetui.segmentation.workflow_contract import BatchPrepare
        return BatchPrepare.model_validate(payload).model_dump(mode="json")
    if kind == "segmentation.batch_export":
        from datasetui.segmentation.workflow_contract import BatchExportPayload

        return BatchExportPayload.model_validate(payload).model_dump(mode="json")
    if kind == "segmentation.sample":
        from datasetui.segmentation.sample import SampleSpec
        if set(payload) != {"spec"}:
            raise ValueError("invalid sample payload")
        return {"spec": SampleSpec.model_validate(payload["spec"]).model_dump(mode="json")}
    if kind == "segmentation.preview":
        from datasetui.segmentation.contract import SegmentationSpec, PendingPreviewSpec

        if set(payload) != {"spec"}:
            raise ValueError("invalid segmentation preview payload")
        return {
            "spec": (SegmentationSpec if payload["spec"].get("fingerprint") else PendingPreviewSpec).model_validate(payload["spec"]).model_dump(
                mode="json"
            )
        }
    if kind == "segmentation.export":
        from datasetui.segmentation.contract import SegmentationExport

        return SegmentationExport.model_validate(
            {**payload, "idempotency_key": "internal"}
        ).model_dump(mode="json", exclude={"idempotency_key"})
    if kind == "phase2.smoke":
        return _validate_empty_payload(payload)
    if kind == "datasets.scan":
        return _validate_dataset_scan_payload(payload)
    if kind == "curation.materialize":
        return _validate_curation_payload(payload)
    if kind == "datasets.merge":
        return _validate_merge_payload(payload)
    if kind in {"datasets.validate", "datasets.delivery_preflight"}:
        return _validate_dataset_job_payload(payload)
    if kind == "datasets.convert_v21":
        return _validate_dataset_job_payload(payload, conversion=True)
    if kind in {
        "datasets.export_nas",
        "datasets.upload_hf",
        "datasets.copy_pc_key",
        "datasets.copy_pc_password",
    }:
        return _validate_delivery_payload(payload, kind)
    if kind in {"hf.delete", "datasets.empty_trash"}:
        from datasetui.library_operations import validate_library_operation

        return validate_library_operation(kind, payload)
    raise ValueError(f"Unsupported job kind: {kind}")


def run_registered_job(
    kind: str,
    payload: dict[str, Any],
    *,
    job_id: str | None = None,
    worker_id: str | None = None,
) -> dict[str, Any]:
    if kind in {
        "datasets.delivery_preflight",
        "hf.delete",
        "datasets.empty_trash",
        "datasets.copy_pc_password",
    }:
        if job_id is None or worker_id is None:
            raise ValueError("operation requires worker ownership")
        settings = Settings.from_env()
        database = Database(settings.database_path)
        database.initialize()
        validated = validate_job_payload(kind, payload)
        if kind == "datasets.delivery_preflight":
            from datasetui.delivery_workflow import run_delivery_preflight

            return run_delivery_preflight(
                database, settings, validated, job_id, worker_id
            )
        if kind == "datasets.copy_pc_password":
            from datasetui.delivery import _source, _copy_verified_to_pc
            from datasetui.credential_store import credential_store

            record, source, manifest = _source(database, settings, validated)
            password = credential_store(settings).take(job_id)
            try:
                return _copy_verified_to_pc(
                    expected_manifest=manifest,
                    settings=settings,
                    source=source,
                    record=record,
                    host=validated["host"],
                    port=validated["port"],
                    username=validated["username"],
                    destination=validated["destination"],
                    password=password,
                    expected_fingerprint=manifest["tree_sha256"],
                    lease_check=lambda: database.assert_job_lease(
                        job_id, worker_id=worker_id
                    ),
                    finalize=lambda: database.begin_job_finalization(
                        job_id, worker_id=worker_id
                    ),
                    on_progress=JobProgressReporter(
                        database, job_id=job_id, worker_id=worker_id
                    ),
                )
            finally:
                password = None
                try:
                    credential_store(settings).delete(job_id)
                except Exception:
                    # GETDEL already consumed the credential. A cleanup outage
                    # must not turn a confirmed remote publication into failure.
                    import logging

                    logging.getLogger(__name__).warning(
                        "credential cleanup delayed for job %s", job_id
                    )
        from datasetui.library_operations import run_library_operation

        return run_library_operation(
            kind, database, settings, validated, job_id, worker_id
        )
    if kind == "segmentation.batch_prepare":
        from datasetui.segmentation.workflow_contract import BatchPrepare, BatchCreate
        from datasetui.segmentation.catalog import dataset_catalog
        from datasetui.segmentation.workflows import create_batch
        from datasetui.segmentation.source import load_source
        from datasetui.queueing import RQDispatcher
        if job_id is None or worker_id is None:
            raise ValueError("batch preparation requires a worker lease")
        settings = Settings.from_env()
        database = Database(settings.database_path)
        database.initialize()
        parsed = BatchPrepare.model_validate(payload)
        database.assert_job_lease(job_id, worker_id=worker_id)
        catalog = dataset_catalog(database, settings, str(parsed.dataset_id))
        if catalog["metadata_revision"] != parsed.metadata_revision:
            raise ValueError("Dataset metadata changed")
        _, source = load_source(database, settings, str(parsed.dataset_id))
        request = parsed.model_dump(mode="json", exclude={"metadata_revision"})
        request["fingerprint"] = source.segmentation_fingerprint
        database.assert_job_lease(job_id, worker_id=worker_id)
        dispatcher = RQDispatcher(settings.redis_url, settings.job_timeout_seconds, settings.io_job_timeout_seconds)
        batch = create_batch(database, dispatcher, settings, BatchCreate.model_validate(request))
        return {"batch_id": batch["id"]}
    if kind == "segmentation.sample":
        from datasetui.segmentation.sample import create_sample
        if job_id is None or worker_id is None:
            raise ValueError("sample jobs require a worker lease")
        settings = Settings.from_env()
        database = Database(settings.database_path)
        database.initialize()
        validated = validate_job_payload(kind, payload)
        return create_sample(database, settings, job_id=job_id, worker_id=worker_id,
                             spec=validated["spec"])
    if kind in {"segmentation.preview", "segmentation.export"}:
        from datasetui.segmentation.preview import create_preview
        from datasetui.segmentation.export import export_preview
        from datasetui.segmentation.preview import verify_approval

        if job_id is None or worker_id is None:
            raise ValueError("segmentation jobs require a worker lease")
        payload = validate_job_payload(kind, payload)
        settings = Settings.from_env()
        database = Database(settings.database_path)
        database.initialize()
        if kind == "segmentation.preview":
            return create_preview(
                database,
                settings,
                job_id=job_id,
                worker_id=worker_id,
                spec=payload["spec"],
            )
        previews = payload.get("previews") or [{
            "preview_id": payload["preview_id"],
            "approval_token": payload["approval_token"],
        }]
        for preview in previews:
            verify_approval(
                database, settings, preview_id=preview["preview_id"],
                profile_id=payload["profile_id"], approval_token=preview["approval_token"],
                # export_preview verifies the complete source once, below.
                verify_source=False,
            )
        return export_preview(
            database, settings, job_id=job_id, worker_id=worker_id,
            previews=previews, output_name=payload["output_name"],
            recompute_statistics=payload.get("recompute_statistics", False),
        )
    if kind == "segmentation.batch_export":
        from datasetui.segmentation.export import export_preview
        from datasetui.segmentation.workflows import verify_batch_export

        if job_id is None or worker_id is None:
            raise ValueError("segmentation jobs require a worker lease")
        payload = validate_job_payload(kind, payload)
        settings = Settings.from_env()
        database = Database(settings.database_path)
        database.initialize()
        verify_batch_export(database, settings, payload)
        return export_preview(
            database, settings, job_id=job_id, worker_id=worker_id,
            preview_ids=payload["preview_ids"], output_name=payload["output_name"],
            recompute_statistics=payload.get("recompute_statistics", False),
        )
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
        job_progress = (
            JobProgressReporter(database, job_id=job_id, worker_id=worker_id)
            if job_id is not None and worker_id is not None
            else None
        )

        def report_validation(progress: dict[str, Any]) -> None:
            if job_id is None or worker_id is None:
                return
            database.update_validation_progress(
                job_id, worker_id=worker_id, progress=progress
            )
            if job_progress is not None:
                stage = str(progress.get("stage", "validate"))
                job_progress(
                    {
                        "stage": {
                            "metadata": "validate",
                            "data": "validate",
                        }.get(stage, stage),
                        "completed": progress.get("completed", 0),
                        "total": progress.get("total", 0),
                        "unit": "episodes",
                        "_force": stage == "complete",
                    }
                )

        return validate_registered_dataset(
            database=database,
            settings=settings,
            payload=_validate_dataset_job_payload(payload),
            on_progress=report_validation
            if job_id is not None and worker_id is not None
            else None,
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
            database=database,
            settings=settings,
            payload=safe_payload,
            job_id=job_id,
            worker_id=worker_id,
        )
    registered = JOB_HANDLERS.get(kind)
    if registered is None:
        raise ValueError(f"Unsupported job kind: {kind}")
    if kind == "datasets.scan":
        return _scan_datasets(payload, job_id=job_id, worker_id=worker_id)
    return registered[1](payload)
