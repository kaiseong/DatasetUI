from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from datasetui.config import Settings
from datasetui.curation.writer import write_dataset
from datasetui.database import Database, RecipeRevisionMismatchError
from datasetui.dataset_io.files import (
    read_regular_bytes,
    safe_dataset_root,
    write_json_atomic,
)
from datasetui.dataset_io.publish import (
    assert_source_unchanged,
    publish_output,
    source_tree_identity,
    tree_manifest,
)
from datasetui.dataset_io.source import DatasetSource
from datasetui.datasets import inspect_dataset, MAX_INFO_BYTES, scan_storage_area
from datasetui.job_progress import JobProgressReporter, report_progress
from datasetui.transform_errors import CurationTransformError
from datasetui.validation import validate_dataset_root




CONVERSION_ENGINE_ID = "datasetui-v3-to-v21-source-codec-exact-stats-v2"
VIDEO_CODEC_POLICY = "source"


def _assert_reusable_manifest(
    result: dict[str, Any], *, record: dict[str, Any], payload: dict[str, Any]
) -> None:
    if result.get("conversion_engine") != CONVERSION_ENGINE_ID:
        raise CurationTransformError(
            "Conversion manifest was produced by a legacy conversion engine; "
            "the existing output was left unchanged"
        )
    if result.get("video_codec_policy") != VIDEO_CODEC_POLICY:
        raise CurationTransformError(
            "Conversion manifest was produced with a legacy video codec policy; "
            "the existing output was left unchanged"
        )
    output = result.get("output")
    if (
        result.get("source_dataset_id") != record["id"]
        or result.get("source_fingerprint") != record["fingerprint"]
        or not isinstance(output, dict)
        or output.get("name") != payload["output_name"]
        or output.get("relative_path") != payload["output_name"]
    ):
        raise CurationTransformError(
            "Conversion manifest conflicts with the requested conversion"
        )


def convert_dataset_to_v21(
    *,
    database: Database,
    settings: Settings,
    payload: dict[str, Any],
    job_id: str,
    worker_id: str,
) -> dict[str, Any]:
    progress = JobProgressReporter(
        database,
        job_id=job_id,
        worker_id=worker_id,
        output_name=payload["output_name"],
    )
    report_progress(
        progress,
        stage="preparing",
        completed=0,
        total=1,
        unit="items",
        current_item="원본 데이터셋 확인",
        force=True,
    )
    record = database.get_dataset(payload["dataset_id"])
    if (
        not record["available"]
        or record["readiness"] != "ready"
        or record["fingerprint"] != payload["fingerprint"]
        or record["storage_area"] != payload["storage_area"]
        or record["relative_path"] != payload["relative_path"]
    ):
        raise RecipeRevisionMismatchError(payload["dataset_id"])
    source_root = safe_dataset_root(
        settings.nas_root, record["storage_area"], record["relative_path"]
    )
    source_identity = source_tree_identity(source_root)
    raw = read_regular_bytes(source_root / "meta/info.json", max_bytes=MAX_INFO_BYTES)
    if hashlib.sha256(raw).hexdigest() != payload["fingerprint"]:
        raise RecipeRevisionMismatchError(payload["dataset_id"])
    from datasetui.relative_artifacts import reject_relative_profile

    reject_relative_profile(source_root, operation="v2.1 conversion")
    info = json.loads(raw)
    if info.get("codebase_version") != "v3.0":
        raise CurationTransformError("Only v3.0 datasets can be converted to v2.1")
    language_features = {
        name
        for name, feature in info.get("features", {}).items()
        if name in {"language_persistent", "language_events"}
        or (isinstance(feature, dict) and feature.get("dtype") == "language")
    }
    if language_features:
        raise CurationTransformError(
            "v2.1 conversion does not support rich language or VQA annotations"
        )
    report_progress(
        progress,
        stage="preparing",
        completed=1,
        total=1,
        unit="items",
        current_item="원본 데이터셋 확인",
    )
    report_progress(
        progress,
        stage="validate",
        completed=0,
        total=3,
        unit="items",
        current_item="원본 Export Gate 검사",
        force=True,
    )
    source_gate = validate_dataset_root(
        source_root, mode="export_gate", fingerprint=payload["fingerprint"]
    )
    if not source_gate["passed"]:
        raise CurationTransformError("Source dataset failed the export gate")
    report_progress(
        progress,
        stage="validate",
        completed=1,
        total=3,
        unit="items",
        current_item="원본 Export Gate 검사",
    )

    manifest_path = settings.nas_root / "manifests/conversion" / f"{job_id}.json"
    if manifest_path.is_file() and not manifest_path.is_symlink():
        result = json.loads(manifest_path.read_text(encoding="utf-8"))
        _assert_reusable_manifest(result, record=record, payload=payload)
        output = settings.nas_root / "derived" / result["output"]["relative_path"]
        if (
            output.is_dir()
            and tree_manifest(output)["tree_sha256"]
            == result["output"]["manifest_sha256"]
        ):
            report_progress(
                progress,
                stage="complete",
                completed=1,
                total=1,
                unit="items",
                current_item="기존 변환 결과 재사용",
                force=True,
            )
            return {**result, "reused": True}

    staging_parent = settings.staging_root / "conversion-v21"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f"{job_id}-", dir=staging_parent))
    try:
        destination = staging_root / payload["output_name"]
        source = DatasetSource(source_root, info)
        built = write_dataset(
            source=source,
            destination=destination,
            source_indices=list(range(int(info["total_episodes"]))),
            trim_config={"enabled": False},
            annotations={},
            output_version="v2.1",
            video_codec_policy=VIDEO_CODEC_POLICY,
            on_progress=progress,
        )
        report_progress(
            progress,
            stage="validate",
            completed=1,
            total=3,
            unit="items",
            current_item="변환 결과 구조 검사",
            force=True,
        )
        candidate = inspect_dataset(
            area_root=staging_root,
            storage_area="derived",
            relative_path=payload["output_name"],
        )
        if candidate.readiness != "ready":
            raise CurationTransformError(
                "Converted dataset failed structural validation"
            )
        report_progress(
            progress,
            stage="validate",
            completed=2,
            total=3,
            unit="items",
            current_item="변환 결과 구조 검사",
        )
        output_gate = validate_dataset_root(destination, mode="export_gate")
        if not output_gate["passed"]:
            raise CurationTransformError("Converted dataset failed the export gate")
        report_progress(
            progress,
            stage="validate",
            completed=3,
            total=3,
            unit="items",
            current_item="변환 결과 Export Gate 검사",
        )
        assert_source_unchanged(source_root, source_identity, payload["dataset_id"])
        database.assert_job_lease(job_id, worker_id=worker_id)
        manifest = publish_output(
            database=database,
            settings=settings,
            job_id=job_id,
            worker_id=worker_id,
            staging_path=destination,
            output_name=payload["output_name"],
            on_progress=progress,
        )
        result = {
            "source_dataset_id": record["id"],
            "source_fingerprint": record["fingerprint"],
            "source_version": "v3.0",
            "output_version": "v2.1",
            "conversion_engine": CONVERSION_ENGINE_ID,
            "video_codec_policy": VIDEO_CODEC_POLICY,
            "output": {
                "name": payload["output_name"],
                "relative_path": payload["output_name"],
                "episodes": int(info["total_episodes"]),
                "manifest_sha256": manifest["tree_sha256"],
                "lineage": built["lineage"],
            },
            "validation": output_gate,
            "reused": False,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(manifest_path, {"schema_version": 1, **result})
        report_progress(
            progress,
            stage="register",
            completed=0,
            total=1,
            unit="items",
            current_item="라이브러리 갱신",
            force=True,
        )
        generation = database.begin_dataset_scan("derived")
        database.synchronize_datasets(
            storage_area="derived",
            records=scan_storage_area(
                settings.nas_root, "derived", max_depth=settings.dataset_scan_max_depth
            ),
            scan_generation=generation,
        )
        report_progress(
            progress,
            stage="register",
            completed=1,
            total=1,
            unit="items",
            current_item="라이브러리 갱신",
        )
        report_progress(
            progress,
            stage="complete",
            completed=1,
            total=1,
            unit="items",
            current_item="v2.1 변환 완료",
            force=True,
        )
        return result
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
