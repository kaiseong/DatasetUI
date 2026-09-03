from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from datasetui.config import Settings
from datasetui.database import Database, RecipeRevisionMismatchError
from datasetui.datasets import inspect_dataset, scan_storage_area
from datasetui.transform_errors import CurationTransformError
from datasetui.transforms import (
    MAX_INFO_BYTES,
    _DatasetSource,
    _publish_output,
    _read_regular_bytes,
    _safe_dataset_root,
    _tree_manifest,
    _write_dataset,
    _write_json_atomic,
)
from datasetui.validation import validate_dataset_root


def convert_dataset_to_v21(
    *, database: Database, settings: Settings, payload: dict[str, Any], job_id: str, worker_id: str
) -> dict[str, Any]:
    record = database.get_dataset(payload["dataset_id"])
    if (
        not record["available"]
        or record["readiness"] != "ready"
        or record["fingerprint"] != payload["fingerprint"]
        or record["storage_area"] != payload["storage_area"]
        or record["relative_path"] != payload["relative_path"]
    ):
        raise RecipeRevisionMismatchError(payload["dataset_id"])
    source_root = _safe_dataset_root(
        settings.nas_root, record["storage_area"], record["relative_path"]
    )
    raw = _read_regular_bytes(source_root / "meta/info.json", max_bytes=MAX_INFO_BYTES)
    if hashlib.sha256(raw).hexdigest() != payload["fingerprint"]:
        raise RecipeRevisionMismatchError(payload["dataset_id"])
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
        raise CurationTransformError("v2.1 conversion does not support rich language or VQA annotations")
    source_gate = validate_dataset_root(
        source_root, mode="export_gate", fingerprint=payload["fingerprint"]
    )
    if not source_gate["passed"]:
        raise CurationTransformError("Source dataset failed the export gate")

    manifest_path = settings.nas_root / "manifests/conversion" / f"{job_id}.json"
    if manifest_path.is_file() and not manifest_path.is_symlink():
        result = json.loads(manifest_path.read_text(encoding="utf-8"))
        output = settings.nas_root / "derived" / result["output"]["relative_path"]
        if output.is_dir() and _tree_manifest(output)["tree_sha256"] == result["output"]["manifest_sha256"]:
            return {**result, "reused": True}

    staging_parent = settings.staging_root / "conversion-v21"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f"{job_id}-", dir=staging_parent))
    try:
        destination = staging_root / payload["output_name"]
        source = _DatasetSource(source_root, info)
        built = _write_dataset(
            source=source,
            destination=destination,
            source_indices=list(range(int(info["total_episodes"]))),
            trim_config={"enabled": False},
            annotations={},
            output_version="v2.1",
        )
        candidate = inspect_dataset(
            area_root=staging_root, storage_area="derived", relative_path=payload["output_name"]
        )
        if candidate.readiness != "ready":
            raise CurationTransformError("Converted dataset failed structural validation")
        output_gate = validate_dataset_root(destination, mode="export_gate")
        if not output_gate["passed"]:
            raise CurationTransformError("Converted dataset failed the export gate")
        database.assert_job_lease(job_id, worker_id=worker_id)
        manifest = _publish_output(
            database=database,
            settings=settings,
            job_id=job_id,
            worker_id=worker_id,
            staging_path=destination,
            output_name=payload["output_name"],
        )
        result = {
            "source_dataset_id": record["id"],
            "source_fingerprint": record["fingerprint"],
            "source_version": "v3.0",
            "output_version": "v2.1",
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
        _write_json_atomic(manifest_path, {"schema_version": 1, **result})
        generation = database.begin_dataset_scan("derived")
        database.synchronize_datasets(
            storage_area="derived",
            records=scan_storage_area(
                settings.nas_root, "derived", max_depth=settings.dataset_scan_max_depth
            ),
            scan_generation=generation,
        )
        return result
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
