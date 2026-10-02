"""Curation job: turn a recipe snapshot into published derived datasets."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Any

from datasetui.config import Settings
from datasetui.curation.writer import curation_processing, write_dataset
from datasetui.database import Database, RecipeRevisionMismatchError
from datasetui.dataset_io.files import (
    decode_json_object,
    read_json,
    read_regular_bytes,
    safe_child,
    safe_dataset_root,
)
from datasetui.dataset_io.publish import (
    assert_source_unchanged,
    publish_output,
    refresh_derived_registry,
    source_tree_identity,
    tree_manifest,
    write_run_manifest,
)
from datasetui.dataset_io.source import DatasetSource
from datasetui.datasets import MAX_INFO_BYTES, inspect_dataset
from datasetui.job_progress import (
    JobProgressReporter,
    report_progress,
)
from datasetui.transform_errors import CurationTransformError

CURATION_PROCESSING_POLICY = "official-preferred-source-relative-v2"


def materialize_curation_recipe(
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
    snapshot = database.get_curation_snapshot(payload["snapshot_id"])
    if (
        snapshot["missing_since"] is not None
        or snapshot["readiness"] != "ready"
        or snapshot["fingerprint"] != snapshot["dataset_fingerprint"]
    ):
        raise RecipeRevisionMismatchError(snapshot["dataset_id"])

    source_root = safe_dataset_root(
        settings.nas_root,
        snapshot["storage_area"],
        snapshot["relative_path"],
    )
    source_identity = source_tree_identity(source_root)
    raw_info = read_regular_bytes(
        source_root / "meta" / "info.json", max_bytes=MAX_INFO_BYTES
    )
    if hashlib.sha256(raw_info).hexdigest() != snapshot["dataset_fingerprint"]:
        raise RecipeRevisionMismatchError(snapshot["dataset_id"])
    info = decode_json_object(raw_info)
    version = info.get("codebase_version")
    if version not in {"v2.0", "v2.1", "v3.0"}:
        raise CurationTransformError("Unsupported source dataset version")
    report_progress(
        progress,
        stage="preparing",
        completed=1,
        total=1,
        unit="items",
        current_item="원본 데이터셋 확인",
    )

    outputs = _output_selections(snapshot, payload["output_name"], job_id)
    annotations = (
        database.get_curation_snapshot_annotations(snapshot["id"])
        if snapshot["include_annotations"]
        else {}
    )
    from datasetui.deferred_statistics import read_deferred_statistics

    processing = curation_processing(
        version, snapshot["trim_config"], annotations, snapshot["relative_action"],
        source_statistics_deferred=read_deferred_statistics(source_root),
    )
    completed = reuse_published_outputs(
        settings=settings,
        job_id=job_id,
        outputs=outputs,
        processing=processing,
        snapshot=snapshot,
    )
    if completed is not None:
        report_progress(
            progress,
            stage="register",
            completed=0,
            total=1,
            unit="items",
            current_item="라이브러리 갱신",
            force=True,
        )
        refresh_derived_registry(database, settings)
        report_progress(
            progress,
            stage="complete",
            completed=1,
            total=1,
            unit="items",
            current_item="기존 출력 재사용",
            force=True,
        )
        return completed

    staging_parent = settings.staging_root / "curation"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f"{job_id}-", dir=staging_parent))
    published: list[dict[str, Any]] = []
    try:
        source = DatasetSource(source_root, info)
        for output_index, output in enumerate(outputs):
            destination = staging_root / output["name"]

            def output_progress(event: dict[str, Any]) -> None:
                update = dict(event)
                item = update.get("current_item")
                update["current_item"] = (
                    f"{output['name']} · {item}" if item else output["name"]
                )
                progress(update)

            built = write_dataset(
                source=source,
                destination=destination,
                source_indices=output["episodes"],
                trim_config=snapshot["trim_config"],
                annotations=annotations,
                relative_action=snapshot["relative_action"],
                video_codec_policy="source",
                on_progress=output_progress,
            )
            lineage = built["lineage"]
            report_progress(
                progress,
                stage="validate",
                completed=output_index,
                total=len(outputs),
                unit="items",
                current_item=f"{output['name']} 구조 검사",
                force=True,
            )
            candidate = inspect_dataset(
                area_root=staging_root,
                storage_area="derived",
                relative_path=output["name"],
            )
            if candidate.readiness != "ready":
                raise CurationTransformError(
                    "Derived dataset failed structural validation"
                )
            report_progress(
                progress,
                stage="validate",
                completed=output_index + 1,
                total=len(outputs),
                unit="items",
                current_item=f"{output['name']} 구조 검사",
            )
            assert_source_unchanged(
                source_root, source_identity, snapshot["dataset_id"]
            )
            database.assert_job_lease(job_id, worker_id=worker_id)
            manifest = publish_output(
                database=database,
                settings=settings,
                job_id=job_id,
                worker_id=worker_id,
                staging_path=destination,
                output_name=output["name"],
                on_progress=output_progress,
            )
            published.append(
                {
                    "role": output["role"],
                    "name": output["name"],
                    "relative_path": output["name"],
                    "episodes": len(output["episodes"]),
                    "frames": sum(item["output_length"] for item in lineage),
                    "manifest_sha256": manifest["tree_sha256"],
                    "lineage": lineage,
                    "relative_action": built["relative_action"],
                    "processing": built.get("processing", processing),
                    "statistics": built.get(
                        "statistics",
                        {
                            "policy": "exact-global-numeric-sampled-rgb-v1",
                            "source": "full-output-recompute",
                            "fallback": False,
                        },
                    ),
                }
            )

        result = {
            "snapshot_id": snapshot["id"],
            "source_dataset_id": snapshot["dataset_id"],
            "source_fingerprint": snapshot["dataset_fingerprint"],
            "video_codec_policy": "source",
            "processing_policy": CURATION_PROCESSING_POLICY,
            "operation": snapshot["operation"],
            "outputs": published,
            "reused": False,
        }
        write_run_manifest(settings, job_id, result)
        report_progress(
            progress,
            stage="register",
            completed=0,
            total=1,
            unit="items",
            current_item="라이브러리 갱신",
            force=True,
        )
        refresh_derived_registry(database, settings)
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
            current_item="데이터셋 처리 완료",
            force=True,
        )
        return result
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def _output_selections(
    snapshot: dict[str, Any], base_name: str, job_id: str
) -> list[dict[str, Any]]:
    suffix = job_id.split("-", 1)[0]
    selected = list(snapshot["selected_episode_indices"])
    if snapshot["operation"] == "train_eval_split":
        evaluation_set = set(snapshot["eval_episode_indices"])
        train = [index for index in selected if index not in evaluation_set]
        evaluation = [index for index in selected if index in evaluation_set]
        outputs = []
        if train:
            outputs.append(
                {
                    "role": "train",
                    "name": f"{base_name}_train--{suffix}",
                    "episodes": train,
                }
            )
        if evaluation:
            outputs.append(
                {
                    "role": "eval",
                    "name": f"{base_name}_eval--{suffix}",
                    "episodes": evaluation,
                }
            )
        if not outputs:
            raise CurationTransformError("Train/eval selection cannot be empty")
        return outputs
    if not selected:
        raise CurationTransformError("Derived output cannot be empty")
    role = "delete_flagged" if snapshot["operation"] == "delete_flagged" else "subset"
    return [{"role": role, "name": f"{base_name}--{suffix}", "episodes": selected}]


def reuse_published_outputs(
    *,
    settings: Settings,
    job_id: str,
    outputs: list[dict[str, Any]],
    processing: dict,
    snapshot: dict,
) -> dict[str, Any] | None:
    path = settings.nas_root / "manifests" / "curation" / f"{job_id}.json"
    if not path.is_file() or path.is_symlink():
        return None
    result = read_json(path)
    if result.get("video_codec_policy") != "source":
        raise CurationTransformError(
            "Curation manifest was produced with a legacy video codec policy"
        )
    if result.get("processing_policy") != CURATION_PROCESSING_POLICY:
        raise CurationTransformError(
            "Curation manifest uses a previous processing engine; create a new job/output"
        )
    expected = {item["name"] for item in outputs}
    actual = {item.get("name") for item in result.get("outputs", [])}
    if (
        expected != actual
        or len(result.get("outputs", [])) != len(outputs)
        or result.get("snapshot_id") != snapshot["id"]
        or result.get("source_dataset_id") != snapshot["dataset_id"]
        or result.get("source_fingerprint") != snapshot["dataset_fingerprint"]
    ):
        raise CurationTransformError(
            "Curation manifest conflicts with the requested output"
        )
    for item in result["outputs"]:
        if (
            item.get("processing") != processing
            or item.get("relative_path") != item["name"]
        ):
            raise CurationTransformError(
                "Curation manifest engine or output path differs from request"
            )
        root = safe_child(settings.nas_root / "derived", item["relative_path"])
        if not root.is_dir() or root.is_symlink():
            return None
        if tree_manifest(root)["tree_sha256"] != item["manifest_sha256"]:
            raise CurationTransformError("Published derived dataset was modified")
    return {**result, "reused": True}
