from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from datasetui.config import Settings
from datasetui.database import Database, RecipeRevisionMismatchError
from datasetui.dataset_io.files import (
    read_regular_bytes,
    safe_child,
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
from datasetui.datasets import inspect_dataset, scan_storage_area
from datasetui.merge_progress import MergeProgressReporter
from datasetui.merge_writer import write_preserved_merge
from datasetui.official_operations import enabled, provenance, write_official_merge
from datasetui.output_statistics import STATISTICS_POLICY
from datasetui.transform_errors import CurationTransformError
from datasetui.datasets import MAX_INFO_BYTES



class MergeCompatibilityError(CurationTransformError):
    pass


class _MergedSource:
    def __init__(self, sources: list[DatasetSource], robot_type: str):
        self.sources = sources
        self.root = sources[0].root
        self.version = sources[0].version
        self.fps = sources[0].fps
        self.video_keys = sources[0].video_keys
        self.info = copy.deepcopy(sources[0].info)
        self.info["robot_type"] = robot_type
        self._episodes: list[tuple[int, int]] = []
        self.tasks: dict[int, str] = {}
        task_lookup: dict[str, int] = {}
        self._task_maps: list[dict[int, int]] = []
        for source_index, source in enumerate(sources):
            mapping: dict[int, int] = {}
            for task_index, text_value in source.tasks.items():
                text = str(text_value)
                if text not in task_lookup:
                    task_lookup[text] = len(self.tasks)
                    self.tasks[task_lookup[text]] = text
                mapping[int(task_index)] = task_lookup[text]
            self._task_maps.append(mapping)
            self._episodes.extend(
                (source_index, episode_index)
                for episode_index in range(int(source.info["total_episodes"]))
            )
        self.total_episodes = len(self._episodes)

    def episode(self, episode_index: int) -> tuple[pd.DataFrame, dict[str, Any]]:
        source_index, local_index = self._episodes[episode_index]
        data, metadata = self.sources[source_index].episode(local_index)
        data = data.copy()
        if "task_index" in data.columns:
            mapping = self._task_maps[source_index]
            data["task_index"] = data["task_index"].map(mapping).astype("int64")
        return data, {
            **metadata,
            "_merge_source_index": source_index,
            "_merge_local_episode": local_index,
        }

    def video_source(
        self, episode_index: int, video_key: str, metadata: dict[str, Any]
    ) -> tuple[Path, int]:
        source_index, local_index = self._episodes[episode_index]
        return self.sources[source_index].video_source(local_index, video_key, metadata)


def _compatible_info(
    infos: list[dict[str, Any]], *, normalize_timestamp: bool = False
) -> None:
    if normalize_timestamp:
        infos = copy.deepcopy(infos)
        for info in infos:
            timestamp = info.get("features", {}).get("timestamp", {})
            if timestamp.get("dtype") in {"float32", "float64"}:
                timestamp["dtype"] = "float64"
    first = infos[0]
    for info in infos[1:]:
        if str(info.get("codebase_version")) != str(first.get("codebase_version")):
            raise MergeCompatibilityError("Merge requires the same dataset version")
        if float(info.get("fps", 0)) != float(first.get("fps", 0)):
            raise MergeCompatibilityError("Merge requires the same FPS")
        if info.get("features") != first.get("features"):
            raise MergeCompatibilityError(
                "Merge requires identical feature keys, dtypes, shapes, names, and cameras"
            )


def merge_datasets(
    *,
    database: Database,
    settings: Settings,
    payload: dict[str, Any],
    job_id: str,
    worker_id: str,
) -> dict[str, Any]:
    progress = MergeProgressReporter(
        database,
        job_id=job_id,
        worker_id=worker_id,
        output_name=payload["output_name"],
    )
    records: list[dict[str, Any]] = []
    sources: list[DatasetSource] = []
    infos: list[dict[str, Any]] = []
    source_requests = payload["sources"]
    progress(
        {
            "stage": "preparing",
            "completed": 0,
            "total": len(source_requests),
            "unit": "items",
            "current_item": "원본 확인",
            "_force": True,
        }
    )
    source_identities: list[str] = []
    for source_number, requested in enumerate(source_requests, start=1):
        record = database.get_dataset(requested["id"])
        if (
            not record["available"]
            or record["readiness"] != "ready"
            or record["fingerprint"] != requested["fingerprint"]
        ):
            raise RecipeRevisionMismatchError(requested["id"])
        root = safe_dataset_root(
            settings.nas_root, record["storage_area"], record["relative_path"]
        )
        source_identities.append(source_tree_identity(root))
        raw = read_regular_bytes(root / "meta/info.json", max_bytes=MAX_INFO_BYTES)
        if hashlib.sha256(raw).hexdigest() != requested["fingerprint"]:
            raise RecipeRevisionMismatchError(requested["id"])
        from datasetui.relative_artifacts import reject_relative_profile

        reject_relative_profile(root, operation="Merge")
        info = json.loads(raw)
        records.append(record)
        infos.append(info)
        sources.append(DatasetSource(root, info))
        progress(
            {
                "stage": "preparing",
                "completed": source_number,
                "total": len(source_requests),
                "unit": "items",
                "current_item": f"원본 {source_number} 확인",
            }
        )
    from datasetui.deferred_statistics import POLICY, read_deferred_statistics

    deferred_inputs = [read_deferred_statistics(source.root) for source in sources]
    has_deferred_statistics = any(deferred_inputs)
    official = sources[0].version == "v3.0" and enabled() and not has_deferred_statistics
    _compatible_info(infos, normalize_timestamp=official)
    processing_policy = (
        "lerobot-v3-official-stats-v3" if official else "datasetui-preserved-merge-v2"
    )
    processing = (
        provenance("merge_datasets")
        if official
        else {"engine": processing_policy, "statistics_policy": STATISTICS_POLICY}
    )
    if has_deferred_statistics:
        processing_policy = "datasetui-preserved-merge-deferred-v1"
        processing = {"engine": processing_policy, "statistics_policy": POLICY}

    manifest_path = settings.nas_root / "manifests/merge" / f"{job_id}.json"
    if manifest_path.is_file() and not manifest_path.is_symlink():
        progress(
            {
                "stage": "preparing",
                "completed": 0,
                "total": 0,
                "unit": "files",
                "current_item": "기존 출력 확인",
                "_force": True,
            }
        )
        result = json.loads(manifest_path.read_text(encoding="utf-8"))
        if result.get("processing_policy") != processing_policy:
            raise CurationTransformError(
                "Merge manifest uses a previous processing engine; create a new job/output"
            )
        if (
            result.get("processing") != processing
            or result.get("source_datasets") != source_requests
            or result.get("robot_type") != payload["robot_type"]
            or result.get("output", {}).get("name") != payload["output_name"]
            or result.get("output", {}).get("relative_path") != payload["output_name"]
        ):
            raise CurationTransformError(
                "Merge manifest provenance or output differs from request"
            )
        output = safe_child(settings.nas_root / "derived", payload["output_name"])
        if (
            result.get("video_policy") == "preserve_source_files"
            and output.is_dir()
            and tree_manifest(output)["tree_sha256"]
            == result["output"]["manifest_sha256"]
        ):
            progress(
                {
                    "stage": "complete",
                    "completed": 1,
                    "total": 1,
                    "unit": "items",
                    "current_item": "기존 출력 재사용",
                    "_force": True,
                }
            )
            return {**result, "reused": True}

    staging_parent = settings.staging_root / "merge"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f"{job_id}-", dir=staging_parent))
    try:
        destination = staging_root / payload["output_name"]
        merged = _MergedSource(sources, payload["robot_type"])
        if official:
            built = write_official_merge(
                sources=sources,
                destination=destination,
                robot_type=payload["robot_type"],
                on_progress=progress,
            )
        else:
            built = write_preserved_merge(
                source=merged, destination=destination, on_progress=progress
            )
        progress(
            {
                "stage": "validate",
                "completed": 0,
                "total": 1,
                "unit": "items",
                "current_item": "구조 검사",
                "_force": True,
            }
        )
        candidate = inspect_dataset(
            area_root=staging_root,
            storage_area="derived",
            relative_path=payload["output_name"],
        )
        if candidate.readiness != "ready":
            raise CurationTransformError("Merged dataset failed structural validation")
        progress(
            {
                "stage": "validate",
                "completed": 1,
                "total": 1,
                "unit": "items",
                "current_item": "구조 검사",
            }
        )
        for source, identity, requested in zip(
            sources, source_identities, source_requests
        ):
            assert_source_unchanged(source.root, identity, requested["id"])
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
        lineage = []
        for item, (source_index, local_index) in zip(
            built["lineage"], merged._episodes
        ):
            lineage.append(
                {
                    **item,
                    "source_dataset_id": records[source_index]["id"],
                    "source_fingerprint": records[source_index]["fingerprint"],
                    "source_episode_index": local_index,
                }
            )
        result = {
            "video_policy": "preserve_source_files",
            "processing_policy": processing_policy,
            "processing": built.get("processing", processing),
            "statistics": built.get(
                "statistics",
                {
                    "policy": STATISTICS_POLICY,
                    "source": "full-output-recompute",
                    "fallback": False,
                },
            ),
            "source_datasets": [
                {"id": record["id"], "fingerprint": record["fingerprint"]}
                for record in records
            ],
            "robot_type": payload["robot_type"],
            "output": {
                "name": payload["output_name"],
                "relative_path": payload["output_name"],
                "episodes": merged.total_episodes,
                "manifest_sha256": manifest["tree_sha256"],
                "lineage": lineage,
            },
            "reused": False,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(manifest_path, {"schema_version": 1, **result})
        progress(
            {
                "stage": "register",
                "completed": 0,
                "total": 1,
                "unit": "items",
                "current_item": "라이브러리 갱신",
                "_force": True,
            }
        )
        generation = database.begin_dataset_scan("derived")
        database.synchronize_datasets(
            storage_area="derived",
            records=scan_storage_area(
                settings.nas_root,
                "derived",
                max_depth=settings.dataset_scan_max_depth,
            ),
            scan_generation=generation,
        )
        progress(
            {
                "stage": "register",
                "completed": 1,
                "total": 1,
                "unit": "items",
                "current_item": "라이브러리 갱신",
            }
        )
        progress(
            {
                "stage": "complete",
                "completed": 1,
                "total": 1,
                "unit": "items",
                "current_item": "병합 완료",
                "_force": True,
            }
        )
        return result
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
