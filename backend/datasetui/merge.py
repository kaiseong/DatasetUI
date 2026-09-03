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


class MergeCompatibilityError(CurationTransformError):
    pass


class _MergedSource:
    def __init__(self, sources: list[_DatasetSource], robot_type: str):
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
        return data, {**metadata, "_merge_source_index": source_index, "_merge_local_episode": local_index}

    def video_source(
        self, episode_index: int, video_key: str, metadata: dict[str, Any]
    ) -> tuple[Path, int]:
        source_index, local_index = self._episodes[episode_index]
        return self.sources[source_index].video_source(local_index, video_key, metadata)


def _compatible_info(infos: list[dict[str, Any]]) -> None:
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
    records: list[dict[str, Any]] = []
    sources: list[_DatasetSource] = []
    infos: list[dict[str, Any]] = []
    for requested in payload["sources"]:
        record = database.get_dataset(requested["id"])
        if (
            not record["available"]
            or record["readiness"] != "ready"
            or record["fingerprint"] != requested["fingerprint"]
        ):
            raise RecipeRevisionMismatchError(requested["id"])
        root = _safe_dataset_root(
            settings.nas_root, record["storage_area"], record["relative_path"]
        )
        raw = _read_regular_bytes(root / "meta/info.json", max_bytes=MAX_INFO_BYTES)
        if hashlib.sha256(raw).hexdigest() != requested["fingerprint"]:
            raise RecipeRevisionMismatchError(requested["id"])
        info = json.loads(raw)
        records.append(record)
        infos.append(info)
        sources.append(_DatasetSource(root, info))
    _compatible_info(infos)

    manifest_path = settings.nas_root / "manifests/merge" / f"{job_id}.json"
    if manifest_path.is_file() and not manifest_path.is_symlink():
        result = json.loads(manifest_path.read_text(encoding="utf-8"))
        output = settings.nas_root / "derived" / result["output"]["relative_path"]
        if output.is_dir() and _tree_manifest(output)["tree_sha256"] == result["output"]["manifest_sha256"]:
            return {**result, "reused": True}

    staging_parent = settings.staging_root / "merge"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f"{job_id}-", dir=staging_parent))
    try:
        destination = staging_root / payload["output_name"]
        merged = _MergedSource(sources, payload["robot_type"])
        built = _write_dataset(
            source=merged,  # type: ignore[arg-type]
            destination=destination,
            source_indices=list(range(merged.total_episodes)),
            trim_config={"enabled": False},
            annotations={},
        )
        candidate = inspect_dataset(
            area_root=staging_root,
            storage_area="derived",
            relative_path=payload["output_name"],
        )
        if candidate.readiness != "ready":
            raise CurationTransformError("Merged dataset failed structural validation")
        database.assert_job_lease(job_id, worker_id=worker_id)
        manifest = _publish_output(
            database=database,
            settings=settings,
            job_id=job_id,
            worker_id=worker_id,
            staging_path=destination,
            output_name=payload["output_name"],
        )
        lineage = []
        for item, (source_index, local_index) in zip(built["lineage"], merged._episodes):
            lineage.append(
                {
                    **item,
                    "source_dataset_id": records[source_index]["id"],
                    "source_fingerprint": records[source_index]["fingerprint"],
                    "source_episode_index": local_index,
                }
            )
        result = {
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
        _write_json_atomic(manifest_path, {"schema_version": 1, **result})
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
        return result
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
