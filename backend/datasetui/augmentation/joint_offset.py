"""Joint offset augmentation: per-episode arm joint offsets on state and action.

Robots under impedance control track differently (reducers, zero-pose error),
so a policy trained on one robot underperforms on another. Each augmented copy
adds one fixed random offset per arm joint to every frame of both
observation.state and action, so the model can read "how far off this robot
is" from the state and act consistently with it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from datasetui.config import Settings
from datasetui.database import Database, RecipeRevisionMismatchError
from datasetui.dataset_io.files import (
    read_regular_bytes,
    safe_child,
    safe_dataset_root,
    write_json,
    write_json_atomic,
)
from datasetui.dataset_io.publish import (
    assert_source_unchanged,
    publish_output,
    source_tree_identity,
    tree_manifest,
)
from datasetui.dataset_io.source import DatasetSource
from datasetui.datasets import MAX_INFO_BYTES, inspect_dataset, scan_storage_area
from datasetui.job_progress import JobProgressReporter, report_progress
from datasetui.merge.writer import write_preserved_merge
from datasetui.statistics.output import STATISTICS_POLICY
from datasetui.transform_errors import CurationTransformError


POLICY = "joint-offset-per-episode-uniform-deg-v1"
JOINTS = 7
ARMS = ("right", "left")
# Defaults (degrees, absolute bound per joint j0..j6) from observed tracking error.
DEFAULT_RANGES_DEG = (0.3, 0.1, 1.0, 0.3, 1.0, 0.2, 1.5)
MAX_RANGE_DEG = 10.0
MAX_COPIES = 10
FEATURES = ("observation.state", "action")
METADATA_FILE = "meta/joint_offset_augmentation.json"


class JointOffsetError(CurationTransformError):
    pass


class JointOffsetAugmentationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str
    ranges_deg: list[float] = Field(
        default_factory=lambda: list(DEFAULT_RANGES_DEG),
        min_length=JOINTS,
        max_length=JOINTS,
    )
    copies: int = Field(default=1, ge=1, le=MAX_COPIES, strict=True)
    seed: int = Field(default=0, ge=0, le=2_147_483_647, strict=True)
    # None augments every episode; otherwise only these get copies.
    episode_indices: list[int] | None = Field(default=None, max_length=100_000)
    output_name: str = Field(
        min_length=1,
        max_length=96,
        pattern=r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$",
    )
    idempotency_key: str = Field(min_length=1, max_length=120)

    @field_validator("ranges_deg")
    @classmethod
    def bounded_ranges(cls, value: list[float]) -> list[float]:
        if any(not np.isfinite(item) or not 0 <= item <= MAX_RANGE_DEG for item in value):
            raise ValueError(f"관절 범위는 0~{MAX_RANGE_DEG:g}도여야 합니다.")
        return [float(item) for item in value]

    @field_validator("episode_indices")
    @classmethod
    def unique_episodes(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("증강할 에피소드를 하나 이상 선택하세요.")
        if any(isinstance(item, bool) or item < 0 for item in value):
            raise ValueError("에피소드 번호가 올바르지 않습니다.")
        if len(set(value)) != len(value):
            raise ValueError("에피소드를 중복 선택할 수 없습니다.")
        return sorted(value)

    @field_validator("output_name")
    @classmethod
    def safe_output_name(cls, value: str) -> str:
        if ".." in value:
            raise ValueError("invalid output name")
        return value


INTERNAL_KEYS = {"source", "ranges_deg", "copies", "seed", "episode_indices", "output_name"}


def internal_payload(dataset: dict[str, Any], request: JointOffsetAugmentationCreate) -> dict:
    return {
        "source": {"id": dataset["id"], "fingerprint": dataset["fingerprint"]},
        **request.model_dump(
            include={"ranges_deg", "copies", "seed", "episode_indices", "output_name"}
        ),
    }


def validate_internal_payload(payload: dict[str, Any]) -> dict[str, Any]:
    import re

    if not isinstance(payload, dict) or set(payload) != INTERNAL_KEYS:
        raise ValueError("invalid internal joint offset payload")
    source = payload["source"]
    if (
        not isinstance(source, dict)
        or set(source) != {"id", "fingerprint"}
        or not isinstance(source["id"], str)
        or not isinstance(source["fingerprint"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", source["fingerprint"])
    ):
        raise ValueError("invalid internal joint offset source")
    request = JointOffsetAugmentationCreate.model_validate(
        {
            **{key: payload[key] for key in INTERNAL_KEYS - {"source"}},
            "profile_id": "internal",
            "idempotency_key": "internal",
        }
    )
    return internal_payload(source, request)


def arm_dimensions(info: dict[str, Any]) -> dict[str, dict[str, list[int]]]:
    """Index of `{arm}_arm_{joint}` in each feature's names, per arm, joints 0..6."""
    features = info.get("features", {})
    result: dict[str, dict[str, list[int]]] = {}
    missing: list[str] = []
    for feature in FEATURES:
        names = features.get(feature, {}).get("names")
        if isinstance(names, dict):  # {"motors": [...]} layout
            names = next(iter(names.values()), None)
        names = list(names) if isinstance(names, list) else []
        result[feature] = {}
        for arm in ARMS:
            wanted = [f"{arm}_arm_{joint}" for joint in range(JOINTS)]
            absent = [name for name in wanted if name not in names]
            missing.extend(f"{feature}.{name}" for name in absent)
            result[feature][arm] = [names.index(name) for name in wanted if name in names]
    if missing:
        raise JointOffsetError(
            "관절 오프셋 증강에는 observation.state와 action에 "
            "right_arm_0..6, left_arm_0..6 이름이 필요합니다. 없는 이름: "
            + ", ".join(missing[:8])
            + (" …" if len(missing) > 8 else "")
        )
    return result


def draw_offsets(
    seed: int, ranges_deg: list[float], episodes: list[int], copies: int
) -> list[dict[str, Any]]:
    """One offset per (episode, copy, arm, joint), drawn in a fixed order."""
    rng = np.random.default_rng(seed)
    bounds = np.asarray(ranges_deg, dtype=np.float64)
    draws = []
    for copy_number in range(1, copies + 1):
        for episode in episodes:
            offsets = {
                arm: (rng.uniform(-1.0, 1.0, JOINTS) * bounds).tolist() for arm in ARMS
            }
            draws.append(
                {"source_episode_index": episode, "copy": copy_number, "offsets_deg": offsets}
            )
    return draws


class AugmentedSource:
    """Originals 0..N-1, then each copy block; read through write_preserved_merge."""

    def __init__(self, source: DatasetSource, draws: list[dict[str, Any]]):
        self.sources = [source]
        self.root = source.root
        self.version = source.version
        self.fps = source.fps
        self.video_keys = source.video_keys
        self.info = copy.deepcopy(source.info)
        self.tasks = source.tasks
        self.dimensions = arm_dimensions(source.info)
        originals = int(source.info["total_episodes"])
        self.plan: list[dict[str, Any] | None] = [None] * originals + list(draws)
        self._episodes = [(0, index) for index in range(originals)] + [
            (0, int(draw["source_episode_index"])) for draw in draws
        ]
        self.total_episodes = len(self._episodes)

    def episode(self, episode_index: int) -> tuple[pd.DataFrame, dict[str, Any]]:
        _, local_index = self._episodes[episode_index]
        data, metadata = self.sources[0].episode(local_index)
        draw = self.plan[episode_index]
        metadata = {**metadata, "_merge_source_index": 0, "_merge_local_episode": local_index}
        if draw is None:
            return data, metadata
        data = data.copy()
        for feature in FEATURES:
            values = np.stack([np.asarray(row) for row in data[feature]])
            shifted = values.astype(np.float64)
            for arm in ARMS:
                shifted[:, self.dimensions[feature][arm]] += np.deg2rad(
                    draw["offsets_deg"][arm]
                )
            shifted = shifted.astype(values.dtype)
            data[feature] = list(shifted)
        return data, metadata

    def video_source(
        self, episode_index: int, video_key: str, metadata: dict[str, Any]
    ) -> tuple[Path, int]:
        _, local_index = self._episodes[episode_index]
        return self.sources[0].video_source(local_index, video_key, metadata)


def augmentation_metadata(payload: dict[str, Any], source: AugmentedSource) -> dict:
    return {
        "format_version": 1,
        "policy": POLICY,
        "unit": "degree",
        "applied_as": "radian (deg2rad) added to observation.state and action",
        "features": list(FEATURES),
        "joints": [f"{{arm}}_arm_{joint}" for joint in range(JOINTS)],
        "ranges_deg": payload["ranges_deg"],
        "distribution": "uniform(-range, +range), one draw per episode copy and arm",
        "seed": payload["seed"],
        "copies": payload["copies"],
        "augmented_source_episodes": sorted(
            {draw["source_episode_index"] for draw in source.plan if draw}
        ),
        "episodes": [
            {
                "output_episode_index": index,
                "source_episode_index": local,
                "copy": draw["copy"] if draw else 0,
                "offsets_deg": draw["offsets_deg"] if draw else None,
            }
            for index, ((_, local), draw) in enumerate(zip(source._episodes, source.plan))
        ],
    }


def augment_joint_offsets(
    *,
    database: Database,
    settings: Settings,
    payload: dict[str, Any],
    job_id: str,
    worker_id: str,
) -> dict[str, Any]:
    progress = JobProgressReporter(
        database, job_id=job_id, worker_id=worker_id, output_name=payload["output_name"]
    )
    report_progress(progress, stage="preparing", completed=0, total=1, unit="items",
                    current_item="원본 확인", force=True)
    requested = payload["source"]
    record = database.get_dataset(requested["id"])
    if (
        not record["available"]
        or record["readiness"] != "ready"
        or record["fingerprint"] != requested["fingerprint"]
    ):
        raise RecipeRevisionMismatchError(requested["id"])
    root = safe_dataset_root(settings.nas_root, record["storage_area"], record["relative_path"])
    identity = source_tree_identity(root)
    raw = read_regular_bytes(root / "meta/info.json", max_bytes=MAX_INFO_BYTES)
    if hashlib.sha256(raw).hexdigest() != requested["fingerprint"]:
        raise RecipeRevisionMismatchError(requested["id"])
    from datasetui.relative.artifacts import reject_relative_profile

    reject_relative_profile(root, operation="Joint offset augmentation")
    info = json.loads(raw)
    arm_dimensions(info)
    total = int(info["total_episodes"])
    episodes = payload["episode_indices"]
    episodes = list(range(total)) if episodes is None else episodes
    if any(index >= total for index in episodes):
        raise JointOffsetError("선택한 에피소드가 원본 범위를 벗어났습니다.")

    manifest_path = settings.nas_root / "manifests/augmentation" / f"{job_id}.json"
    output = safe_child(settings.nas_root / "derived", payload["output_name"])
    if manifest_path.is_file() and not manifest_path.is_symlink():
        result = json.loads(manifest_path.read_text(encoding="utf-8"))
        if result.get("request") != payload or result.get("policy") != POLICY:
            raise CurationTransformError(
                "Augmentation manifest differs from request; create a new job/output"
            )
        if output.is_dir() and tree_manifest(output)["tree_sha256"] == result["output"]["manifest_sha256"]:
            report_progress(progress, stage="complete", completed=1, total=1, unit="items",
                            current_item="기존 출력 재사용", force=True)
            return {**result, "reused": True}

    source = AugmentedSource(
        DatasetSource(root, info),
        draw_offsets(payload["seed"], payload["ranges_deg"], episodes, payload["copies"]),
    )
    staging_parent = settings.staging_root / "augmentation"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f"{job_id}-", dir=staging_parent))
    try:
        destination = staging_root / payload["output_name"]
        built = write_preserved_merge(
            source=source,
            destination=destination,
            on_progress=progress,
            recompute_statistics=True,
        )
        write_json(destination / METADATA_FILE, augmentation_metadata(payload, source))
        report_progress(progress, stage="validate", completed=0, total=1, unit="items",
                        current_item="구조 검사", force=True)
        candidate = inspect_dataset(
            area_root=staging_root, storage_area="derived", relative_path=payload["output_name"]
        )
        if candidate.readiness != "ready":
            raise CurationTransformError("Augmented dataset failed structural validation")
        assert_source_unchanged(root, identity, requested["id"])
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
            "policy": POLICY,
            "request": payload,
            "video_policy": "preserve_source_files",
            "statistics": built.get(
                "statistics",
                {"policy": STATISTICS_POLICY, "source": "full-output-recompute"},
            ),
            "source_dataset": {"id": record["id"], "fingerprint": record["fingerprint"]},
            "output": {
                "name": payload["output_name"],
                "relative_path": payload["output_name"],
                "episodes": source.total_episodes,
                "original_episodes": total,
                "augmented_episodes": source.total_episodes - total,
                "manifest_sha256": manifest["tree_sha256"],
            },
            "reused": False,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(manifest_path, {"schema_version": 1, **result})
        report_progress(progress, stage="register", completed=0, total=1, unit="items",
                        current_item="라이브러리 갱신", force=True)
        generation = database.begin_dataset_scan("derived")
        database.synchronize_datasets(
            storage_area="derived",
            records=scan_storage_area(
                settings.nas_root, "derived", max_depth=settings.dataset_scan_max_depth
            ),
            scan_generation=generation,
        )
        report_progress(progress, stage="complete", completed=1, total=1, unit="items",
                        current_item="관절 오프셋 증강 완료", force=True)
        return result
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
