from __future__ import annotations

import copy
import hashlib
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from datasetui.config import Settings
from datasetui.database import Database, RecipeRevisionMismatchError
from datasetui.dataset_io.files import (
    decode_json_object,
    read_json,
    read_regular_bytes,
    safe_child,
    safe_dataset_root,
    write_json,
    write_json_lines,
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
from datasetui.dataset_io.stats import write_stats
from datasetui.dataset_io.tables import (
    episode_task_names,
    language_column_types,
    LANGUAGE_EVENTS,
    LANGUAGE_PERSISTENT,
    write_parquet,
    write_v3_tasks,
)
from datasetui.dataset_io.video import (
    probe_video_codec,
    slice_video,
    update_video_feature_codec,
    video_encoder,
)
from datasetui.datasets import inspect_dataset, MAX_INFO_BYTES
from datasetui.job_progress import (
    JobProgressReporter,
    ProgressCallback,
    report_progress,
)
from datasetui.transform_errors import CurationTransformError



PERSISTENT_STYLES = {"task_aug", "subtask", "plan", "memory"}
EVENT_STYLES = {"interjection", "vqa"}
CURATION_PROCESSING_POLICY = "official-preferred-source-relative-v2"
SAY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "say",
        "description": "Speak a short utterance to the user via the TTS executor.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The verbatim text to speak.",
                }
            },
            "required": ["text"],
        },
    },
}


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

    processing = _curation_processing(
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

            built = _write_dataset(
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


def _curation_processing(
    version, trim_config, annotations, relative_action, output_version=None,
    *, source_statistics_deferred=False,
):
    from datasetui.official_operations import enabled, provenance
    from datasetui.output_statistics import STATISTICS_POLICY
    from datasetui.deferred_statistics import POLICY, should_defer

    if should_defer(trim_config, relative_action, inherited=source_statistics_deferred):
        return {"engine": "datasetui-custom-extension-v5", "statistics_policy": POLICY,
                "relative_action_policy": "lerobot-exact-mask-chunk-v1"}

    if (
        version == "v3.0"
        and output_version in {None, "v3.0"}
        and not trim_config.get("enabled", False)
        and not annotations
        and not (relative_action or {}).get("enabled", False)
        and enabled()
    ):
        return provenance("split_dataset")
    return {
        "engine": "datasetui-custom-extension-v4",
        "statistics_policy": STATISTICS_POLICY,
        "relative_action_policy": "lerobot-exact-mask-chunk-v1",
    }


def _write_dataset(
    *,
    source: DatasetSource,
    destination: Path,
    source_indices: list[int],
    trim_config: dict[str, Any],
    annotations: dict[int, dict[str, Any]],
    relative_action: dict[str, Any] | None = None,
    output_version: str | None = None,
    video_codec_policy: str = "source",
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    from datasetui.official_operations import write_official_subset
    from datasetui.relative_artifacts import reject_relative_profile
    from datasetui.deferred_statistics import (
        read_deferred_statistics, should_defer, preserve_deferred_statistics,
    )

    if not (relative_action or {}).get("enabled", False):
        reject_relative_profile(source.root, operation="Curation without Relative")

    stationary_trim = trim_config.get("enabled", False) and (
        trim_config.get("method", "legacy_motion") == "stationary"
    )
    target_version = output_version or source.version
    if stationary_trim and (source.version != "v3.0" or target_version != "v3.0"):
        raise CurationTransformError(
            "Stationary no-reencode trim supports only v3.0 to v3.0 datasets; "
            "use legacy motion trim when a v2 dataset must be re-encoded"
        )
    if stationary_trim and video_codec_policy != "source":
        raise CurationTransformError(
            "Stationary no-reencode trim requires the source video codec policy"
        )

    source_statistics_deferred = read_deferred_statistics(source.root)
    defer_statistics = should_defer(
        trim_config, relative_action, inherited=source_statistics_deferred,
    )

    if (
        (relative_action or {}).get("enabled", False)
        and not source_statistics_deferred
        and not trim_config.get("enabled", False)
        and not annotations
        and output_version in {None, source.version}
        and source_indices == sorted(source.episode_metadata)
    ):
        # Relative is a training transform, not a video or row rewrite. Preserve
        # every original data/video byte when the whole dataset is selected.
        from datasetui.exact_statistics import recompute_numeric_statistics
        from datasetui.official_operations import _validate_destination
        from datasetui.processing_sources import private_sources
        from datasetui.relative_artifacts import DatasetEpisodes
        from datasetui.visual_statistics import recompute_visual_statistics

        _validate_destination(destination, [source.root.resolve()])
        with private_sources(
            [source.root], destination.parent, on_progress=on_progress
        ) as copies:
            copied = DatasetSource(copies[0], read_json(copies[0] / "meta/info.json"))
            relative_profile = _relative_action_profile(
                copied.info,
                DatasetEpisodes(copied),
                relative_action,
                on_progress=on_progress,
            )
            stats_path = copies[0] / "meta/stats.json"
            absolute_stats = read_json(stats_path) if stats_path.exists() else {}
            absolute_stats.update(
                recompute_numeric_statistics(copies[0], on_progress=on_progress)
            )
            absolute_stats.update(
                recompute_visual_statistics(copies[0], on_progress=on_progress)
            )
            write_json(copies[0] / "meta/stats.json", absolute_stats)
            _write_relative_profile(copies[0], relative_profile)
            shutil.move(str(copies[0]), str(destination))
        return {
            "lineage": [
                {
                    "source_episode_index": index,
                    "output_episode_index": index,
                    "source_start_frame": 0,
                    "source_end_frame": int(source.episode_metadata[index]["length"]),
                    "output_length": int(source.episode_metadata[index]["length"]),
                    "trim_method": "disabled",
                    "task_overridden": False,
                    "persistent_annotations": 0,
                    "event_annotations": 0,
                }
                for index in source_indices
            ],
            "relative_action": relative_profile,
        }

    if (
        _curation_processing(
            source.version, trim_config, annotations, relative_action, output_version,
            source_statistics_deferred=source_statistics_deferred,
        ).get("official_function")
        == "split_dataset"
    ):
        return write_official_subset(
            source=source,
            destination=destination,
            source_indices=source_indices,
            on_progress=on_progress,
        )
    (destination / "meta").mkdir(parents=True)
    (destination / "data").mkdir()
    if source.video_keys:
        (destination / "videos").mkdir()
    readme = source.root / "README.md"
    if readme.is_file() and not readme.is_symlink():
        shutil.copy2(readme, destination / "README.md")

    episodes: list[tuple[pd.DataFrame, dict[str, Any], int, int]] = []
    lineage: list[dict[str, Any]] = []
    global_index = 0
    report_progress(
        on_progress,
        stage="read",
        completed=0,
        total=len(source_indices),
        unit="episodes",
        force=True,
    )
    for output_index, source_index in enumerate(source_indices):
        data, metadata = source.episode(source_index)
        metadata = {**metadata, "_source_episode_index": source_index}
        annotation = annotations.get(source_index)
        atoms = (
            annotation["atoms"]
            if annotation is not None
            else _extract_existing_language_atoms(data)
        )
        if annotation is not None:
            metadata["_task_override"] = annotation["task_override"]
        start, end, method = _trim_bounds(
            data,
            source.info,
            source.fps,
            trim_config,
            source_index,
        )
        trimmed = data.iloc[start:end].copy().reset_index(drop=True)
        if len(trimmed) < 2:
            raise CurationTransformError(
                "Trim would create an episode shorter than two frames"
            )
        trimmed["episode_index"] = output_index
        if start != 0 or end != len(data):
            trimmed["frame_index"] = np.arange(len(trimmed), dtype=np.int64)
            timestamp_dtype = data["timestamp"].dtype
            trimmed["timestamp"] = (
                np.arange(len(trimmed), dtype=np.float64) / source.fps
            ).astype(timestamp_dtype)
        trimmed["index"] = np.arange(
            global_index, global_index + len(trimmed), dtype=np.int64
        )
        persistent_count, event_count = _replace_language_columns(
            trimmed,
            source_data=data,
            atoms=atoms,
            start=start,
            end=end,
            fps=source.fps,
            video_keys=source.video_keys,
        )
        global_index += len(trimmed)
        episodes.append((trimmed, metadata, start, end))
        lineage.append(
            {
                "source_episode_index": source_index,
                "output_episode_index": output_index,
                "source_start_frame": start,
                "source_end_frame": end,
                "output_length": len(trimmed),
                "trim_method": method,
                "task_overridden": bool(metadata.get("_task_override")),
                "persistent_annotations": persistent_count,
                "event_annotations": event_count,
            }
        )
        report_progress(
            on_progress,
            stage="read",
            completed=output_index + 1,
            total=len(source_indices),
            unit="episodes",
            current_item=f"에피소드 {source_index}",
        )

    task_mapping, tasks = _remap_tasks(
        episodes, _apply_task_overrides(episodes, source.tasks)
    )
    for data, _, _, _ in episodes:
        if "task_index" in data.columns:
            data["task_index"] = data["task_index"].map(task_mapping).astype("int64")

    relative_profile = _relative_action_profile(
        source.info,
        [item[0] for item in episodes],
        relative_action or {},
        on_progress=on_progress,
    )

    language_types = language_column_types(episodes)
    output_video_codecs = (
        {}
        if stationary_trim
        else _output_video_codecs(source, episodes, policy=video_codec_policy)
    )
    if target_version == "v3.0":
        _write_v3(
            source,
            destination,
            episodes,
            tasks,
            language_types,
            output_video_codecs=output_video_codecs,
            preserve_source_codec=video_codec_policy == "source",
            logical_stationary=stationary_trim,
            on_progress=on_progress,
        )
    else:
        _write_v2(
            source,
            destination,
            episodes,
            tasks,
            language_types,
            target_version=target_version,
            output_video_codecs=output_video_codecs,
            preserve_source_codec=video_codec_policy == "source",
            on_progress=on_progress,
        )
    used_legacy_aggregate = False
    legacy_aggregate_eligible = (
        source.version in {"v2.0", "v2.1"}
        and not source_statistics_deferred
        and target_version == source.version
        and not trim_config.get("enabled", False)
        and not annotations
        and not (relative_action or {}).get("enabled", False)
    )
    if legacy_aggregate_eligible:
        from datasetui.official_operations import (
            write_legacy_aggregated_statistics,
        )

        used_legacy_aggregate = write_legacy_aggregated_statistics(
            destination,
            [
                (source.root, source_index, item[0])
                for source_index, item in zip(source_indices, episodes, strict=True)
            ],
        )
    if not used_legacy_aggregate and not defer_statistics:
        if legacy_aggregate_eligible:
            report_progress(
                on_progress,
                stage="statistics",
                completed=0,
                total=0,
                unit="items",
                current_item=(
                    "에피소드 통계가 없거나 불완전하여 전체 통계를 재계산합니다"
                ),
                force=True,
            )
        write_stats(
            destination / "meta" / "stats.json",
            [item[0] for item in episodes],
            on_progress=on_progress,
        )
    from datasetui.output_metadata import copy_modality_metadata

    copy_modality_metadata([source.root], destination)
    _write_relative_profile(destination, relative_profile)
    deferred_result = None
    if defer_statistics:
        deferred_result = preserve_deferred_statistics(source.root, destination)
        report_progress(on_progress, stage="statistics", completed=1, total=1,
                         unit="items", current_item="분포 통계 재계산 생략 · 학습 전 norm_stats 계산 필요")
    return {
        "lineage": lineage,
        "relative_action": relative_profile,
        "statistics_reused": used_legacy_aggregate,
        "statistics": deferred_result or {
            "policy": (
                "lerobot-official-aggregate-v1"
                if used_legacy_aggregate
                else "exact-global-numeric-sampled-rgb-v1"
            ),
            "source": (
                "legacy-episode-statistics"
                if used_legacy_aggregate
                else "full-output-recompute"
            ),
            "fallback": legacy_aggregate_eligible and not used_legacy_aggregate,
        },
    }


def _write_relative_profile(destination: Path, relative_profile: dict) -> None:
    if relative_profile["enabled"]:
        absolute_stats = read_json(destination / "meta/stats.json")
        write_json(destination / "meta/stats.absolute.json", absolute_stats)
        write_json(
            destination / "meta/stats.json",
            {**absolute_stats, "action": relative_profile["statistics"]},
        )
        write_json(
            destination / "meta/relative_action.json",
            {"format_version": 1, **relative_profile},
        )
        from datasetui.relative_artifacts import write_training_instructions

        write_training_instructions(destination)


def _relative_action_profile(
    info: dict[str, Any],
    episodes: list[pd.DataFrame],
    config: dict[str, Any],
    *,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    from datasetui.relative_actions import compute_relative_action_profile

    return compute_relative_action_profile(
        info, episodes, config, on_progress=on_progress
    )


def _coerce_atom(
    value: Any, *, fallback_timestamp: float | None = None
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        try:
            value = dict(value)
        except (TypeError, ValueError):
            return None
    role = value.get("role")
    if not isinstance(role, str) or not role:
        return None
    timestamp = value.get("timestamp", fallback_timestamp)
    try:
        timestamp = float(timestamp if timestamp is not None else 0.0)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(timestamp) or timestamp < 0:
        return None
    tool_calls = value.get("tool_calls")
    if isinstance(tool_calls, np.ndarray):
        tool_calls = tool_calls.tolist()
    if tool_calls is not None:
        if not isinstance(tool_calls, list):
            return None
        from datasetui.models import AnnotationToolCall

        try:
            tool_calls = [
                AnnotationToolCall.model_validate(call).model_dump()
                for call in tool_calls
            ]
        except ValueError:
            # The current writer only represents SAY(text). Reject other JSON
            # shapes before Arrow can silently discard unknown arguments.
            return None
    camera = value.get("camera")
    return {
        "role": role,
        "content": None if value.get("content") is None else str(value["content"]),
        "style": value.get("style"),
        "timestamp": timestamp,
        "camera": camera if isinstance(camera, str) and camera else None,
        "tool_calls": tool_calls or None,
    }


def _extract_existing_language_atoms(data: pd.DataFrame) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []

    def read_many(values: Any, *, timestamp: float | None = None) -> list[dict[str, Any]]:
        if values is None:
            return []
        if isinstance(values, np.ndarray):
            values = values.tolist()
        if not isinstance(values, list):
            raise CurationTransformError("Existing annotation column must contain lists")
        parsed = []
        for value in values:
            if not isinstance(value, dict) or set(value) - {
                "role", "content", "style", "timestamp", "camera", "tool_calls"
            }:
                raise CurationTransformError(
                    "Existing annotation contains unsupported fields or structure"
                )
            atom = _coerce_atom(value, fallback_timestamp=timestamp)
            if atom is None:
                raise CurationTransformError("Existing annotation payload is invalid")
            parsed.append(atom)
        return parsed

    persistent_atoms = None
    for _, row in data.iterrows():
        if LANGUAGE_PERSISTENT in data.columns:
            current = read_many(row[LANGUAGE_PERSISTENT])
            if persistent_atoms is None:
                persistent_atoms = current
                atoms.extend(current)
            elif current != persistent_atoms:
                raise CurationTransformError(
                    "Persistent annotation rows differ; unsupported broadcast structure"
                )
        if LANGUAGE_EVENTS in data.columns:
            atoms.extend(read_many(row[LANGUAGE_EVENTS], timestamp=float(row["timestamp"])))
    # Match upstream canonical ordering without discarding repeated messages.
    atoms.sort(
        key=lambda atom: (atom["timestamp"], atom.get("style") or "", atom["role"])
    )
    return atoms


def _language_row(atom: dict[str, Any], *, persistent: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "role": str(atom["role"]),
        "content": None if atom.get("content") is None else str(atom["content"]),
        "style": atom.get("style"),
    }
    if persistent:
        row["timestamp"] = np.float32(atom["timestamp"])
    row["camera"] = atom.get("camera")
    row["tool_calls"] = copy.deepcopy(atom.get("tool_calls")) or None
    return row


def _replace_language_columns(
    output: pd.DataFrame,
    *,
    source_data: pd.DataFrame,
    atoms: list[dict[str, Any]],
    start: int,
    end: int,
    fps: float,
    video_keys: list[str],
) -> tuple[int, int]:
    unsupported = {"tools", "subtask_index"}.intersection(source_data.columns)
    if unsupported:
        raise CurationTransformError(
            "Unsupported language columns cannot be removed during curation: "
            + ", ".join(sorted(unsupported))
        )
    output.drop(
        columns=[
            name
            for name in (LANGUAGE_PERSISTENT, LANGUAGE_EVENTS, "tools", "subtask_index")
            if name in output.columns
        ],
        inplace=True,
    )
    if not atoms:
        return 0, 0

    source_timestamps = source_data["timestamp"].astype(float).to_numpy()
    persistent: list[dict[str, Any]] = []
    events_by_frame: dict[int, list[dict[str, Any]]] = {}
    pre_trim_latest: dict[str, dict[str, Any]] = {}
    for raw_atom in atoms:
        atom = _coerce_atom(raw_atom)
        if atom is None:
            raise CurationTransformError("Annotation payload is invalid")
        style = atom.get("style")
        if style == "vqa" and atom.get("camera") not in video_keys:
            raise CurationTransformError("Annotation references an unavailable camera")
        if style != "vqa" and atom.get("camera") is not None:
            raise CurationTransformError("Only VQA annotations may reference a camera")
        source_frame = int(np.argmin(np.abs(source_timestamps - atom["timestamp"])))
        if style in PERSISTENT_STYLES:
            if source_frame >= end:
                continue
            if source_frame < start:
                if style == "memory":
                    continue
                if style in {"subtask", "plan"}:
                    pre_trim_latest[style] = atom
                    continue
                projected_timestamp = 0.0
            else:
                projected_timestamp = (source_frame - start) / fps
            projected = {**atom, "timestamp": projected_timestamp}
            persistent.append(_language_row(projected, persistent=True))
        elif style in EVENT_STYLES or style is None:
            if start <= source_frame < end:
                output_frame = source_frame - start
                events_by_frame.setdefault(output_frame, []).append(
                    _language_row(atom, persistent=False)
                )
        else:
            raise CurationTransformError("Annotation style is unsupported")

    for atom in pre_trim_latest.values():
        persistent.append(_language_row({**atom, "timestamp": 0.0}, persistent=True))
    persistent.sort(
        key=lambda row: (float(row["timestamp"]), row.get("style") or "", row["role"])
    )
    for rows in events_by_frame.values():
        rows.sort(
            key=lambda row: (
                row.get("style") or "",
                row["role"],
                row.get("camera") or "",
            )
        )

    if persistent:
        output[LANGUAGE_PERSISTENT] = [
            copy.deepcopy(persistent) for _ in range(len(output))
        ]
    if events_by_frame:
        output[LANGUAGE_EVENTS] = [
            copy.deepcopy(events_by_frame.get(frame_index, []))
            for frame_index in range(len(output))
        ]
    return len(persistent), sum(len(rows) for rows in events_by_frame.values())


def _apply_task_overrides(
    episodes: list[tuple[pd.DataFrame, dict[str, Any], int, int]],
    source_tasks: dict[int, str],
) -> dict[int, str]:
    tasks = dict(source_tasks)
    by_name = {name: index for index, name in tasks.items()}
    next_index = max(tasks, default=-1) + 1
    for data, metadata, _, _ in episodes:
        task = metadata.get("_task_override")
        if not task:
            continue
        task_index = by_name.get(task)
        if task_index is None:
            task_index = next_index
            next_index += 1
            tasks[task_index] = task
            by_name[task] = task_index
        data["task_index"] = np.full(len(data), task_index, dtype=np.int64)
    return tasks


def _updated_info(
    source_info: dict[str, Any],
    episodes: list[tuple[pd.DataFrame, dict[str, Any], int, int]],
    language_types: dict[str, pa.DataType],
    output_video_codecs: dict[str, str] | None = None,
) -> dict[str, Any]:
    info = copy.deepcopy(source_info)
    features = info.setdefault("features", {})
    features.pop("subtask_index", None)
    features.pop("tools", None)
    for name in (LANGUAGE_PERSISTENT, LANGUAGE_EVENTS):
        if name in language_types:
            features[name] = {"dtype": "language", "shape": [1], "names": None}
        else:
            features.pop(name, None)
    if any("task_index" in data.columns for data, _, _, _ in episodes):
        features.setdefault(
            "task_index", {"dtype": "int64", "shape": [1], "names": None}
        )
    for key, codec in (output_video_codecs or {}).items():
        feature = features.get(key)
        if not isinstance(feature, dict):
            continue
        update_video_feature_codec(feature, codec)
    has_speech = any(
        row.get("style") is None and row.get("tool_calls")
        for data, _, _, _ in episodes
        if LANGUAGE_EVENTS in data.columns
        for rows in data[LANGUAGE_EVENTS]
        for row in rows
    )
    if has_speech:
        existing_tools = [
            tool for tool in info.get("tools", []) if isinstance(tool, dict)
        ]
        existing_names = {
            (tool.get("function") or {}).get("name") for tool in existing_tools
        }
        if "say" not in existing_names:
            existing_tools.append(copy.deepcopy(SAY_TOOL_SCHEMA))
        info["tools"] = existing_tools
    return info


def _trim_bounds(
    data: pd.DataFrame,
    info: dict[str, Any],
    fps: float,
    config: dict[str, Any],
    episode_index: int,
) -> tuple[int, int, str]:
    if not config.get("enabled", False):
        return 0, len(data), "disabled"
    override = config.get("episode_overrides", {}).get(str(episode_index))
    if override is None:
        override = config.get("episode_overrides", {}).get(episode_index)
    if override is not None:
        start, end = int(override["start_frame"]), int(override["end_frame"])
        if start < 0 or end > len(data) or end <= start:
            raise CurationTransformError("Manual trim override is outside the episode")
        return start, end, "manual"

    method = config.get("method", "legacy_motion")
    if method == "stationary":
        from datasetui.stationary_trim import stationary_trim_bounds

        return stationary_trim_bounds(
            data, fps=fps, config=config, episode_index=episode_index
        )
    if method != "legacy_motion":
        raise CurationTransformError(f"Unsupported trim method: {method}")

    action = _matrix_column(data, "action")
    state = _matrix_column(data, "observation.state")
    action_names = _feature_names(info, "action", action.shape[1])
    state_names = _feature_names(info, "observation.state", state.shape[1])
    common = [name for name in action_names if name in set(state_names)]
    requested = list(config.get("dimensions") or common)
    if not requested:
        raise CurationTransformError("Trim requires matching action/state dimensions")
    if any(name not in common for name in requested):
        raise CurationTransformError(
            "A selected trim dimension is not shared by action and state"
        )
    columns = []
    for name in requested:
        columns.append(action[:, action_names.index(name)])
        columns.append(state[:, state_names.index(name)])
    signals = np.column_stack(columns).astype(np.float64)
    q05 = np.nanpercentile(signals, 5, axis=0)
    q95 = np.nanpercentile(signals, 95, axis=0)
    scale = q95 - q05 + 1e-8
    score = np.max(np.abs(np.diff(signals, axis=0)) / scale, axis=1)
    active = np.isfinite(score) & (score >= float(config.get("threshold", 0.02)))

    def seconds(side: str, field: str, default: float) -> float:
        value = config.get(f"{side}_{field}")
        return float(config.get(field, default) if value is None else value)

    start_hold = max(1, int(math.ceil(seconds("start", "hold_time_s", 0.5) * fps)))
    end_hold = max(1, int(math.ceil(seconds("end", "hold_time_s", 0.5) * fps)))
    start_runs = _true_runs(active, start_hold)
    end_runs = _true_runs(active, end_hold)
    if not start_runs and not end_runs:
        return 0, len(data), "no_sustained_motion"
    start_margin = max(0, int(round(seconds("start", "margin_s", 1.0) * fps)))
    end_margin = max(0, int(round(seconds("end", "margin_s", 1.0) * fps)))
    # If one side finds no sustained motion, preserve that edge.
    start = max(0, start_runs[0][0] + 1 - start_margin) if start_runs else 0
    end = min(len(data), end_runs[-1][1] + 2 + end_margin) if end_runs else len(data)
    return start, end, "motion"


def _true_runs(values: np.ndarray, minimum: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values.tolist() + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= minimum:
                runs.append((start, index - 1))
            start = None
    return runs


def _write_v2(
    source: DatasetSource,
    root: Path,
    episodes: list[tuple[pd.DataFrame, dict[str, Any], int, int]],
    tasks: list[dict[str, Any]],
    language_types: dict[str, pa.DataType],
    *,
    target_version: str | None = None,
    output_video_codecs: dict[str, str] | None = None,
    preserve_source_codec: bool = False,
    on_progress: ProgressCallback | None = None,
) -> None:
    info = _updated_info(source.info, episodes, language_types, output_video_codecs)
    info.pop("total_chunks", None)
    info["data_path"] = (
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    info["video_path"] = (
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )
    info.update(
        codebase_version=target_version or source.version,
        total_episodes=len(episodes),
        total_frames=sum(len(item[0]) for item in episodes),
        total_tasks=len(tasks),
        splits={"train": f"0:{len(episodes)}"},
    )
    write_json(root / "meta" / "info.json", info)
    write_json_lines(root / "meta" / "tasks.jsonl", tasks)
    episode_rows = []
    chunk_size = int(info.get("chunks_size", 1000))
    template = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    report_progress(
        on_progress,
        stage="write",
        completed=0,
        total=len(episodes),
        unit="episodes",
        force=True,
    )
    for index, (data, metadata, start, end) in enumerate(episodes):
        relative = template.format(
            episode_chunk=index // chunk_size, episode_index=index
        )
        path = safe_child(root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_parquet(data, path, language_types)
        episode_rows.append(
            {
                "episode_index": index,
                "tasks": episode_task_names(data, tasks),
                "length": len(data),
            }
        )
        report_progress(
            on_progress,
            stage="write",
            completed=index + 1,
            total=len(episodes),
            unit="episodes",
            current_item=f"에피소드 {index}",
        )
    write_json_lines(root / "meta" / "episodes.jsonl", episode_rows)
    video_total = sum(len(item[0]) for item in episodes) * len(source.video_keys) * 2
    video_completed = 0
    if video_total:
        report_progress(
            on_progress,
            stage="video",
            completed=0,
            total=video_total,
            unit="frame_operations",
            force=True,
        )
    for index, (data, metadata, start, end) in enumerate(episodes):
        video_completed += _write_episode_videos(
            source,
            root,
            index,
            metadata,
            start,
            end,
            len(data),
            output_version=target_version or source.version,
            output_video_codecs=output_video_codecs or {},
            preserve_source_codec=preserve_source_codec,
            on_progress=on_progress,
            progress_base=video_completed,
            progress_total=video_total,
        )


def _write_v3(
    source: DatasetSource,
    root: Path,
    episodes: list[tuple[pd.DataFrame, dict[str, Any], int, int]],
    tasks: list[dict[str, Any]],
    language_types: dict[str, pa.DataType],
    *,
    output_video_codecs: dict[str, str] | None = None,
    preserve_source_codec: bool = False,
    logical_stationary: bool = False,
    on_progress: ProgressCallback | None = None,
) -> None:
    if logical_stationary:
        _write_v3_stationary(
            source,
            root,
            episodes,
            tasks,
            language_types,
            on_progress=on_progress,
        )
        return
    info = _updated_info(source.info, episodes, language_types, output_video_codecs)
    info.update(
        total_episodes=len(episodes),
        total_frames=sum(len(item[0]) for item in episodes),
        total_tasks=len(tasks),
        total_chunks=max(1, math.ceil(len(episodes) / 1000)),
        chunks_size=1000,
        splits={"train": f"0:{len(episodes)}"},
        data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )
    write_json(root / "meta" / "info.json", info)
    write_v3_tasks(root, tasks)
    metadata_rows: list[dict[str, Any]] = []
    offset = 0
    report_progress(
        on_progress,
        stage="write",
        completed=0,
        total=len(episodes),
        unit="episodes",
        force=True,
    )
    for index, (data, metadata, start, end) in enumerate(episodes):
        chunk = index // 1000
        file_index = index % 1000
        path = root / f"data/chunk-{chunk:03d}/file-{file_index:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_parquet(data, path, language_types)
        row: dict[str, Any] = {
            "episode_index": index,
            "tasks": episode_task_names(data, tasks),
            "length": len(data),
            "data/chunk_index": chunk,
            "data/file_index": file_index,
            "dataset_from_index": offset,
            "dataset_to_index": offset + len(data),
        }
        for key in source.video_keys:
            row[f"videos/{key}/chunk_index"] = chunk
            row[f"videos/{key}/file_index"] = file_index
            row[f"videos/{key}/from_timestamp"] = 0.0
            row[f"videos/{key}/to_timestamp"] = len(data) / source.fps
        metadata_rows.append(row)
        offset += len(data)
        report_progress(
            on_progress,
            stage="write",
            completed=index + 1,
            total=len(episodes),
            unit="episodes",
            current_item=f"에피소드 {index}",
        )
    metadata_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metadata_rows).to_parquet(metadata_path, index=False)
    video_total = sum(len(item[0]) for item in episodes) * len(source.video_keys) * 2
    video_completed = 0
    if video_total:
        report_progress(
            on_progress,
            stage="video",
            completed=0,
            total=video_total,
            unit="frame_operations",
            force=True,
        )
    for index, (data, metadata, start, end) in enumerate(episodes):
        video_completed += _write_episode_videos(
            source,
            root,
            index,
            metadata,
            start,
            end,
            len(data),
            output_version="v3.0",
            output_video_codecs=output_video_codecs or {},
            preserve_source_codec=preserve_source_codec,
            on_progress=on_progress,
            progress_base=video_completed,
            progress_total=video_total,
        )


def _write_v3_stationary(
    source: DatasetSource,
    root: Path,
    episodes: list[tuple[pd.DataFrame, dict[str, Any], int, int]],
    tasks: list[dict[str, Any]],
    language_types: dict[str, pa.DataType],
    *,
    on_progress: ProgressCallback | None = None,
) -> None:
    """Write a v3 logical trim while preserving source video bytes."""
    from datasetui.merge_writer import _copy_videos

    info = _updated_info(source.info, episodes, language_types)
    info.update(
        total_episodes=len(episodes),
        total_frames=sum(len(item[0]) for item in episodes),
        total_tasks=len(tasks),
        total_chunks=max(1, math.ceil(len(episodes) / 1000)),
        chunks_size=1000,
        splits={"train": f"0:{len(episodes)}"},
        data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )
    write_json(root / "meta" / "info.json", info)
    write_v3_tasks(root, tasks)

    video_locations: dict[tuple[str, Path], tuple[int, int]] = {}
    next_video_file: dict[str, int] = {key: 0 for key in source.video_keys}
    copies: list[tuple[Path, Path, Path]] = []
    metadata_rows: list[dict[str, Any]] = []
    offset = 0
    report_progress(
        on_progress,
        stage="write",
        completed=0,
        total=len(episodes),
        unit="episodes",
        force=True,
    )
    for index, (data, metadata, start, end) in enumerate(episodes):
        chunk, file_index = index // 1000, index % 1000
        path = root / f"data/chunk-{chunk:03d}/file-{file_index:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_parquet(data, path, language_types)
        row: dict[str, Any] = {
            "episode_index": index,
            "tasks": episode_task_names(data, tasks),
            "length": len(data),
            "data/chunk_index": chunk,
            "data/file_index": file_index,
            "dataset_from_index": offset,
            "dataset_to_index": offset + len(data),
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        source_index = int(metadata["_source_episode_index"])
        for key in source.video_keys:
            prefix = f"videos/{key}"
            from_field = f"{prefix}/from_timestamp"
            to_field = f"{prefix}/to_timestamp"
            try:
                original_from = float(metadata[from_field])
                original_to = float(metadata[to_field])
                source_chunk = int(metadata[f"{prefix}/chunk_index"])
                source_file = int(metadata[f"{prefix}/file_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise CurationTransformError(
                    "Source v3 video segment metadata is missing or invalid"
                ) from exc
            if (
                not math.isfinite(original_from)
                or not math.isfinite(original_to)
                or original_from < 0
                or original_to <= original_from
                or source_chunk < 0
                or source_file < 0
            ):
                raise CurationTransformError(
                    "Source v3 video segment metadata is missing or invalid"
                )
            new_from = original_from + start / source.fps
            new_to = original_from + end / source.fps
            if new_to > original_to + (0.5 / source.fps):
                raise CurationTransformError(
                    "Stationary trim range exceeds its source video segment"
                )
            source_path, _ = source.video_source(source_index, key, metadata)
            identity = (key, source_path)
            location = video_locations.get(identity)
            if location is None:
                ordinal = next_video_file[key]
                next_video_file[key] += 1
                location = ordinal // 1000, ordinal % 1000
                video_locations[identity] = location
                destination = root / info["video_path"].format(
                    video_key=key,
                    chunk_index=location[0],
                    file_index=location[1],
                )
                copies.append((source_path, destination, source.root))
            row[f"{prefix}/chunk_index"] = location[0]
            row[f"{prefix}/file_index"] = location[1]
            row[from_field] = new_from
            row[to_field] = new_to
        metadata_rows.append(row)
        offset += len(data)
        report_progress(
            on_progress,
            stage="write",
            completed=index + 1,
            total=len(episodes),
            unit="episodes",
            current_item=f"에피소드 {index}",
        )
    metadata_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metadata_rows).to_parquet(metadata_path, index=False)
    info["total_videos"] = len(copies)
    write_json(root / "meta" / "info.json", info)
    _copy_videos(copies, on_progress)


def _write_episode_videos(
    source: DatasetSource,
    output_root: Path,
    output_index: int,
    metadata: dict[str, Any],
    trim_start: int,
    trim_end: int,
    expected_frames: int,
    *,
    output_version: str,
    output_video_codecs: dict[str, str],
    preserve_source_codec: bool,
    on_progress: ProgressCallback | None = None,
    progress_base: int = 0,
    progress_total: int = 0,
) -> int:
    completed = 0
    for key in source.video_keys:
        source_path, segment_start = source.video_source(
            int(metadata["_source_episode_index"]), key, metadata
        )
        if not source_path.is_file() or source_path.is_symlink():
            raise CurationTransformError("Source episode video is missing")
        if output_version == "v3.0":
            chunk, file_index = output_index // 1000, output_index % 1000
            destination = (
                output_root
                / f"videos/{key}/chunk-{chunk:03d}/file-{file_index:03d}.mp4"
            )
        else:
            template = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
            destination = safe_child(
                output_root,
                template.format(
                    episode_chunk=output_index
                    // int(source.info.get("chunks_size", 1000)),
                    episode_index=output_index,
                    video_key=key,
                ),
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        slice_video(
            source_path,
            destination,
            segment_start + trim_start,
            segment_start + trim_end,
            source.fps,
            expected_frames,
            codec=output_video_codecs[key],
            expected_source_codec=(
                output_video_codecs[key] if preserve_source_codec else None
            ),
            on_progress=on_progress,
            progress_base=progress_base + completed,
            progress_total=progress_total,
            current_item=f"에피소드 {output_index} · 카메라 {key}",
        )
        completed += expected_frames * 2
    return completed


def _output_video_codecs(
    source: DatasetSource,
    episodes: list[tuple[pd.DataFrame, dict[str, Any], int, int]],
    *,
    policy: str,
) -> dict[str, str]:
    if policy not in {"h264", "source"}:
        raise CurationTransformError("Unsupported video codec policy")
    if policy == "h264":
        return {key: "h264" for key in source.video_keys}

    codecs: dict[str, set[str]] = {key: set() for key in source.video_keys}
    inspected: dict[Path, str] = {}
    for _, metadata, _, _ in episodes:
        source_index = int(metadata["_source_episode_index"])
        for key in source.video_keys:
            path, _ = source.video_source(source_index, key, metadata)
            actual = inspected.get(path)
            if actual is None:
                actual = probe_video_codec(path)
                inspected[path] = actual
            codecs[key].add(actual)
    resolved: dict[str, str] = {}
    for key, values in codecs.items():
        if len(values) != 1:
            raise CurationTransformError(
                f"Source videos use inconsistent codecs for {key}"
            )
        codec = next(iter(values))
        video_encoder(codec)
        resolved[key] = codec
    return resolved


def _remap_tasks(
    episodes: list[tuple[pd.DataFrame, dict[str, Any], int, int]],
    source_tasks: dict[int, str],
) -> tuple[dict[int, int], list[dict[str, Any]]]:
    used = sorted(
        {
            int(value)
            for data, _, _, _ in episodes
            if "task_index" in data.columns
            for value in data["task_index"].dropna().unique()
        }
    )
    missing = sorted(set(used) - set(source_tasks))
    if missing:
        raise CurationTransformError(
            f"Task metadata is missing referenced indices: {missing}"
        )
    mapping = {old: new for new, old in enumerate(used)}
    tasks = [{"task_index": mapping[old], "task": source_tasks[old]} for old in used]
    return mapping, tasks


def _matrix_column(data: pd.DataFrame, name: str) -> np.ndarray:
    if name not in data.columns:
        raise CurationTransformError(f"Trim feature is missing: {name}")
    values = [np.asarray(value, dtype=np.float64).reshape(-1) for value in data[name]]
    if not values or len({len(value) for value in values}) != 1:
        raise CurationTransformError(f"Trim feature has inconsistent shape: {name}")
    return np.stack(values)


def _feature_names(info: dict[str, Any], key: str, width: int) -> list[str]:
    names: Any = info.get("features", {}).get(key, {}).get("names")
    while isinstance(names, dict) and names:
        names = next(iter(names.values()))
    if (
        isinstance(names, list)
        and len(names) == width
        and all(isinstance(item, str) for item in names)
    ):
        return names
    return [str(index) for index in range(width)]


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
