from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from datasetui.config import Settings
from datasetui.database import Database, RecipeRevisionMismatchError
from datasetui.datasets import MAX_INFO_BYTES, inspect_dataset, scan_storage_area
from datasetui.transform_errors import CurationTransformError


MAX_METADATA_BYTES = 256 * 1024 * 1024
LANGUAGE_PERSISTENT = "language_persistent"
LANGUAGE_EVENTS = "language_events"
PERSISTENT_STYLES = {"task_aug", "subtask", "plan", "memory"}
EVENT_STYLES = {"interjection", "vqa"}
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
    snapshot = database.get_curation_snapshot(payload["snapshot_id"])
    if (
        snapshot["missing_since"] is not None
        or snapshot["readiness"] != "ready"
        or snapshot["fingerprint"] != snapshot["dataset_fingerprint"]
    ):
        raise RecipeRevisionMismatchError(snapshot["dataset_id"])

    source_root = _safe_dataset_root(
        settings.nas_root,
        snapshot["storage_area"],
        snapshot["relative_path"],
    )
    raw_info = _read_regular_bytes(
        source_root / "meta" / "info.json", max_bytes=MAX_INFO_BYTES
    )
    if hashlib.sha256(raw_info).hexdigest() != snapshot["dataset_fingerprint"]:
        raise RecipeRevisionMismatchError(snapshot["dataset_id"])
    info = _decode_json_object(raw_info)
    version = info.get("codebase_version")
    if version not in {"v2.0", "v2.1", "v3.0"}:
        raise CurationTransformError("Unsupported source dataset version")

    outputs = _output_selections(snapshot, payload["output_name"], job_id)
    completed = _reuse_published_outputs(
        settings=settings, job_id=job_id, outputs=outputs
    )
    if completed is not None:
        _refresh_derived_registry(database, settings)
        return completed

    staging_parent = settings.staging_root / "curation"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f"{job_id}-", dir=staging_parent))
    published: list[dict[str, Any]] = []
    try:
        source = _DatasetSource(source_root, info)
        annotations = (
            database.get_curation_snapshot_annotations(snapshot["id"])
            if snapshot["include_annotations"]
            else {}
        )
        for output in outputs:
            destination = staging_root / output["name"]
            built = _write_dataset(
                source=source,
                destination=destination,
                source_indices=output["episodes"],
                trim_config=snapshot["trim_config"],
                annotations=annotations,
                relative_action=snapshot["relative_action"],
            )
            lineage = built["lineage"]
            candidate = inspect_dataset(
                area_root=staging_root,
                storage_area="derived",
                relative_path=output["name"],
            )
            if candidate.readiness != "ready":
                raise CurationTransformError(
                    "Derived dataset failed structural validation"
                )
            database.assert_job_lease(job_id, worker_id=worker_id)
            manifest = _publish_output(
                database=database,
                settings=settings,
                job_id=job_id,
                worker_id=worker_id,
                staging_path=destination,
                output_name=output["name"],
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
                }
            )

        result = {
            "snapshot_id": snapshot["id"],
            "source_dataset_id": snapshot["dataset_id"],
            "source_fingerprint": snapshot["dataset_fingerprint"],
            "operation": snapshot["operation"],
            "outputs": published,
            "reused": False,
        }
        _write_run_manifest(settings, job_id, result)
        _refresh_derived_registry(database, settings)
        return result
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def _output_selections(
    snapshot: dict[str, Any], base_name: str, job_id: str
) -> list[dict[str, Any]]:
    suffix = job_id.split("-", 1)[0]
    selected = list(snapshot["selected_episode_indices"])
    flagged = set(snapshot["flagged_episode_indices"])
    if snapshot["operation"] == "train_eval_split":
        train = [index for index in selected if index not in flagged]
        evaluation = [index for index in selected if index in flagged]
        if not train or not evaluation:
            raise CurationTransformError(
                "Train and eval outputs must both be non-empty"
            )
        return [
            {
                "role": "train",
                "name": f"{base_name}_train--{suffix}",
                "episodes": train,
            },
            {
                "role": "eval",
                "name": f"{base_name}_eval--{suffix}",
                "episodes": evaluation,
            },
        ]
    if not selected:
        raise CurationTransformError("Derived output cannot be empty")
    role = "delete_flagged" if snapshot["operation"] == "delete_flagged" else "subset"
    return [{"role": role, "name": f"{base_name}--{suffix}", "episodes": selected}]


class _DatasetSource:
    def __init__(self, root: Path, info: dict[str, Any]):
        self.root = root
        self.info = info
        self.version = str(info["codebase_version"])
        self.fps = float(info["fps"])
        self.video_keys = [
            key
            for key, value in info.get("features", {}).items()
            if isinstance(value, dict) and value.get("dtype") == "video"
        ]
        if any(
            key in {"", ".", ".."} or "/" in key or "\\" in key
            for key in self.video_keys
        ):
            raise CurationTransformError(
                "Dataset contains an unsafe video feature name"
            )
        self.tasks = self._load_tasks()
        self.episode_metadata = self._load_episode_metadata()

    def _load_tasks(self) -> dict[int, str]:
        if self.version == "v3.0":
            path = self.root / "meta" / "tasks.parquet"
            if not path.is_file():
                return {}
            _require_regular_file(path)
            frame = _read_parquet(path)
            return {
                int(row["task_index"]): str(row.get("task", row.get("name", "")))
                for row in frame.to_dict("records")
            }
        path = self.root / "meta" / "tasks.jsonl"
        if not path.is_file():
            return {}
        _require_regular_file(path)
        return {
            int(row["task_index"]): str(row.get("task", row.get("name", "")))
            for row in _read_json_lines(path)
        }

    def _load_episode_metadata(self) -> dict[int, dict[str, Any]]:
        if self.version != "v3.0":
            path = self.root / "meta" / "episodes.jsonl"
            return {
                int(row["episode_index"]): row
                for row in (_read_json_lines(path) if path.is_file() else [])
            }
        rows: list[dict[str, Any]] = []
        for path in _safe_parquet_files(self.root / "meta" / "episodes"):
            rows.extend(_read_parquet(path).to_dict("records"))
        return {int(row["episode_index"]): row for row in rows}

    def episode(self, episode_index: int) -> tuple[pd.DataFrame, dict[str, Any]]:
        metadata = self.episode_metadata.get(episode_index, {})
        if self.version == "v3.0":
            chunk = int(metadata.get("data/chunk_index", 0))
            file_index = int(metadata.get("data/file_index", 0))
            path = self.root / f"data/chunk-{chunk:03d}/file-{file_index:03d}.parquet"
            _require_regular_file(path)
            data = _read_parquet(path)
            data = data[data["episode_index"] == episode_index].copy()
        else:
            chunk_size = int(self.info.get("chunks_size", 1000))
            template = self.info.get(
                "data_path",
                "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            )
            relative = template.format(
                episode_chunk=episode_index // chunk_size,
                episode_index=episode_index,
            )
            path = _safe_child(self.root, relative)
            _require_regular_file(path)
            data = _read_parquet(path)
        if data.empty:
            raise CurationTransformError(f"Episode {episode_index} has no frames")
        return data.reset_index(drop=True), metadata

    def video_source(
        self, episode_index: int, video_key: str, metadata: dict[str, Any]
    ) -> tuple[Path, int]:
        if self.version == "v3.0":
            chunk = int(metadata.get(f"videos/{video_key}/chunk_index", 0))
            file_index = int(metadata.get(f"videos/{video_key}/file_index", 0))
            start = int(
                round(
                    float(metadata.get(f"videos/{video_key}/from_timestamp", 0))
                    * self.fps
                )
            )
            return (
                _safe_child(
                    self.root,
                    f"videos/{video_key}/chunk-{chunk:03d}/file-{file_index:03d}.mp4",
                ),
                start,
            )
        chunk_size = int(self.info.get("chunks_size", 1000))
        template = self.info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
        relative = template.format(
            episode_chunk=episode_index // chunk_size,
            episode_index=episode_index,
            video_key=video_key,
        )
        return _safe_child(self.root, relative), 0


def _write_dataset(
    *,
    source: _DatasetSource,
    destination: Path,
    source_indices: list[int],
    trim_config: dict[str, Any],
    annotations: dict[int, dict[str, Any]],
    relative_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
        trimmed["frame_index"] = np.arange(len(trimmed), dtype=np.int64)
        trimmed["timestamp"] = np.arange(len(trimmed), dtype=np.float64) / source.fps
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

    task_mapping, tasks = _remap_tasks(
        episodes, _apply_task_overrides(episodes, source.tasks)
    )
    for data, _, _, _ in episodes:
        if "task_index" in data.columns:
            data["task_index"] = data["task_index"].map(task_mapping).astype("int64")

    language_types = _language_column_types(episodes)
    if source.version == "v3.0":
        _write_v3(source, destination, episodes, tasks, language_types)
    else:
        _write_v2(source, destination, episodes, tasks, language_types)
    _write_stats(destination / "meta" / "stats.json", [item[0] for item in episodes])
    return {
        "lineage": lineage,
        "relative_action": _relative_action_profile(
            source.info, [item[0] for item in episodes], relative_action or {}
        ),
    }


def _relative_action_profile(
    info: dict[str, Any],
    episodes: list[pd.DataFrame],
    config: dict[str, Any],
) -> dict[str, Any]:
    if not config.get("enabled", False):
        return {"enabled": False, "dimensions": [], "statistics": {}}
    action_names = _feature_names(
        info, "action", _matrix_column(episodes[0], "action").shape[1]
    )
    state_names = _feature_names(
        info,
        "observation.state",
        _matrix_column(episodes[0], "observation.state").shape[1],
    )
    dimensions = list(config.get("dimensions") or [])
    for name in dimensions:
        if name not in action_names or name not in state_names:
            raise CurationTransformError(
                "A relative action dimension is not shared by action and state"
            )
        if action_names.index(name) != state_names.index(name):
            raise CurationTransformError(
                "Relative action requires matching action/state dimension order"
            )
    values: list[np.ndarray] = []
    for data in episodes:
        action = _matrix_column(data, "action").astype(np.float64)
        state = _matrix_column(data, "observation.state").astype(np.float64)
        values.append(
            np.column_stack(
                [action[:, action_names.index(name)] - state[:, state_names.index(name)] for name in dimensions]
            )
        )
    combined = np.concatenate(values, axis=0)
    if not np.isfinite(combined).all():
        raise CurationTransformError("Relative action contains non-finite values")
    statistics = {
        name: {
            "min": float(np.min(combined[:, index])),
            "max": float(np.max(combined[:, index])),
            "mean": float(np.mean(combined[:, index])),
            "std": float(np.std(combined[:, index])),
        }
        for index, name in enumerate(dimensions)
    }
    return {
        "enabled": True,
        "dimensions": dimensions,
        "absolute_dimensions": [name for name in action_names if name not in dimensions],
        "formula": "action[i] - observation.state[i]",
        "stored_action": "absolute",
        "statistics": statistics,
    }


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
    if tool_calls is not None and not isinstance(tool_calls, list):
        tool_calls = [tool_calls]
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
    seen: set[str] = set()

    def append_many(values: Any, *, timestamp: float | None = None) -> None:
        if values is None:
            return
        if isinstance(values, np.ndarray):
            values = values.tolist()
        if not isinstance(values, list):
            return
        for value in values:
            atom = _coerce_atom(value, fallback_timestamp=timestamp)
            if atom is None:
                continue
            key = json.dumps(atom, ensure_ascii=False, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                atoms.append(atom)

    persistent_loaded = False
    for _, row in data.iterrows():
        if LANGUAGE_PERSISTENT in data.columns and not persistent_loaded:
            append_many(row[LANGUAGE_PERSISTENT])
            persistent_loaded = True
        if LANGUAGE_EVENTS in data.columns:
            append_many(row[LANGUAGE_EVENTS], timestamp=float(row["timestamp"]))
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


def _language_column_types(
    episodes: list[tuple[pd.DataFrame, dict[str, Any], int, int]],
) -> dict[str, pa.DataType]:
    tool_call_type = pa.struct(
        [
            pa.field("type", pa.string()),
            pa.field(
                "function",
                pa.struct(
                    [
                        pa.field("name", pa.string()),
                        pa.field(
                            "arguments", pa.struct([pa.field("text", pa.string())])
                        ),
                    ]
                ),
            ),
        ]
    )
    common = [
        pa.field("role", pa.string(), nullable=False),
        pa.field("content", pa.string()),
        pa.field("style", pa.string()),
    ]
    persistent_type = pa.list_(
        pa.struct(
            [
                *common,
                pa.field("timestamp", pa.float32(), nullable=False),
                pa.field("camera", pa.string()),
                pa.field("tool_calls", pa.list_(tool_call_type)),
            ]
        )
    )
    event_type = pa.list_(
        pa.struct(
            [
                *common,
                pa.field("camera", pa.string()),
                pa.field("tool_calls", pa.list_(tool_call_type)),
            ]
        )
    )
    present = {
        name
        for data, _, _, _ in episodes
        for name in (LANGUAGE_PERSISTENT, LANGUAGE_EVENTS)
        if name in data.columns
    }
    return {
        name: persistent_type if name == LANGUAGE_PERSISTENT else event_type
        for name in present
    }


def _updated_info(
    source_info: dict[str, Any],
    episodes: list[tuple[pd.DataFrame, dict[str, Any], int, int]],
    language_types: dict[str, pa.DataType],
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


def _write_parquet(
    data: pd.DataFrame, path: Path, language_types: dict[str, pa.DataType]
) -> None:
    language_columns = [name for name in language_types if name in data.columns]
    table = pa.Table.from_pandas(
        data.drop(columns=language_columns), preserve_index=False
    )
    for name in language_columns:
        table = table.append_column(
            name, pa.array(data[name].tolist(), type=language_types[name])
        )
    pq.write_table(table, path)


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
    hold = max(1, int(math.ceil(float(config.get("hold_time_s", 0.5)) * fps)))
    runs = _true_runs(active, hold)
    if not runs:
        return 0, len(data), "no_sustained_motion"
    margin = max(0, int(round(float(config.get("margin_s", 1.0)) * fps)))
    start = max(0, runs[0][0] + 1 - margin)
    end = min(len(data), runs[-1][1] + 2 + margin)
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
    source: _DatasetSource,
    root: Path,
    episodes: list[tuple[pd.DataFrame, dict[str, Any], int, int]],
    tasks: list[dict[str, Any]],
    language_types: dict[str, pa.DataType],
) -> None:
    info = _updated_info(source.info, episodes, language_types)
    info.update(
        total_episodes=len(episodes),
        total_frames=sum(len(item[0]) for item in episodes),
        total_tasks=len(tasks),
        splits={"train": f"0:{len(episodes)}"},
    )
    _write_json(root / "meta" / "info.json", info)
    _write_json_lines(root / "meta" / "tasks.jsonl", tasks)
    episode_rows = []
    chunk_size = int(info.get("chunks_size", 1000))
    template = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    for index, (data, metadata, start, end) in enumerate(episodes):
        relative = template.format(
            episode_chunk=index // chunk_size, episode_index=index
        )
        path = _safe_child(root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_parquet(data, path, language_types)
        episode_rows.append(
            {
                "episode_index": index,
                "tasks": _episode_task_names(data, tasks),
                "length": len(data),
            }
        )
        _write_episode_videos(source, root, index, metadata, start, end, len(data))
    _write_json_lines(root / "meta" / "episodes.jsonl", episode_rows)


def _write_v3(
    source: _DatasetSource,
    root: Path,
    episodes: list[tuple[pd.DataFrame, dict[str, Any], int, int]],
    tasks: list[dict[str, Any]],
    language_types: dict[str, pa.DataType],
) -> None:
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
    _write_json(root / "meta" / "info.json", info)
    pd.DataFrame(tasks).to_parquet(root / "meta" / "tasks.parquet", index=False)
    metadata_rows: list[dict[str, Any]] = []
    offset = 0
    for index, (data, metadata, start, end) in enumerate(episodes):
        chunk = index // 1000
        file_index = index % 1000
        path = root / f"data/chunk-{chunk:03d}/file-{file_index:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_parquet(data, path, language_types)
        row: dict[str, Any] = {
            "episode_index": index,
            "tasks": _episode_task_names(data, tasks),
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
        _write_episode_videos(source, root, index, metadata, start, end, len(data))
    metadata_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metadata_rows).to_parquet(metadata_path, index=False)


def _write_episode_videos(
    source: _DatasetSource,
    output_root: Path,
    output_index: int,
    metadata: dict[str, Any],
    trim_start: int,
    trim_end: int,
    expected_frames: int,
) -> None:
    for key in source.video_keys:
        source_path, segment_start = source.video_source(
            int(metadata["_source_episode_index"]), key, metadata
        )
        if not source_path.is_file() or source_path.is_symlink():
            raise CurationTransformError("Source episode video is missing")
        if source.version == "v3.0":
            chunk, file_index = output_index // 1000, output_index % 1000
            destination = (
                output_root
                / f"videos/{key}/chunk-{chunk:03d}/file-{file_index:03d}.mp4"
            )
        else:
            template = source.info.get(
                "video_path",
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            )
            destination = _safe_child(
                output_root,
                template.format(
                    episode_chunk=output_index
                    // int(source.info.get("chunks_size", 1000)),
                    episode_index=output_index,
                    video_key=key,
                ),
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        _slice_video(
            source_path,
            destination,
            segment_start + trim_start,
            segment_start + trim_end,
            source.fps,
            expected_frames,
        )


def _slice_video(
    source: Path,
    destination: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    expected_frames: int,
) -> None:
    try:
        import av
    except ImportError as exc:
        raise CurationTransformError("Video transform support is unavailable") from exc
    try:
        descriptor = os.open(source, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise CurationTransformError("Source episode video is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise CurationTransformError("Source episode video is unavailable")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            input_container = av.open(stream)
            try:
                frames = [
                    frame
                    for index, frame in enumerate(input_container.decode(video=0))
                    if start_frame <= index < end_frame
                ]
            finally:
                input_container.close()
    finally:
        os.close(descriptor)
    if len(frames) != expected_frames:
        raise CurationTransformError("Video and data frame counts do not match")
    output = av.open(str(destination), mode="w")
    try:
        stream = output.add_stream("libx264", rate=fps)
        stream.width = frames[0].width
        stream.height = frames[0].height
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    finally:
        output.close()


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
    mapping = {old: new for new, old in enumerate(used)}
    tasks = [
        {"task_index": mapping[old], "task": source_tasks.get(old, f"task_{old}")}
        for old in used
    ]
    return mapping, tasks


def _episode_task_names(data: pd.DataFrame, tasks: list[dict[str, Any]]) -> list[str]:
    by_index = {row["task_index"]: row["task"] for row in tasks}
    if "task_index" not in data.columns:
        return []
    return [by_index[index] for index in sorted(set(data["task_index"].astype(int)))]


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


def _write_stats(path: Path, episodes: list[pd.DataFrame]) -> None:
    combined = pd.concat(episodes, ignore_index=True)
    stats: dict[str, Any] = {}
    for name in combined.columns:
        try:
            matrix = np.stack(
                [
                    np.asarray(value, dtype=np.float64).reshape(-1)
                    for value in combined[name]
                ]
            )
        except (TypeError, ValueError):
            continue
        if matrix.size == 0 or not np.isfinite(matrix).all():
            continue
        stats[name] = {
            "min": np.min(matrix, axis=0).tolist(),
            "max": np.max(matrix, axis=0).tolist(),
            "mean": np.mean(matrix, axis=0).tolist(),
            "std": np.std(matrix, axis=0).tolist(),
            "q01": np.quantile(matrix, 0.01, axis=0).tolist(),
            "q10": np.quantile(matrix, 0.10, axis=0).tolist(),
            "q50": np.quantile(matrix, 0.50, axis=0).tolist(),
            "q90": np.quantile(matrix, 0.90, axis=0).tolist(),
            "q99": np.quantile(matrix, 0.99, axis=0).tolist(),
            "count": [len(matrix)] * matrix.shape[1],
        }
    _write_json(path, stats)


def _publish_output(
    *,
    database: Database,
    settings: Settings,
    job_id: str,
    worker_id: str,
    staging_path: Path,
    output_name: str,
) -> dict[str, Any]:
    derived = _real_directory(settings.nas_root / "derived")
    final = derived / output_name
    source_manifest = _tree_manifest(staging_path)
    if final.exists():
        if (
            final.is_symlink()
            or not final.is_dir()
            or _tree_manifest(final) != source_manifest
        ):
            raise CurationTransformError("Derived output name already exists")
        return source_manifest
    incoming = derived / f".incoming-{job_id}-{uuid.uuid4().hex}"
    try:
        shutil.copytree(staging_path, incoming)
        if _tree_manifest(incoming) != source_manifest:
            raise CurationTransformError("Derived output copy verification failed")
        database.assert_job_lease(job_id, worker_id=worker_id)
        incoming.rename(final)
        return source_manifest
    finally:
        shutil.rmtree(incoming, ignore_errors=True)


def _reuse_published_outputs(
    *, settings: Settings, job_id: str, outputs: list[dict[str, Any]]
) -> dict[str, Any] | None:
    path = settings.nas_root / "manifests" / "curation" / f"{job_id}.json"
    if not path.is_file() or path.is_symlink():
        return None
    result = _read_json(path)
    expected = {item["name"] for item in outputs}
    actual = {item.get("name") for item in result.get("outputs", [])}
    if expected != actual:
        raise CurationTransformError(
            "Curation manifest conflicts with the requested output"
        )
    for item in result["outputs"]:
        root = settings.nas_root / "derived" / item["relative_path"]
        if not root.is_dir() or root.is_symlink():
            return None
        if _tree_manifest(root)["tree_sha256"] != item["manifest_sha256"]:
            raise CurationTransformError("Published derived dataset was modified")
    return {**result, "reused": True}


def _write_run_manifest(
    settings: Settings, job_id: str, result: dict[str, Any]
) -> None:
    manifests = _real_directory(settings.nas_root / "manifests")
    root = manifests / "curation"
    root.mkdir(exist_ok=True)
    root = _real_directory(root)
    _write_json_atomic(root / f"{job_id}.json", {"schema_version": 1, **result})


def _refresh_derived_registry(database: Database, settings: Settings) -> None:
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


def _tree_manifest(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    total = 0
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories.sort()
        for name in directories:
            if (current_path / name).is_symlink():
                raise CurationTransformError("Derived dataset contains a symlink")
        for name in sorted(files):
            path = current_path / name
            descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            file_digest = hashlib.sha256()
            size = 0
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise CurationTransformError(
                        "Derived dataset contains an unsafe file"
                    )
                while chunk := os.read(descriptor, 1024 * 1024):
                    file_digest.update(chunk)
                    size += len(chunk)
            finally:
                os.close(descriptor)
            relative = path.relative_to(root).as_posix()
            digest.update(f"{relative}\0{size}\0{file_digest.hexdigest()}\n".encode())
            count += 1
            total += size
    return {
        "tree_sha256": digest.hexdigest(),
        "file_count": count,
        "total_bytes": total,
    }


def _safe_dataset_root(nas_root: Path, storage_area: str, relative_path: str) -> Path:
    root = _real_directory(nas_root / storage_area)
    current = root
    for component in Path(relative_path).parts:
        if component in {"", ".", ".."}:
            raise CurationTransformError("Unsafe dataset path")
        current = current / component
        if current.is_symlink() or not current.is_dir():
            raise CurationTransformError("Dataset source is unavailable")
    return current


def _safe_child(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise CurationTransformError("Unsafe dataset file path")
    return root / path


def _real_directory(path: Path) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise CurationTransformError("Dataset storage is unavailable")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    value = _decode_json_object(_read_regular_bytes(path, max_bytes=MAX_METADATA_BYTES))
    return value


def _decode_json_object(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CurationTransformError("Dataset metadata is invalid") from exc
    if not isinstance(value, dict):
        raise CurationTransformError("Dataset metadata is invalid")
    return value


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    try:
        text = _read_regular_bytes(path, max_bytes=MAX_METADATA_BYTES).decode("utf-8")
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CurationTransformError("Dataset metadata is invalid") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise CurationTransformError("Dataset metadata is invalid")
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )


def _write_json_lines(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        _write_json(temporary, value)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_regular_file(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CurationTransformError("Dataset file is unavailable") from exc
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise CurationTransformError("Dataset contains an unsafe file entry")


def _read_regular_bytes(path: Path, *, max_bytes: int) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise CurationTransformError("Dataset file is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > max_bytes:
            raise CurationTransformError("Dataset metadata is invalid")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > max_bytes:
            raise CurationTransformError("Dataset metadata is invalid")
        return raw
    finally:
        os.close(descriptor)


def _read_parquet(path: Path) -> pd.DataFrame:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise CurationTransformError("Dataset file is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise CurationTransformError("Dataset contains an unsafe file entry")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return pd.read_parquet(stream)
    finally:
        os.close(descriptor)


def _safe_parquet_files(root: Path) -> list[Path]:
    if root.is_symlink() or not root.is_dir():
        raise CurationTransformError("Dataset episode metadata is unavailable")
    files: list[Path] = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            if (current_path / name).is_symlink():
                raise CurationTransformError(
                    "Dataset contains an unsafe directory entry"
                )
        for name in names:
            if not name.endswith(".parquet"):
                continue
            path = current_path / name
            _require_regular_file(path)
            files.append(path)
    return sorted(files)
