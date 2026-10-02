"""Language-annotation columns (persistent/events) and task overrides."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import pandas as pd

from datasetui.dataset_io.tables import (
    LANGUAGE_EVENTS,
    LANGUAGE_PERSISTENT,
)
from datasetui.transform_errors import CurationTransformError

PERSISTENT_STYLES = {"task_aug", "subtask", "plan", "memory"}


EVENT_STYLES = {"interjection", "vqa"}


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


def extract_existing_language_atoms(data: pd.DataFrame) -> list[dict[str, Any]]:
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


def replace_language_columns(
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


def apply_task_overrides(
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


def remap_tasks(
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
