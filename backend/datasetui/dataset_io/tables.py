"""Parquet I/O, task rows and language-annotation column types."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from datasetui.dataset_io.files import require_regular_file
from datasetui.transform_errors import CurationTransformError


LANGUAGE_PERSISTENT = "language_persistent"


LANGUAGE_EVENTS = "language_events"


def task_rows_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Preserve v3 task text stored either as a column or a string index."""
    rows = frame.to_dict("records")
    description_column = next(
        (name for name in ("task", "name") if name in frame.columns), None
    )
    if description_column is not None:
        return rows
    for row, description in zip(rows, frame.index, strict=True):
        row["task"] = description if isinstance(description, str) else None
    return rows


def language_column_types(
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


def write_parquet(
    data: pd.DataFrame, path: Path, language_types: dict[str, pa.DataType]
) -> None:
    language_columns = [name for name in language_types if name in data.columns]
    ordinary = data.drop(columns=language_columns)
    ordinary.attrs = {}
    table = pa.Table.from_pandas(ordinary, preserve_index=False)
    for name, source_type in data.attrs.get("datasetui_arrow_types", {}).items():
        if name not in table.column_names or name in language_columns:
            continue
        column_index = table.schema.get_field_index(name)
        column = table.column(name)
        if column.type != source_type:
            table = table.set_column(
                column_index, name, column.cast(source_type, safe=True)
            )
    for name in language_columns:
        table = table.append_column(
            name, pa.array(data[name].tolist(), type=language_types[name])
        )
    pq.write_table(table, path)


def write_v3_tasks(root: Path, tasks: list[dict[str, Any]]) -> None:
    from datasetui.official.operations import enabled

    frame = pd.DataFrame(tasks, columns=["task_index", "task"]).set_index("task")
    if enabled():
        from datasetui.official.runtime import require_runtime

        require_runtime()
        from lerobot.datasets.io_utils import write_tasks

        write_tasks(frame, root)
    else:
        frame.to_parquet(root / "meta/tasks.parquet")


def episode_task_names(data: pd.DataFrame, tasks: list[dict[str, Any]]) -> list[str]:
    by_index = {row["task_index"]: row["task"] for row in tasks}
    if "task_index" not in data.columns:
        return []
    return [by_index[index] for index in sorted(set(data["task_index"].astype(int)))]


def read_parquet(path: Path) -> pd.DataFrame:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise CurationTransformError("Dataset file is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise CurationTransformError("Dataset contains an unsafe file entry")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            table = pq.read_table(stream)
            frame = table.to_pandas()
            frame.attrs["datasetui_arrow_types"] = {
                field.name: field.type for field in table.schema
            }
            return frame
    finally:
        os.close(descriptor)


def safe_parquet_files(root: Path) -> list[Path]:
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
            require_regular_file(path)
            files.append(path)
    return sorted(files)
