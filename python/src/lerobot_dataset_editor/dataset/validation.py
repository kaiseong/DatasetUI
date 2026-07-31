"""Explicit structural validation for supported LeRobot dataset layouts.

Unlike :func:`load_document`, validation may read selected metadata and timestamp
columns. It never mutates the source dataset.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .document import ValidationResult
from .version import DatasetVersion, detect_version, uses_legacy_episode_layout

_EPISODE_IN_NAME = re.compile(r"episode[_-](\d+)")


def _check_episodes(
    episodes: list[dict[str, Any]], info: dict[str, Any]
) -> list[str]:
    """Require unique, contiguous zero-based episodes and declared totals."""
    errors: list[str] = []
    indices: list[int] = []
    for episode in episodes:
        raw_index = episode.get("episode_index", episode.get("index"))
        if not isinstance(raw_index, int) or isinstance(raw_index, bool) or raw_index < 0:
            errors.append(f"Invalid episode index: {raw_index!r}")
            continue
        indices.append(raw_index)

    seen: set[int] = set()
    for index in indices:
        if index in seen:
            errors.append(f"Duplicate episode index: {index}")
        seen.add(index)

    expected_indices = list(range(len(seen)))
    actual_indices = sorted(seen)
    if actual_indices != expected_indices:
        missing = sorted(set(expected_indices) - seen)
        if actual_indices and actual_indices[0] != 0:
            errors.append(
                f"Episode indices must start at 0; first index is {actual_indices[0]}"
            )
        if missing:
            errors.append(f"Missing episode indices (gap in sequence): {missing}")

    declared_episodes = info.get("total_episodes")
    if isinstance(declared_episodes, int) and not isinstance(declared_episodes, bool):
        if len(episodes) != declared_episodes:
            errors.append(
                "total_episodes mismatch: "
                f"info.json declares {declared_episodes}, metadata contains {len(episodes)}"
            )

    lengths = [episode.get("length") for episode in episodes]
    if lengths and all(isinstance(length, int) and length >= 0 for length in lengths):
        declared_frames = info.get("total_frames")
        actual_frames = sum(lengths)
        if isinstance(declared_frames, int) and actual_frames != declared_frames:
            errors.append(
                "total_frames mismatch: "
                f"info.json declares {declared_frames}, episodes contain {actual_frames}"
            )
    return errors


def _load_episodes_for_validation(
    dataset_path: Path, version: DatasetVersion
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load only episode metadata, reporting malformed metadata as validation errors."""
    errors: list[str] = []
    if uses_legacy_episode_layout(version):
        episodes_path = dataset_path / "meta" / "episodes.jsonl"
        if not episodes_path.is_file():
            return [], ["Missing episode metadata: meta/episodes.jsonl"]
        episodes: list[dict[str, Any]] = []
        try:
            with episodes_path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if line.strip():
                        value = json.loads(line)
                        if not isinstance(value, dict):
                            raise ValueError("episode entry is not an object")
                        episodes.append(value)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"Invalid episode metadata meta/episodes.jsonl: {exc}")
        return episodes, errors

    episodes_dir = dataset_path / "meta" / "episodes"
    if not episodes_dir.is_dir():
        return [], ["Missing episode metadata: meta/episodes"]
    episodes = []
    for parquet_path in sorted(episodes_dir.rglob("*.parquet")):
        try:
            parquet = pq.ParquetFile(parquet_path)
            for row_group_index in range(parquet.metadata.num_row_groups):
                table = parquet.read_row_group(row_group_index)
                for row_index in range(table.num_rows):
                    episodes.append(
                        {
                            column: table.column(column)[row_index].as_py()
                            for column in table.column_names
                        }
                    )
        except Exception:
            # The common footer check emits the precise source-relative error.
            continue
    return episodes, errors


def _parquet_paths(dataset_path: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for directory in (dataset_path / "meta", dataset_path / "data"):
        if directory.is_dir():
            paths.extend(path for path in directory.rglob("*.parquet") if path.is_file())
    return tuple(sorted(paths))


def _check_parquet_corrupt(dataset_path: Path) -> list[str]:
    """Check every metadata and data Parquet footer without decoding frame data."""
    errors: list[str] = []
    for parquet_path in _parquet_paths(dataset_path):
        try:
            pq.read_metadata(parquet_path)
        except Exception as exc:
            errors.append(
                f"Corrupt Parquet file {parquet_path.relative_to(dataset_path)}: {exc}"
            )
    return errors


def _check_data_row_total(dataset_path: Path, info: dict[str, Any]) -> list[str]:
    """Compare declared total_frames with all readable data footer row counts."""
    declared = info.get("total_frames")
    if not isinstance(declared, int) or isinstance(declared, bool):
        return []
    data_directory = dataset_path / "data"
    paths = sorted(data_directory.rglob("*.parquet")) if data_directory.is_dir() else []
    actual = 0
    for parquet_path in paths:
        try:
            actual += pq.read_metadata(parquet_path).num_rows
        except Exception:
            continue
    if actual != declared:
        return [f"data rows mismatch: Parquet contains {actual}, total_frames declares {declared}"]
    return []


def _leaf_arrow_type(arrow_type: pa.DataType) -> pa.DataType:
    while (
        pa.types.is_list(arrow_type)
        or pa.types.is_large_list(arrow_type)
        or pa.types.is_fixed_size_list(arrow_type)
    ):
        arrow_type = arrow_type.value_type
    return arrow_type


def _matches_declared_primitive(arrow_type: pa.DataType, dtype: str) -> bool | None:
    """Return None for logical media/language or unknown extension dtypes."""
    leaf = _leaf_arrow_type(arrow_type)
    expected: dict[str, pa.DataType] = {
        "float16": pa.float16(),
        "float32": pa.float32(),
        "float64": pa.float64(),
        "int8": pa.int8(),
        "int16": pa.int16(),
        "int32": pa.int32(),
        "int64": pa.int64(),
        "uint8": pa.uint8(),
        "uint16": pa.uint16(),
        "uint32": pa.uint32(),
        "uint64": pa.uint64(),
        "bool": pa.bool_(),
    }
    if dtype == "string":
        return pa.types.is_string(leaf) or pa.types.is_large_string(leaf)
    if dtype not in expected:
        return None
    return leaf.equals(expected[dtype])


def _check_schema_drift(dataset_path: Path, info: dict[str, Any]) -> list[str]:
    """Check every readable data file against declared feature columns."""
    features = info.get("features", {})
    if not isinstance(features, dict) or not features:
        return []
    declared_columns = set(features)
    errors: list[str] = []
    data_directory = dataset_path / "data"
    if not data_directory.is_dir():
        return ["Missing data directory"]

    for parquet_path in sorted(data_directory.rglob("*.parquet")):
        try:
            schema = pq.read_schema(parquet_path)
            parquet_columns = set(schema.names)
        except Exception:
            continue
        relative_path = parquet_path.relative_to(dataset_path)
        missing = declared_columns - parquet_columns
        if missing:
            errors.append(
                f"Schema drift in {relative_path}: "
                f"info.json declares columns not in Parquet: {sorted(missing)}"
            )
        for column in sorted(declared_columns & parquet_columns):
            spec = features.get(column)
            if not isinstance(spec, dict) or not isinstance(spec.get("dtype"), str):
                continue
            arrow_type = schema.field(column).type
            matches = _matches_declared_primitive(arrow_type, spec["dtype"])
            if matches is False:
                errors.append(
                    f"Schema type drift in {relative_path}: {column} declares "
                    f"{spec['dtype']} but Parquet has {arrow_type}"
                )
    return errors


def _episode_from_path(path: Path) -> int:
    match = _EPISODE_IN_NAME.search(path.stem)
    return int(match.group(1)) if match else 0


def _check_timestamps(dataset_path: Path) -> list[str]:
    """Check finite, monotonic timestamps across row groups and data files."""
    errors: list[str] = []
    previous_by_episode: dict[int, float] = {}
    data_directory = dataset_path / "data"
    if not data_directory.is_dir():
        return errors

    for parquet_path in sorted(data_directory.rglob("*.parquet")):
        try:
            parquet = pq.ParquetFile(parquet_path)
            available = set(parquet.schema_arrow.names)
            if "timestamp" not in available:
                continue
            columns = ["timestamp"]
            if "episode_index" in available:
                columns.append("episode_index")
            for row_group_index in range(parquet.metadata.num_row_groups):
                table = parquet.read_row_group(row_group_index, columns=columns)
                timestamps = table.column("timestamp").to_pylist()
                episode_indices = (
                    table.column("episode_index").to_pylist()
                    if "episode_index" in table.column_names
                    else [_episode_from_path(parquet_path)] * len(timestamps)
                )
                for row_index, (raw_timestamp, raw_episode) in enumerate(
                    zip(timestamps, episode_indices)
                ):
                    episode = int(raw_episode)
                    try:
                        timestamp = float(raw_timestamp)
                    except (TypeError, ValueError):
                        errors.append(
                            f"Invalid timestamp in {parquet_path.relative_to(dataset_path)} "
                            f"row {row_index}: {raw_timestamp!r}"
                        )
                        continue
                    if not math.isfinite(timestamp):
                        errors.append(
                            f"Timestamp must be finite in "
                            f"{parquet_path.relative_to(dataset_path)} row {row_index}: "
                            f"{raw_timestamp!r}"
                        )
                        continue
                    previous = previous_by_episode.get(episode)
                    if previous is not None and timestamp < previous:
                        errors.append(
                            f"Non-monotonic timestamp in episode {episode}: "
                            f"{previous} followed by {timestamp}"
                        )
                    previous_by_episode[episode] = timestamp
        except Exception:
            # Corruption is reported by _check_parquet_corrupt.
            continue
    return errors


def _check_missing_videos(
    dataset_path: Path, info: dict[str, Any], episodes: list[dict[str, Any]]
) -> list[str]:
    """Check every path referenced by image or video features."""
    video_path_template = info.get("video_path")
    if not isinstance(video_path_template, str) or not video_path_template:
        return []
    features = info.get("features", {})
    if not isinstance(features, dict):
        return []
    video_keys = [
        name
        for name, spec in features.items()
        if isinstance(spec, dict) and spec.get("dtype") in {"image", "video"}
    ]
    errors: list[str] = []
    for episode in episodes:
        episode_index = int(episode.get("episode_index", episode.get("index", 0)))
        data_chunk = int(episode.get("data/chunk_index", episode.get("chunk_index", 0)))
        data_file = int(episode.get("data/file_index", episode.get("file_index", episode_index)))
        for video_key in video_keys:
            video_chunk = int(
                episode.get(f"videos/{video_key}/chunk_index", data_chunk)
            )
            video_file = int(
                episode.get(f"videos/{video_key}/file_index", data_file)
            )
            substitutions = {
                "episode_chunk": data_chunk,
                "episode_index": episode_index,
                "chunk_index": video_chunk,
                "file_index": video_file,
                "video_key": video_key,
            }
            try:
                relative_path = video_path_template.format(**substitutions)
            except (KeyError, ValueError) as exc:
                errors.append(f"Invalid video_path template: {exc}")
                return errors
            if not (dataset_path / relative_path).is_file():
                errors.append(f"Missing video file: {relative_path}")
    return errors


def validate_dataset(dataset_path: str | Path) -> ValidationResult:
    """Run source-wide structural checks and fail closed on unknown versions."""
    source = Path(dataset_path).resolve()
    info_path = source / "meta" / "info.json"
    if not info_path.is_file():
        return ValidationResult(False, ("meta/info.json not found",), ())
    try:
        with info_path.open(encoding="utf-8") as stream:
            info = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return ValidationResult(False, (f"Invalid meta/info.json: {exc}",), ())

    version = detect_version(info)
    episodes, episode_load_errors = _load_episodes_for_validation(source, version)
    errors = list(episode_load_errors)
    errors.extend(_check_episodes(episodes, info))
    errors.extend(_check_parquet_corrupt(source))
    errors.extend(_check_data_row_total(source, info))
    errors.extend(_check_schema_drift(source, info))
    errors.extend(_check_timestamps(source))
    errors.extend(_check_missing_videos(source, info, episodes))
    return ValidationResult(valid=not errors, errors=tuple(errors), warnings=())
