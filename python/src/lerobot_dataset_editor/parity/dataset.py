"""Frame, media, and episode views over v2.1/v3 LeRobot datasets."""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any

from .common import dimension_names, episode_records, episode_table, finite, load_info
from .progress import read_progress


def _first_scalar(value: Any) -> float | None:
    while isinstance(value, list):
        if not value:
            return None
        value = value[0]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        return result if math.isfinite(result) else None
    return None


def _depth_encoding(feature: Any) -> dict[str, Any] | None:
    if not isinstance(feature, dict) or not isinstance(feature.get("info"), dict):
        return None
    info = feature["info"]
    depth_min, depth_max = info.get("video.depth_min"), info.get("video.depth_max")
    shift, use_log = info.get("video.shift"), info.get("video.use_log")
    if info.get("is_depth_map") is not True or not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in (depth_min, depth_max, shift)
    ) or not isinstance(use_log, bool) or not depth_max > depth_min \
            or (use_log and depth_min + shift <= 0):
        return None
    return {"depth_min": float(depth_min), "depth_max": float(depth_max),
            "shift": float(shift), "use_log": use_log,
            "depth_unit": info.get("depth_unit"),
            "unit_to_metres": .001 if info.get("depth_unit") == "mm" else 1.0}


def _colormap_band(stats: Any, encoding: dict[str, Any] | None) -> tuple[float, float] | None:
    if not isinstance(stats, dict):
        return None
    low = _first_scalar(stats.get("q01"))
    high = _first_scalar(stats.get("q99"))
    if low is None:
        low = _first_scalar(stats.get("min"))
    if high is None:
        high = _first_scalar(stats.get("max"))
    if low is None or high is None or high <= low:
        return None
    if encoding is None:
        return low, high
    depth_min, depth_max = encoding["depth_min"], encoding["depth_max"]
    shift, scale = encoding["shift"], encoding["unit_to_metres"]
    def normalize(value: float) -> float:
        depth = value * scale
        if encoding["use_log"]:
            normalized = ((math.log(depth + shift) - math.log(depth_min + shift)) /
                          (math.log(depth_max + shift) - math.log(depth_min + shift)))
        else:
            normalized = (depth - depth_min) / (depth_max - depth_min)
        return min(1.0, max(0.0, normalized))
    normalized = normalize(low), normalize(high)
    return normalized if normalized[1] > normalized[0] else None


def _media_metadata(source: Path, key: str, feature: Any) -> dict[str, Any]:
    stats_path = source / "meta" / "stats.json"
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8")).get(key, {})
    except (OSError, json.JSONDecodeError, AttributeError):
        stats = {}
    feature = feature if isinstance(feature, dict) else {}
    shape = feature.get("shape") if isinstance(feature.get("shape"), list) else []
    encoding = _depth_encoding(feature)
    band = _colormap_band(stats, encoding)
    raw_low = _first_scalar(stats.get("q01")) if isinstance(stats, dict) else None
    raw_high = _first_scalar(stats.get("q99")) if isinstance(stats, dict) else None
    if raw_low is None and isinstance(stats, dict):
        raw_low = _first_scalar(stats.get("min"))
    if raw_high is None and isinstance(stats, dict):
        raw_high = _first_scalar(stats.get("max"))
    result: dict[str, Any] = {"shape": shape, "grayscale": len(shape) >= 3 and shape[2] == 1,
                              "is_depth_map": encoding is not None}
    if raw_low is not None and raw_high is not None and raw_high > raw_low:
        result["raw_band"] = [raw_low, raw_high]
    if band:
        result["q01"], result["q99"] = band
    if encoding:
        result["depth_encoding"] = encoding
    return result


def _episode_metadata(source: Path, episode_index: int) -> dict[str, Any]:
    legacy = source / "meta" / "episodes.jsonl"
    if legacy.is_file():
        for line in legacy.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if int(row.get("episode_index", -1)) == episode_index:
                return row
        return {}
    import pyarrow.parquet as pq
    for path in sorted((source / "meta" / "episodes").rglob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            if int(row.get("episode_index", -1)) == episode_index:
                return row
    return {}


def _media_value(value: Any, *, source: Path, key: str, row: int,
                 episode_index: int) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    raw = value.get("bytes")
    path = value.get("path")
    if isinstance(raw, bytes):
        return {"kind": "image", "key": key, "frame": row,
                "data_url": "data:image/png;base64," + base64.b64encode(raw).decode("ascii")}
    if isinstance(path, str) and path:
        direct = source / path
        conventional = source / "images" / key / f"episode_{episode_index:06d}" / path
        resolved = conventional if conventional.is_file() else direct
        try:
            relative = str(resolved.resolve().relative_to(source.resolve()))
        except ValueError:
            relative = path
        return {"kind": "image", "key": key, "frame": row, "relative_path": relative}
    return None


def dataset_summary(source: Path) -> dict[str, Any]:
    info = load_info(source)
    features = info.get("features", {})
    return finite({
        "version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "fps": info.get("fps"),
        "total_episodes": info.get("total_episodes", 0),
        "total_frames": info.get("total_frames", 0),
        "total_tasks": info.get("total_tasks", 0),
        "features": [
            {"name": key, **(value if isinstance(value, dict) else {})}
            for key, value in sorted(features.items())
        ],
        "cameras": [key for key, value in features.items()
                    if isinstance(value, dict) and value.get("dtype") in {"image", "video"}],
    })


def dataset_episode(source: Path, episode_index: int) -> dict[str, Any]:
    if not isinstance(episode_index, int) or isinstance(episode_index, bool) or episode_index < 0:
        raise ValueError("episode_index must be a non-negative integer")
    info = load_info(source)
    features = info.get("features", {})
    video_columns = [field for name, spec in features.items()
                     if isinstance(spec, dict) and spec.get("dtype") == "video"
                     for field in (f"videos/{name}/chunk_index", f"videos/{name}/file_index",
                                   f"videos/{name}/from_timestamp", f"videos/{name}/to_timestamp")]
    records = episode_records(source, video_columns)
    metadata = next((row for row in records
                     if int(row.get("episode_index", -1)) == episode_index), {})
    projected = ["episode_index", "timestamp", "action", "observation.state"]
    projected.extend(name for name, spec in features.items()
                     if isinstance(spec, dict) and spec.get("dtype") == "image")
    table = episode_table(source, episode_index, projected, records=records)
    if table.num_rows == 0:
        raise ValueError(f"episode not found: {episode_index}")
    timestamps = [float(value) for value in table["timestamp"].to_pylist()]
    duration = round(table.num_rows / max(float(info.get("fps", 1)), 1), 6)
    named_series: dict[str, list[dict[str, Any]]] = {"action": [], "observation.state": []}
    media: list[dict[str, Any]] = []
    for name in table.column_names:
        values = table[name].to_pylist()
        spec = features.get(name, {}) if isinstance(features, dict) else {}
        dtype = spec.get("dtype") if isinstance(spec, dict) else None
        if dtype in {"image", "video"}:
            media_metadata = _media_metadata(source, name, spec)
            descriptors = [item for row, value in enumerate(values)
                           if (item := _media_value(value, source=source, key=name, row=row,
                                                   episode_index=episode_index))]
            if descriptors:
                # Embedded-image datasets expose the first frame as a track URL;
                # callers can seek frames through a subsequent episode request.
                track = {"camera": name, "kind": "depth" if media_metadata["is_depth_map"] else "image",
                         "frames": descriptors, **media_metadata}
                if "data_url" in descriptors[0]:
                    track["url"] = descriptors[0]["data_url"]
                else:
                    track["relative_path"] = descriptors[0]["relative_path"]
                media.append(track)
            elif isinstance(info.get("video_path"), str):
                template = info["video_path"]
                chunk_index = int(metadata.get(f"videos/{name}/chunk_index", metadata.get("data/chunk_index", episode_index // 1000)))
                file_index = int(metadata.get(f"videos/{name}/file_index", metadata.get("data/file_index", episode_index)))
                relative = template.format(episode_chunk=chunk_index, episode_index=episode_index,
                                           chunk_index=chunk_index, file_index=file_index,
                                           video_key=name)
                media.append({"camera": name,
                              "kind": "depth" if media_metadata["is_depth_map"] else "video",
                              "relative_path": relative, **media_metadata,
                              "from": metadata.get(f"videos/{name}/from_timestamp", 0),
                              "to": metadata.get(f"videos/{name}/to_timestamp", duration)})
            continue
        if not values or name in {"index", "episode_index", "frame_index", "timestamp", "task_index"}:
            continue
        first = next((value for value in values if value is not None), None)
        if isinstance(first, (int, float)) and not isinstance(first, bool):
            if name in named_series:
                named_series[name].append({"name": name, "values": [float(v) for v in values]})
        elif isinstance(first, list) and all(isinstance(v, (int, float)) for v in first):
            if name in named_series:
                names = dimension_names(info, name, len(first))
                named_series[name].extend({"name": dimension, "values": [float(row[index]) for row in values]}
                                          for index, dimension in enumerate(names))
    # Video features are metadata-backed in standard LeRobot datasets and do
    # not need a Parquet column.  Add every declared video camera that was not
    # encountered above so multi-camera and consolidated segment playback work
    # for real Hub datasets, not only embedded-image fixtures.
    present_cameras = {track["camera"] for track in media}
    video_template = info.get("video_path")
    if isinstance(features, dict) and isinstance(video_template, str):
        for name, spec in features.items():
            if name in present_cameras or not isinstance(spec, dict) or spec.get("dtype") != "video":
                continue
            media_metadata = _media_metadata(source, name, spec)
            chunk_index = int(metadata.get(
                f"videos/{name}/chunk_index", metadata.get("data/chunk_index", episode_index // 1000)
            ))
            file_index = int(metadata.get(
                f"videos/{name}/file_index", metadata.get("data/file_index", episode_index)
            ))
            relative = video_template.format(
                episode_chunk=chunk_index,
                episode_index=episode_index,
                chunk_index=chunk_index,
                file_index=file_index,
                video_key=name,
            )
            media.append({
                "camera": name,
                "kind": "depth" if media_metadata["is_depth_map"] else "video",
                "relative_path": relative,
                **media_metadata,
                "from": metadata.get(f"videos/{name}/from_timestamp", 0),
                "to": metadata.get(f"videos/{name}/to_timestamp", duration),
            })
    progress_result = read_progress(source, episode_index, duration)
    progress = [{"name": item["name"], "values": [point["value"] for point in item["points"]]}
                for item in progress_result["series"]]
    robot_type = str(info.get("robot_type", ""))
    return finite({
        "episode_index": episode_index,
        "episode_count": int(info.get("total_episodes", 0)),
        "frame_count": table.num_rows,
        "fps": float(info.get("fps", 0)),
        "duration": duration,
        "timestamps": timestamps,
        "media": media,
        "action": named_series["action"],
        "state": named_series["observation.state"],
        "progress": progress,
        "robot": {"type": robot_type, "joints": named_series["observation.state"]}
                 if str(info.get("codebase_version", "")).startswith("v3") else None,
    })
