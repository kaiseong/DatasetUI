"""Semantic schema checks, separate from materialization's permissive reader."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from datasetui.transforms import (
    _read_json_lines,
    _read_parquet,
    _safe_parquet_files,
    _task_rows_from_frame,
)


def integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def validate_info(info, issue) -> bool:
    valid = isinstance(info, dict)
    if not valid:
        issue("FAIL", "metadata_schema_invalid", "info.json must be an object")
        return False
    for name, minimum in (
        ("total_episodes", 1),
        ("total_frames", 1),
        ("total_tasks", 1),
    ):
        maximum = 1_000_000 if name != "total_frames" else 1_000_000_000_000
        if not integer(info.get(name)) or not minimum <= info[name] <= maximum:
            issue(
                "FAIL",
                "metadata_schema_invalid",
                f"{name} must be an integer within {minimum}..{maximum}",
            )
            valid = False
    fps = info.get("fps")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(fps)
        or fps <= 0
    ):
        issue("FAIL", "metadata_schema_invalid", "fps must be finite and positive")
        valid = False
    if info.get("codebase_version") not in {"v2.0", "v2.1", "v3.0"}:
        issue("FAIL", "metadata_schema_invalid", "Unsupported dataset version")
        valid = False
    features = info.get("features")
    if not isinstance(features, dict) or not features:
        issue("FAIL", "metadata_schema_invalid", "Feature schema is required")
        return False
    for key, feature in features.items():
        if (
            not isinstance(feature, dict)
            or not isinstance(feature.get("shape"), list)
            or not feature["shape"]
            or any(not integer(n) or n < 1 for n in feature["shape"])
        ):
            issue("FAIL", "metadata_schema_invalid", f"Invalid feature shape: {key}")
            valid = False
            continue
        dtype = feature.get("dtype")
        try:
            if (
                dtype not in {"video", "image", "string", "str", "language"}
                and np.dtype(dtype).kind not in "biuf"
            ):
                raise ValueError
            if not isinstance(dtype, str):
                raise ValueError
        except (TypeError, ValueError):
            issue(
                "FAIL", "metadata_schema_invalid", f"Unsupported feature dtype: {key}"
            )
            valid = False
    for key in (
        "action",
        "observation.state",
        "timestamp",
        "index",
        "frame_index",
        "episode_index",
        "task_index",
    ):
        if key not in features:
            issue(
                "FAIL",
                "metadata_schema_invalid",
                f"Required feature declaration missing: {key}",
            )
            valid = False
    splits = info.get("splits")
    if not isinstance(splits, dict) or not splits:
        issue("FAIL", "metadata_schema_invalid", "Episode splits are required")
        valid = False
    elif integer(info.get("total_episodes")):
        intervals = []
        for name, value in splits.items():
            match = (
                re.fullmatch(r"(\d+):(\d+)", value) if isinstance(value, str) else None
            )
            if (
                match is None
                or not 0 <= int(match[1]) < int(match[2]) <= info["total_episodes"]
            ):
                issue("FAIL", "metadata_schema_invalid", f"Invalid split range: {name}")
                valid = False
                continue
            intervals.append((int(match[1]), int(match[2])))
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor:
                issue(
                    "FAIL",
                    "metadata_schema_invalid",
                    "Split ranges overlap or leave unassigned episodes",
                )
                valid = False
            cursor = end
        if cursor != info["total_episodes"]:
            issue(
                "FAIL",
                "metadata_schema_invalid",
                "Split ranges do not cover every episode",
            )
            valid = False
    return valid


def validate_metadata(source, issue) -> None:
    root: Path = source.root
    try:
        if source.version == "v3.0":
            task_path = root / "meta/tasks.parquet"
            if task_path.is_symlink():
                raise ValueError("Unsafe task metadata")
            tasks = _task_rows_from_frame(_read_parquet(task_path))
            paths = _safe_parquet_files(root / "meta/episodes")
            episodes = [
                row for path in paths for row in _read_parquet(path).to_dict("records")
            ]
        else:
            tasks = _read_json_lines(root / "meta/tasks.jsonl")
            episodes = _read_json_lines(root / "meta/episodes.jsonl")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issue(
            "FAIL",
            "metadata_schema_invalid",
            f"Task/episode metadata is unreadable: {type(exc).__name__}",
        )
        return
    for rows, key, total in (
        (tasks, "task_index", source.info["total_tasks"]),
        (episodes, "episode_index", source.info["total_episodes"]),
    ):
        ids = [row.get(key) for row in rows]
        if (
            len(ids) != total
            or any(not integer(value) for value in ids)
            or any(value != expected for expected, value in enumerate(sorted(ids)))
        ):
            issue(
                "FAIL",
                "metadata_schema_invalid",
                f"Missing, duplicate or invalid {key} metadata",
            )
    if any(
        not isinstance(row.get("task", row.get("name")), str)
        or not row.get("task", row.get("name")).strip()
        for row in tasks
    ):
        issue(
            "FAIL",
            "metadata_schema_invalid",
            "Task descriptions must be nonempty strings",
        )


def validate_episode_structure(
    source, data, metadata, episode, global_start, *, full, issue
):
    length = len(data)
    valid_indices = {}
    for name in ("episode_index", "frame_index", "index", "task_index"):
        if name not in data:
            issue(
                "FAIL",
                "missing_required_column",
                f"Required column is missing: {name}",
                episode,
            )
            continue
        values = np.asarray(data[name])
        if (
            values.ndim != 1
            or values.dtype.kind not in "iu"
            or values.dtype.kind == "b"
        ):
            issue(
                "FAIL",
                "numeric_data_invalid",
                f"{name} must contain integers without coercion",
                episode,
            )
            continue
        valid_indices[name] = values
    for name, expected, code in (
        ("episode_index", np.full(length, episode), "episode_index_mismatch"),
        ("frame_index", np.arange(length), "frame_index_mismatch"),
        (
            "index",
            np.arange(global_start, global_start + length),
            "global_index_mismatch",
        ),
    ):
        if (
            name in valid_indices
            and (name != "index" or full)
            and not np.array_equal(valid_indices[name], expected)
        ):
            issue("FAIL", code, f"{name} does not match episode order", episode)
    tasks = valid_indices.get("task_index")
    if tasks is not None and not set(tasks).issubset(source.tasks):
        issue(
            "FAIL",
            "task_reference_invalid",
            "task_index references an unknown task",
            episode,
        )
    if not integer(metadata.get("length")) or metadata["length"] != length:
        issue(
            "FAIL",
            "episode_metadata_mismatch",
            "Episode length differs from metadata",
            episode,
        )
    task_labels = metadata.get("tasks")
    if isinstance(task_labels, np.ndarray):
        task_labels = task_labels.tolist()
    if not isinstance(task_labels, list) or any(
        not isinstance(v, str) for v in task_labels
    ):
        issue(
            "FAIL",
            "task_reference_invalid",
            "Episode task descriptions are missing or invalid",
            episode,
        )
    elif tasks is not None and set(task_labels) != {
        source.tasks.get(int(i)) for i in tasks
    }:
        issue(
            "FAIL",
            "task_reference_invalid",
            "Episode tasks disagree with frame task references",
            episode,
        )
    if source.version == "v3.0":
        start, end = (
            metadata.get("dataset_from_index"),
            metadata.get("dataset_to_index"),
        )
        if (
            not integer(start)
            or not integer(end)
            or end - start != length
            or (full and start != global_start)
        ):
            issue(
                "FAIL",
                "invalid_episode_offsets",
                "Dataset offsets do not match episode rows",
                episode,
            )
    if "timestamp" not in data:
        issue(
            "FAIL",
            "missing_required_column",
            "Required column is missing: timestamp",
            episode,
        )
        return None
    timestamps = np.asarray(data["timestamp"])
    if (
        timestamps.ndim != 1
        or timestamps.dtype.kind not in "fiu"
        or not np.isfinite(timestamps).all()
    ):
        issue(
            "FAIL",
            "timestamp_non_finite",
            "Timestamps must be finite numeric scalars",
            episode,
        )
        return None
    tolerance = max(1e-4, 0.01 / source.fps)
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        issue(
            "FAIL",
            "timestamp_regression",
            "Timestamps must be strictly increasing (no duplicates)",
            episode,
        )
    if not np.allclose(
        timestamps, np.arange(length) / source.fps, rtol=0, atol=tolerance
    ):
        issue(
            "FAIL",
            "timestamp_alignment",
            "Timestamps do not match frame_index / fps from zero",
            episode,
        )
    return timestamps


def validate_features(info, data, episode, issue):
    """Validate logical dtype and exact value shape; scalar [1] is canonical LeRobot."""
    for name, feature in info["features"].items():
        dtype = feature["dtype"]
        if dtype == "video":
            continue
        if name not in data:
            issue(
                "FAIL",
                "feature_shape_mismatch",
                f"Declared feature is missing: {name}",
                episode,
            )
            continue
        if dtype == "image":
            issue(
                "FAIL",
                "feature_type_unsupported",
                f"Embedded image decoding is not supported by this validator: {name}",
                episode,
            )
            continue
        if dtype == "language":
            for value in data[name]:
                rows = value.tolist() if isinstance(value, np.ndarray) else value
                if not isinstance(rows, list) or any(
                    not isinstance(row, dict)
                    or not isinstance(row.get("role"), str)
                    or not row["role"]
                    for row in rows
                ):
                    issue(
                        "FAIL",
                        "feature_dtype_mismatch",
                        f"Malformed language annotation: {name}",
                        episode,
                    )
                    break
            continue
        expected_shape = tuple(feature["shape"])
        bad_shape = bad_type = bad_finite = False
        findings = {}

        def note(kind, row, actual):
            if kind not in findings:
                findings[kind] = [0, row, actual]
            findings[kind][0] += 1

        def location(kind):
            count, row, actual = findings[kind]
            return (
                f"위치: 에피소드 {episode}, feature '{name}', "
                f"첫 문제 행 {row} (에피소드 내 0부터 시작).\n"
                f"범위: 검사한 {len(data)}개 행 중 {count}개 행. "
                f"첫 문제 행의 실제 상태: {actual}.\n"
            )

        for row, value in enumerate(data[name]):
            row_bad_type = False
            actual_type = "읽을 수 없음"
            try:
                array = np.asarray(value)
                actual_type = str(array.dtype)
                if array.shape != expected_shape and not (
                    expected_shape == (1,) and array.shape == ()
                ):
                    bad_shape = True
                    note("shape", row, str(list(array.shape)))
                if dtype in {"string", "str"}:
                    row_bad_type = array.dtype.kind not in "US"
                    continue
                expected = np.dtype(dtype)
                storage_dtype = (
                    np.dtype(data[name].dtype)
                    if array.shape == () and data[name].dtype.kind in "biuf"
                    else array.dtype
                )
                actual_type = str(storage_dtype)
                if storage_dtype != expected:
                    row_bad_type = True
                # Arrow/Pandas may widen numeric storage. Do not coerce strings,
                # booleans or fractional floats into a declared integer feature.
                allowed = (
                    "fiu"
                    if expected.kind == "f"
                    else "iu"
                    if expected.kind in "iu"
                    else "b"
                )
                if array.dtype.kind not in allowed:
                    row_bad_type = True
                    continue
                if not np.isfinite(array).all():
                    bad_finite = True
                    note("finite", row, "NaN 또는 +Inf/-Inf 포함")
                if expected.kind in "iu":
                    bounds = np.iinfo(expected)
                    row_bad_type |= bool(
                        np.any(array < bounds.min) or np.any(array > bounds.max)
                    )
                elif expected.kind == "f":
                    row_bad_type |= bool(np.any(np.abs(array) > np.finfo(expected).max))
            except (TypeError, ValueError, OverflowError):
                row_bad_type = True
            finally:
                if row_bad_type:
                    bad_type = True
                    note("dtype", row, actual_type)
        if bad_shape:
            issue(
                "FAIL",
                "feature_shape_mismatch",
                f"배열 크기가 선언과 다릅니다.\n"
                f"기준: meta/info.json의 features['{name}'].shape = {list(expected_shape)}.\n"
                + location("shape")
                + "조치: feature 차원 정의와 실제 배열을 확인하세요. 원본을 보존하고 "
                "수정한 별도 데이터셋에서 shape·통계를 다시 검사하세요.",
                episode,
            )
        if bad_type:
            issue(
                "FAIL",
                "feature_dtype_mismatch",
                f"저장 타입 또는 표현 가능한 값의 범위가 선언과 맞지 않습니다.\n"
                f"기준: meta/info.json의 features['{name}'].dtype = {dtype}.\n"
                + location("dtype")
                + "의미: 선언된 타입과 실제 저장 타입이 다르거나, 값이 선언 타입의 범위를 "
                "벗어났습니다. 이 오류만으로 시간 간격이나 동작값 자체가 틀렸다는 뜻은 아닙니다.\n"
                "조치: 생성 코드와 학습 로더의 dtype 요구사항을 확인하세요. "
                "원본을 보존하고 별도 데이터셋에서 실제 값을 의도한 타입으로 변환하거나, "
                "실제 타입이 의도된 것인지 확인한 뒤 메타데이터를 맞추고 전체 재검사하세요. "
                "float64 → float32 변환은 반올림 오차가 생길 수 있습니다.",
                episode,
            )
        if bad_finite:
            issue(
                "FAIL",
                "feature_non_finite",
                "유한하지 않은 수치가 있습니다.\n"
                + location("finite")
                + "조치: 해당 행의 센서 기록·계산 과정을 확인하세요. NaN/Inf는 정상적인 "
                "수치로 학습에 사용할 수 없습니다. 원본을 보존하고 제외 또는 보정 정책을 "
                "정한 뒤 별도 데이터셋의 통계를 재계산하고 전체 재검사하세요.",
                episode,
            )
        if (
            name in {"action", "observation.state"}
            and len(data) > 2
            and not (bad_shape or bad_type or bad_finite)
        ):
            matrix = np.stack([np.asarray(v).reshape(-1) for v in data[name]])
            constant = np.flatnonzero(np.std(matrix, axis=0) < 1e-9)
            if constant.size:
                issue(
                    "WARN",
                    "constant_sensor",
                    f"Constant dimensions in {name}: {constant.tolist()}",
                    episode,
                )
            if name == "action" and np.max(np.abs(np.diff(matrix, axis=0))) > 10_000:
                issue(
                    "WARN",
                    "action_jump",
                    "Action jump exceeds the generic 10000-unit heuristic (not a robot-specific limit)",
                    episode,
                )
