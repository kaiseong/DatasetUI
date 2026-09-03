from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from datasetui.config import Settings
from datasetui.database import Database, RecipeRevisionMismatchError
from datasetui.transform_errors import CurationTransformError
from datasetui.transforms import (
    MAX_INFO_BYTES,
    _DatasetSource,
    _matrix_column,
    _read_regular_bytes,
    _safe_dataset_root,
)


class DatasetValidationError(CurationTransformError):
    pass


def validate_registered_dataset(
    *, database: Database, settings: Settings, payload: dict[str, Any]
) -> dict[str, Any]:
    record = database.get_dataset(payload["dataset_id"])
    if (
        not record["available"]
        or record["readiness"] != "ready"
        or record["fingerprint"] != payload["fingerprint"]
        or record["storage_area"] != payload["storage_area"]
        or record["relative_path"] != payload["relative_path"]
    ):
        raise RecipeRevisionMismatchError(payload["dataset_id"])
    root = _safe_dataset_root(
        settings.nas_root, record["storage_area"], record["relative_path"]
    )
    return validate_dataset_root(root, mode=payload["mode"], fingerprint=payload["fingerprint"])


def validate_dataset_root(
    root: Path, *, mode: str, fingerprint: str | None = None
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []

    def issue(severity: str, code: str, message: str, episode: int | None = None) -> None:
        item: dict[str, Any] = {"severity": severity, "code": code, "message": message}
        if episode is not None:
            item["episode_index"] = episode
        issues.append(item)

    raw = _read_regular_bytes(root / "meta/info.json", max_bytes=MAX_INFO_BYTES)
    actual_fingerprint = hashlib.sha256(raw).hexdigest()
    if fingerprint is not None and actual_fingerprint != fingerprint:
        raise RecipeRevisionMismatchError(actual_fingerprint)
    try:
        info = json.loads(raw)
        source = _DatasetSource(root, info)
    except Exception as exc:
        raise DatasetValidationError("Dataset metadata could not be loaded") from exc

    total_episodes = int(info.get("total_episodes", -1))
    if total_episodes < 1:
        issue("FAIL", "invalid_episode_total", "Dataset must contain at least one episode")
        indices: list[int] = []
    elif mode == "quick" and total_episodes > 2:
        indices = [0, total_episodes - 1]
    else:
        indices = list(range(total_episodes))

    total_frames = 0
    lengths: list[int] = []
    expected_global_index = 0
    for episode_index in indices:
        try:
            data, metadata = source.episode(episode_index)
        except Exception:
            issue("FAIL", "episode_read_failed", "Episode data could not be read", episode_index)
            continue
        length = len(data)
        lengths.append(length)
        total_frames += length
        if length < 1:
            issue("FAIL", "empty_episode", "Episode contains no frames", episode_index)
            continue
        for column in ("episode_index", "frame_index", "timestamp", "index"):
            if column not in data.columns:
                issue("FAIL", "missing_required_column", f"Required column is missing: {column}", episode_index)
        if "episode_index" in data and not (data["episode_index"].astype(int) == episode_index).all():
            issue("FAIL", "episode_index_mismatch", "Episode indices do not match metadata", episode_index)
        if "frame_index" in data and data["frame_index"].astype(int).tolist() != list(range(length)):
            issue("FAIL", "frame_index_mismatch", "Frame indices are not contiguous", episode_index)
        if mode != "quick" and "index" in data:
            expected = list(range(expected_global_index, expected_global_index + length))
            if data["index"].astype(int).tolist() != expected:
                issue("FAIL", "global_index_mismatch", "Global frame indices are not contiguous", episode_index)
            expected_global_index += length
        if "timestamp" in data:
            timestamps = np.asarray(data["timestamp"], dtype=np.float64)
            if not np.isfinite(timestamps).all():
                issue("FAIL", "timestamp_non_finite", "Timestamps contain NaN or infinity", episode_index)
            elif np.any(np.diff(timestamps) < 0):
                issue("FAIL", "timestamp_regression", "Timestamps move backwards", episode_index)
            elif len(timestamps) > 1:
                deltas = np.diff(timestamps)
                missing = int(sum(max(0, round(float(delta) * source.fps) - 1) for delta in deltas))
                if missing:
                    issue("WARN", "estimated_missing_frames", f"Approximately {missing} frames may be missing", episode_index)
                expected = 1.0 / source.fps
                if np.max(np.abs(deltas - expected)) > expected * 0.1:
                    issue("WARN", "fps_jitter", "Timestamp spacing varies by more than 10%", episode_index)
        for feature_name in ("action", "observation.state"):
            try:
                values = _matrix_column(data, feature_name)
            except CurationTransformError:
                issue("FAIL", "feature_shape_mismatch", f"Feature has missing or inconsistent shape: {feature_name}", episode_index)
                continue
            expected_shape = info.get("features", {}).get(feature_name, {}).get("shape")
            if isinstance(expected_shape, list) and math.prod(expected_shape) != values.shape[1]:
                issue("FAIL", "feature_shape_mismatch", f"Feature shape differs from metadata: {feature_name}", episode_index)
            if not np.isfinite(values).all():
                issue("FAIL", "feature_non_finite", f"Feature contains NaN or infinity: {feature_name}", episode_index)
            if length > 2 and np.all(np.nanstd(values, axis=0) < 1e-9):
                issue("WARN", "constant_sensor", f"Feature is constant: {feature_name}", episode_index)
        if "action" in data and length > 2:
            action = _matrix_column(data, "action")
            if np.max(np.abs(np.diff(action, axis=0))) > 10_000:
                issue("WARN", "action_jump", "Action contains an unusually large jump", episode_index)
        if mode != "quick":
            if source.version == "v3.0":
                start_offset = int(metadata.get("dataset_from_index", -1))
                end_offset = int(metadata.get("dataset_to_index", -1))
                if start_offset < 0 or end_offset - start_offset != length:
                    issue("FAIL", "invalid_episode_offsets", "Episode offsets are invalid", episode_index)
            _validate_videos(source, episode_index, metadata, length, issue)

    if mode != "quick" and len(indices) == total_episodes:
        if total_frames != int(info.get("total_frames", -1)):
            issue("FAIL", "frame_total_mismatch", "Frame total differs from meta/info.json")
        metadata_indices = sorted(source.episode_metadata)
        if metadata_indices and metadata_indices != list(range(total_episodes)):
            issue("FAIL", "episode_metadata_mismatch", "Episode metadata indices are incomplete")
        if lengths and max(lengths) > max(2, int(np.median(lengths) * 5)):
            issue("WARN", "abnormal_episode_length", "An episode is much longer than the median")
        stats_path = root / "meta/stats.json"
        try:
            stats = json.loads(_read_regular_bytes(stats_path, max_bytes=64 * 1024 * 1024))
            if not isinstance(stats, dict):
                raise ValueError
        except Exception:
            issue("FAIL", "stats_load_failed", "Dataset statistics could not be loaded")

    failures = sum(item["severity"] == "FAIL" for item in issues)
    warnings = sum(item["severity"] == "WARN" for item in issues)
    return {
        "mode": mode,
        "passed": failures == 0,
        "fingerprint": actual_fingerprint,
        "checked_episodes": len(indices),
        "total_episodes": total_episodes,
        "checked_frames": total_frames,
        "failures": failures,
        "warnings": warnings,
        "issues": issues,
        "loader_probe": "datasetui-structural-v1",
    }


def _validate_videos(
    source: _DatasetSource,
    episode_index: int,
    metadata: dict[str, Any],
    expected_frames: int,
    issue,
) -> None:
    if not source.video_keys:
        return
    try:
        import av
    except ImportError:
        issue("FAIL", "video_decoder_unavailable", "Video decoder is unavailable", episode_index)
        return
    for key in source.video_keys:
        try:
            path, start = source.video_source(episode_index, key, metadata)
            if start < 0:
                raise ValueError("negative video offset")
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError
            container = av.open(str(path))
            try:
                decoded = sum(1 for _ in container.decode(video=0))
            finally:
                container.close()
            if decoded < start + expected_frames:
                issue("FAIL", "video_frame_mismatch", f"Video frames do not cover data: {key}", episode_index)
        except Exception:
            issue("FAIL", "video_decode_failed", f"Video could not be decoded: {key}", episode_index)
