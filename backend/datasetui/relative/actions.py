"""Official LeRobot relative-action statistics with exact dimension selection."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from datasetui.official.runtime import require_runtime, UPSTREAM_COMMIT
from datasetui.statistics.exact import StableMoments, exact_quantiles
from datasetui.transform_errors import CurationTransformError


DEFAULT_CHUNK_SIZE = 50


MIN_CHUNK_SIZE = 1


MAX_CHUNK_SIZE = 1024


TARGET_BATCH_VALUES = 65_536


DEFAULT_MAX_SCRATCH_BYTES = 16 * 1024**3


MIN_FREE_SPACE_MARGIN_BYTES = 64 * 1024**2


ProgressCallback = Callable[[dict[str, Any]], None]


def _load_official_to_relative_actions():
    """Load the official helper only after the pinned runtime identity gate passes."""

    require_runtime()
    from lerobot.processor.relative_action_processor import to_relative_actions

    return to_relative_actions


def _flatten_names(value: Any, *, feature: str) -> list[str]:
    if isinstance(value, list):
        names = value
    elif isinstance(value, dict) and len(value) == 1:
        group, names = next(iter(value.items()))
        if not isinstance(group, str) or not group:
            raise CurationTransformError(
                f"Relative action {feature} names contain an invalid group"
            )
        if not isinstance(names, list):
            raise CurationTransformError(
                f"Relative action {feature} names must be arrays"
            )
    else:
        raise CurationTransformError(
            f"Relative action requires explicit {feature} dimension names"
        )
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise CurationTransformError(
            f"Relative action {feature} dimension names are invalid"
        )
    if len(names) != len(set(names)):
        raise CurationTransformError(
            f"Relative action {feature} dimension names must be unique"
        )
    return list(names)


def _feature_descriptor(info: dict[str, Any], key: str) -> tuple[list[str], int]:
    features = info.get("features")
    descriptor = features.get(key) if isinstance(features, dict) else None
    if not isinstance(descriptor, dict):
        raise CurationTransformError(f"Relative action feature is missing: {key}")
    if descriptor.get("dtype") != "float32":
        raise CurationTransformError(
            f"Relative action requires float32 feature metadata: {key}"
        )
    shape = descriptor.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 1
        or isinstance(shape[0], bool)
        or not isinstance(shape[0], int)
        or shape[0] < 1
    ):
        raise CurationTransformError(f"Relative action feature shape is invalid: {key}")
    width = shape[0]
    names = _flatten_names(descriptor.get("names"), feature=key)
    if len(names) != width:
        raise CurationTransformError(
            f"Relative action names differ from declared shape: {key}"
        )
    return names, width


def dimension_options(info: dict[str, Any]) -> list[str]:
    """Return dimensions that can safely be made relative by exact name and index."""

    action_names, action_width = _feature_descriptor(info, "action")
    state_names, state_width = _feature_descriptor(info, "observation.state")
    if action_width != state_width:
        raise CurationTransformError(
            "Relative action requires equal action and state widths"
        )
    return [
        name for index, name in enumerate(action_names) if state_names[index] == name
    ]


def _chunk_size(config: dict[str, Any]) -> int:
    value = config.get("chunk_size", config.get("action_horizon", DEFAULT_CHUNK_SIZE))
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < MIN_CHUNK_SIZE
        or value > MAX_CHUNK_SIZE
    ):
        raise CurationTransformError(
            f"Relative action chunk_size must be between {MIN_CHUNK_SIZE} and {MAX_CHUNK_SIZE}"
        )
    return value


def _scratch_settings(required_bytes: int) -> Path:
    raw_limit = os.environ.get(
        "DATASETUI_RELATIVE_MAX_SCRATCH_BYTES", str(DEFAULT_MAX_SCRATCH_BYTES)
    )
    try:
        maximum_bytes = int(raw_limit)
    except ValueError as exc:
        raise CurationTransformError(
            "DATASETUI_RELATIVE_MAX_SCRATCH_BYTES is invalid"
        ) from exc
    if maximum_bytes < 1:
        raise CurationTransformError(
            "DATASETUI_RELATIVE_MAX_SCRATCH_BYTES must be positive"
        )
    if required_bytes > maximum_bytes:
        raise CurationTransformError(
            "Relative action exact statistics exceed the scratch limit "
            f"({required_bytes} > {maximum_bytes} bytes)"
        )
    configured = os.environ.get("DATASETUI_RELATIVE_SCRATCH_DIR")
    parent = Path(configured) if configured else Path(tempfile.gettempdir())
    if not parent.is_dir():
        raise CurationTransformError("Relative action scratch directory is unavailable")
    try:
        free_bytes = shutil.disk_usage(parent).free
    except OSError as exc:
        raise CurationTransformError(
            "Relative action scratch free space is unavailable"
        ) from exc
    margin = max(
        MIN_FREE_SPACE_MARGIN_BYTES,
        min(required_bytes // 10, 1024**3),
    )
    if free_bytes < required_bytes + margin:
        raise CurationTransformError(
            "Relative action exact statistics need more scratch space "
            f"(required={required_bytes}, margin={margin}, free={free_bytes} bytes)"
        )
    return parent


def _matrix(data: pd.DataFrame, key: str, width: int, episode_index: int) -> np.ndarray:
    if key not in data.columns:
        raise CurationTransformError(f"Relative action feature is missing: {key}")
    rows: list[np.ndarray] = []
    for row_index, value in enumerate(data[key].tolist()):
        if value is None:
            raise CurationTransformError(
                f"Relative action contains null values: {key} "
                f"(episode {episode_index}, row {row_index})"
            )
        array = np.asarray(value)
        if array.dtype != np.dtype(np.float32):
            raise CurationTransformError(
                f"Relative action requires float32 values: {key} "
                f"(episode {episode_index}, row {row_index})"
            )
        flat = array.reshape(-1)
        if len(flat) != width:
            raise CurationTransformError(
                f"Relative action value width differs from metadata: {key} "
                f"(episode {episode_index}, row {row_index})"
            )
        if not np.isfinite(flat).all():
            raise CurationTransformError(
                f"Relative action contains NaN or infinity: {key} "
                f"(episode {episode_index}, row {row_index})"
            )
        rows.append(flat.copy())
    if not rows:
        return np.empty((0, width), dtype=np.float32)
    return np.stack(rows)


def _progress(
    callback: ProgressCallback | None,
    *,
    completed: int,
    total: int,
    current_item: str,
    force: bool = False,
) -> None:
    if callback is None:
        return
    event: dict[str, Any] = {
        "stage": "relative_action_statistics",
        "completed": completed,
        "total": total,
        "unit": "chunks",
        "current_item": current_item,
    }
    if force:
        event["_force"] = True
    callback(event)


def compute_relative_action_profile(
    info: dict[str, Any],
    episodes: Sequence[pd.DataFrame],
    config: dict[str, Any],
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Compute exact mixed relative/absolute action statistics over valid chunks."""

    if not isinstance(config, dict):
        raise CurationTransformError("Relative action configuration is invalid")
    if not config.get("enabled", False):
        return {"enabled": False, "dimensions": [], "statistics": {}}

    action_names, action_width = _feature_descriptor(info, "action")
    state_names, state_width = _feature_descriptor(info, "observation.state")
    if action_width != state_width:
        raise CurationTransformError(
            "Relative action requires equal action and state widths"
        )
    raw_dimensions = config.get("dimensions")
    if (
        not isinstance(raw_dimensions, list)
        or not raw_dimensions
        or any(not isinstance(name, str) or not name for name in raw_dimensions)
        or len(raw_dimensions) != len(set(raw_dimensions))
    ):
        raise CurationTransformError(
            "Relative action dimensions must be a non-empty unique name list"
        )
    dimensions = list(raw_dimensions)
    action_positions = {name: index for index, name in enumerate(action_names)}
    for name in dimensions:
        index = action_positions.get(name)
        if index is None:
            raise CurationTransformError(f"Unknown relative action dimension: {name}")
        if state_names[index] != name:
            raise CurationTransformError(
                f"Relative action dimension does not match state at the same index: {name}"
            )
    selected_dimensions = set(dimensions)
    mask = np.asarray(
        [name in selected_dimensions for name in action_names], dtype=np.bool_
    )
    chunk_size = _chunk_size(config)
    if not isinstance(episodes, Sequence) or isinstance(episodes, (str, bytes)):
        raise CurationTransformError("Relative action episodes are invalid")
    episode_count = len(episodes)
    if episode_count == 0:
        raise CurationTransformError("Relative action source has no episodes")

    valid_chunks = 0
    skipped_short = 0
    for episode_index in range(episode_count):
        episode = episodes[episode_index]
        if not isinstance(episode, pd.DataFrame):
            raise CurationTransformError("Relative action episode is invalid")
        episode_length = len(episode)
        if episode_length < chunk_size:
            skipped_short += 1
        else:
            valid_chunks += episode_length - chunk_size + 1
    if valid_chunks == 0:
        raise CurationTransformError(
            "No valid same-episode relative action chunks were found "
            f"(episodes={episode_count}, skipped_short={skipped_short}, "
            f"chunk_size={chunk_size})"
        )

    value_count = valid_chunks * chunk_size
    required_scratch_bytes = action_width * value_count * np.dtype(np.float64).itemsize
    scratch_parent = _scratch_settings(required_scratch_bytes)
    official_to_relative_actions = _load_official_to_relative_actions()
    import torch

    moments = StableMoments(action_width)
    batch_chunks = max(1, TARGET_BATCH_VALUES // chunk_size)
    completed_chunks = 0
    aliases = [f"datasetui_dim_{index:04d}" for index in range(action_width)]
    processor_config = {
        "enabled": True,
        "action_names": aliases,
        "exclude_joints": [
            alias for alias, selected in zip(aliases, mask.tolist()) if not selected
        ],
    }
    _progress(
        on_progress,
        completed=0,
        total=valid_chunks,
        current_item="relative action",
        force=True,
    )
    with tempfile.TemporaryDirectory(
        prefix="datasetui-relative-action-", dir=scratch_parent
    ) as scratch:
        storage = np.memmap(
            Path(scratch) / "action.float64",
            mode="w+",
            dtype=np.float64,
            shape=(action_width, value_count),
        )
        offset = 0
        try:
            horizon_offsets = np.arange(chunk_size)
            for episode_index in range(episode_count):
                episode = episodes[episode_index]
                if not isinstance(episode, pd.DataFrame):
                    raise CurationTransformError("Relative action episode is invalid")
                action = _matrix(episode, "action", action_width, episode_index)
                state = _matrix(
                    episode, "observation.state", state_width, episode_index
                )
                if len(action) != len(state):
                    raise CurationTransformError(
                        "Relative action/state row counts differ in episode "
                        f"{episode_index}"
                    )
                starts_count = max(0, len(action) - chunk_size + 1)
                for batch_start in range(0, starts_count, batch_chunks):
                    starts = np.arange(
                        batch_start,
                        min(starts_count, batch_start + batch_chunks),
                    )
                    indices = starts[:, None] + horizon_offsets[None, :]
                    chunks = action[indices]
                    references = state[starts]
                    if (
                        not np.isfinite(chunks).all()
                        or not np.isfinite(references).all()
                    ):
                        raise CurationTransformError(
                            "Relative action input contains NaN or infinity"
                        )
                    action_tensor = torch.from_numpy(chunks.copy())
                    state_tensor = torch.from_numpy(references.copy())
                    converted_tensor = official_to_relative_actions(
                        action_tensor, state_tensor, mask.tolist()
                    )
                    if not isinstance(converted_tensor, torch.Tensor):
                        raise CurationTransformError(
                            "Official relative action function returned an invalid value"
                        )
                    converted = converted_tensor.detach().cpu().numpy()
                    if (
                        converted.shape != chunks.shape
                        or not np.isfinite(converted).all()
                    ):
                        raise CurationTransformError(
                            "Official relative action output is invalid"
                        )
                    if not np.array_equal(converted[..., ~mask], chunks[..., ~mask]):
                        raise CurationTransformError(
                            "Official relative action changed an unchecked dimension"
                        )
                    flat = converted.reshape(-1, action_width)
                    next_offset = offset + len(flat)
                    storage[:, offset:next_offset] = flat.T
                    moments.update(flat)
                    offset = next_offset
                    completed_chunks += len(starts)
                    _progress(
                        on_progress,
                        completed=completed_chunks,
                        total=valid_chunks,
                        current_item="relative action",
                    )
            if offset != value_count or completed_chunks != valid_chunks:
                raise CurationTransformError(
                    "Relative action statistics row count is inconsistent"
                )
            storage.flush()
            statistics = moments.finish()
            statistics.update(exact_quantiles(storage, value_count))
            serialized_statistics = {
                key: np.asarray(values, dtype=np.float64).tolist()
                for key, values in statistics.items()
            }
            serialized_statistics["count"] = [value_count]
        finally:
            del storage

    _progress(
        on_progress,
        completed=valid_chunks,
        total=valid_chunks,
        current_item="relative action",
        force=True,
    )
    return {
        "enabled": True,
        "dimensions": dimensions,
        "absolute_dimensions": [
            name for name in action_names if name not in dimensions
        ],
        "mask": mask.tolist(),
        "action_names": action_names,
        "chunk_size": chunk_size,
        "action_horizon": chunk_size,
        "stored_action": "absolute",
        "training_normalization": True,
        "statistics_scope": "chunk-relative",
        "engine": "datasetui-official-relative-action-v1",
        "official_function": (
            "lerobot.processor.relative_action_processor.to_relative_actions"
        ),
        "upstream_commit": UPSTREAM_COMMIT,
        "processor_config": processor_config,
        "episodes_total": episode_count,
        "episodes_used": episode_count - skipped_short,
        "episodes_skipped_short": skipped_short,
        "valid_chunks": valid_chunks,
        "statistics": serialized_statistics,
    }
