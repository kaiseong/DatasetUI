"""Stationary-segment trimming: find the moving span of each episode."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from datasetui.transform_errors import CurationTransformError



def trim_bounds(
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
        from datasetui.curation.stationary import stationary_trim_bounds

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
