"""Logical stationary-edge trimming adapted from the local 5090 tool.

Source: ``VLA/lerobot-original`` commit ``e15350de``, specifically
``tools/lerobot_dataset_tools/trim.py`` functions ``_load_state_array``,
``_stationary_prefix``, and ``_compute_one_range``.  That tool is a local custom
extension, not an official upstream LeRobot operation.  DatasetUI adds explicit
validation and maps its existing per-side margin fields onto the source tool's
keep-start/keep-end behavior.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from datasetui.transform_errors import CurationTransformError


def stationary_trim_bounds(
    data: pd.DataFrame,
    *,
    fps: float,
    config: dict[str, Any],
    episode_index: int,
) -> tuple[int, int, str]:
    """Return the logical trim range used by the local 5090 stationary tool.

    This deliberately follows
    ``VLA/lerobot-original/tools/lerobot_dataset_tools/trim.py`` rather than an
    upstream LeRobot API: leading/trailing state samples are compared with the
    first/last sample in absolute state units.  DatasetUI keeps the established
    per-side margin names as the amount of stationary context to retain.
    """
    state_key = str(config.get("state_key") or "observation.state")
    if state_key != "observation.state":
        raise CurationTransformError(
            "Stationary trim currently supports only observation.state"
        )
    if state_key not in data.columns:
        raise CurationTransformError(
            f"Stationary trim state feature is missing in episode {episode_index}: "
            f"{state_key}"
        )
    try:
        epsilon = float(config.get("state_epsilon", 5e-4))
    except (TypeError, ValueError) as exc:
        raise CurationTransformError(
            "Stationary trim state_epsilon must be a positive finite number"
        ) from exc
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise CurationTransformError(
            "Stationary trim state_epsilon must be a positive finite number"
        )

    values = data[state_key].to_numpy()
    if len(values) == 0:
        raise CurationTransformError(f"Episode {episode_index} has no frames")
    first = values[0]
    try:
        if isinstance(first, (list, tuple, np.ndarray)):
            states = np.stack(values).astype(np.float64)
        else:
            states = np.asarray(values)
            if states.ndim == 1:
                states = states.reshape(-1, 1)
            states = states.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise CurationTransformError(
            f"Stationary trim state values are invalid in episode {episode_index}"
        ) from exc
    if states.ndim != 2 or states.shape[1] == 0 or not np.isfinite(states).all():
        raise CurationTransformError(
            f"Stationary trim requires finite state values in episode {episode_index}"
        )

    start_diffs = np.max(np.abs(states - states[0]), axis=1)
    end_diffs = np.max(np.abs(states - states[-1]), axis=1)
    start_stationary = _stationary_prefix(start_diffs <= epsilon)
    end_stationary = _stationary_prefix((end_diffs <= epsilon)[::-1])
    start_keep = _margin_frames(config, "start", fps)
    end_keep = _margin_frames(config, "end", fps)
    start = max(0, start_stationary - start_keep)
    end = len(data) - max(0, end_stationary - end_keep)
    if start >= end:
        raise CurationTransformError(
            "Stationary trim would remove all frames from episode "
            f"{episode_index}: start={start}, end={end}, length={len(data)}"
        )
    return int(start), int(end), "stationary"


def _stationary_prefix(mask: np.ndarray) -> int:
    for index, value in enumerate(mask):
        if not bool(value):
            return index
    return len(mask)


def _margin_frames(config: dict[str, Any], side: str, fps: float) -> int:
    raw = config.get(f"{side}_margin_s")
    if raw is None:
        raw = config.get("margin_s", 1.0)
    try:
        seconds = float(raw)
    except (TypeError, ValueError) as exc:
        raise CurationTransformError(
            f"Stationary trim {side} margin must be a non-negative finite number"
        ) from exc
    if not np.isfinite(seconds) or seconds < 0:
        raise CurationTransformError(
            f"Stationary trim {side} margin must be a non-negative finite number"
        )
    return max(0, round(seconds * fps))
