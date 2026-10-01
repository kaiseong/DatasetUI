from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from datasetui.transform_errors import CurationTransformError


IssueCallback = Callable[[str, str, str, int | None], None]
REQUIRED_STATISTICS = ("min", "max", "mean", "std", "count")
VISUAL_DTYPES = {"image", "video"}
NON_NUMERIC_DTYPES = {"string", "language"}
MAX_STATS_BYTES = 64 * 1024 * 1024
OFFICIAL_BOOKKEEPING_FEATURES = {"index", "episode_index", "task_index"}


def _official_policy(root: Path, issue: IssueCallback) -> str | None:
    marker_path = root / "meta/datasetui_provenance.json"
    if not marker_path.exists() and not marker_path.is_symlink():
        return None
    try:
        if (
            marker_path.is_symlink()
            or not marker_path.is_file()
            or marker_path.stat().st_size > 64 * 1024
        ):
            raise ValueError("unsafe provenance file")
        marker = json.loads(marker_path.read_bytes())
        from datasetui.lerobot_runtime import ENGINE_POLICY, UPSTREAM_COMMIT
        from datasetui.official_operations import (
            MERGE_STATISTICS_POLICY,
            OFFICIAL_STATISTICS_MARKER,
            _digest,
            _data_digest,
            _episode_metadata_digest,
            _video_digest,
        )

        required = {
            "schema",
            "statistics_policy",
            "engine",
            "upstream_commit",
            "official_function",
            "info_sha256",
            "stats_sha256",
            "episodes_sha256",
            "videos_sha256",
            "data_sha256",
        }
        if not isinstance(marker, dict) or set(marker) != required:
            raise ValueError("unexpected provenance schema")
        operation = marker["official_function"]
        if (
            marker["schema"] != OFFICIAL_STATISTICS_MARKER
            or marker["statistics_policy"] != MERGE_STATISTICS_POLICY
            or marker["engine"] != ENGINE_POLICY
            or marker["upstream_commit"] != UPSTREAM_COMMIT
            or operation
            not in {"merge_datasets", "split_dataset", "aggregate_stats"}
        ):
            raise ValueError("unsupported provenance policy")
        hashes = {
            "info_sha256": _digest(root / "meta/info.json"),
            "stats_sha256": _digest(root / "meta/stats.json"),
            "episodes_sha256": _episode_metadata_digest(root),
            "videos_sha256": _video_digest(root),
            "data_sha256": _data_digest(root),
        }
        if any(
            not isinstance(marker[key], str)
            or len(marker[key]) != 64
            or marker[key] != value
            for key, value in hashes.items()
        ):
            raise ValueError("provenance content hashes differ")
        return operation
    except (OSError, ValueError, TypeError, KeyError, CurationTransformError) as exc:
        issue(
            "FAIL",
            "official_statistics_provenance_invalid",
            f"Official statistics provenance is invalid: {exc}",
            None,
        )
        return None


@dataclass(frozen=True)
class StatisticsValidationSummary:
    validated_features: tuple[str, ...]
    unverified_visual_features: tuple[str, ...]


@dataclass
class RunningMoments:
    count: int = 0
    mean: np.ndarray | None = None
    sum_squared_deviation: np.ndarray | None = None
    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None

    def update(self, values: np.ndarray) -> None:
        batch_count = len(values)
        if batch_count == 0:
            return
        batch_mean = np.mean(values, axis=0, dtype=np.float64)
        batch_sum_squared_deviation = np.sum(
            (values - batch_mean) ** 2, axis=0, dtype=np.float64
        )
        batch_min = np.min(values, axis=0)
        batch_max = np.max(values, axis=0)
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.sum_squared_deviation = batch_sum_squared_deviation
            self.minimum = batch_min
            self.maximum = batch_max
            return
        assert self.mean is not None
        assert self.sum_squared_deviation is not None
        assert self.minimum is not None
        assert self.maximum is not None
        combined_count = self.count + batch_count
        weight = batch_count / combined_count
        delta = batch_mean - self.mean
        self.sum_squared_deviation += (
            batch_sum_squared_deviation
            + delta * delta * self.count * batch_count / combined_count
        )
        self.mean += delta * weight
        self.minimum = np.minimum(self.minimum, batch_min)
        self.maximum = np.maximum(self.maximum, batch_max)
        self.count = combined_count

    def values(self) -> dict[str, np.ndarray]:
        assert self.mean is not None
        assert self.sum_squared_deviation is not None
        assert self.minimum is not None
        assert self.maximum is not None
        variance = np.maximum(0.0, self.sum_squared_deviation / self.count)
        return {
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "std": np.sqrt(variance),
            "count": np.asarray([self.count], dtype=np.int64),
        }


class NumericStatisticsValidator:
    """Recompute normalization statistics without retaining dataset frames."""

    def __init__(self, info: dict[str, Any], issue: IssueCallback) -> None:
        self._info = info
        self._issue = issue
        features = info.get("features")
        self._features = features if isinstance(features, dict) else {}
        self._running: dict[str, RunningMoments] = {}
        self._invalid_features: set[str] = set()

    def add_episode(self, data: pd.DataFrame, *, episode_index: int) -> None:
        for name, descriptor in self._features.items():
            if not isinstance(name, str) or not isinstance(descriptor, dict):
                continue
            dtype = descriptor.get("dtype")
            if dtype in VISUAL_DTYPES or dtype in NON_NUMERIC_DTYPES:
                continue
            if name not in data.columns or name in self._invalid_features:
                continue
            try:
                expected_width = _feature_width(descriptor)
                values = _numeric_matrix(data[name], expected_width)
            except (TypeError, ValueError):
                self._invalid_features.add(name)
                self._issue(
                    "FAIL",
                    "stats_source_invalid",
                    f"Statistics cannot be recomputed from feature: {name}",
                    episode_index,
                )
                continue
            if not np.isfinite(values).all():
                self._invalid_features.add(name)
                self._issue(
                    "FAIL",
                    "stats_source_non_finite",
                    f"Statistics source contains NaN or infinity: {name}",
                    episode_index,
                )
                continue
            self._running.setdefault(name, RunningMoments()).update(values)

    def validate(
        self,
        root: Path,
        *,
        complete_dataset: bool,
        visual_statistics: Mapping[str, Mapping[str, Any]] | None = None,
        on_progress=None,
    ) -> StatisticsValidationSummary:
        if not complete_dataset:
            self._issue(
                "FAIL",
                "stats_recompute_incomplete",
                "Statistics cannot be verified because not every episode was read",
                None,
            )
            return StatisticsValidationSummary((), self._visual_feature_names())
        try:
            stored = _read_stats(root / "meta/stats.json")
        except (OSError, ValueError, json.JSONDecodeError):
            self._issue(
                "FAIL",
                "stats_load_failed",
                "Dataset statistics could not be loaded",
                None,
            )
            return StatisticsValidationSummary((), self._visual_feature_names())

        official_operation = _official_policy(root, self._issue)

        validated: list[str] = []
        unverified_visual = list(self._visual_feature_names())
        relative_expected = None
        relative_invalid = False
        try:
            from datasetui.relative_artifacts import (
                read_relative_profile,
                recompute_relative_artifact,
            )
            from datasetui.transform_errors import CurationTransformError

            relative_profile = read_relative_profile(root)
            if relative_profile is not None:
                relative_expected, absolute_numeric = recompute_relative_artifact(
                    root, self._info, relative_profile, on_progress=on_progress
                )
                absolute_backup = _read_stats(root / "meta/stats.absolute.json")
                for key, actual_stats in absolute_numeric.items():
                    if not _compare_relative_quantiles(
                        actual_stats,
                        absolute_backup.get(key, {}),
                        self._issue,
                        feature=f"absolute backup/{key}",
                    ):
                        relative_invalid = True
                for key in self._visual_feature_names():
                    if absolute_backup.get(key) != stored.get(key):
                        relative_invalid = True
                        self._issue(
                            "FAIL",
                            "relative_profile_invalid",
                            f"Relative output changed visual statistics: {key}",
                            None,
                        )
                if not _compare_relative_quantiles(
                    relative_expected,
                    relative_profile.get("statistics", {}),
                    self._issue,
                ):
                    relative_invalid = True
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            CurationTransformError,
        ) as exc:
            relative_invalid = True
            self._issue("FAIL", "relative_profile_invalid", str(exc), None)
        for name, descriptor in self._features.items():
            if not isinstance(name, str) or not isinstance(descriptor, dict):
                continue
            dtype = descriptor.get("dtype")
            if dtype in NON_NUMERIC_DTYPES:
                continue
            feature_stats = stored.get(name)
            if not isinstance(feature_stats, dict):
                self._issue(
                    "FAIL",
                    "stats_feature_missing",
                    f"Statistics are missing for feature: {name}",
                    None,
                )
                continue
            schema_valid = _validate_stat_schema(
                name, feature_stats, descriptor, self._issue
            )
            if dtype in VISUAL_DTYPES:
                actual_visual = (visual_statistics or {}).get(name)
                if (
                    not official_operation
                    and all(key in feature_stats for key in REQUIRED_STATISTICS)
                ):
                    _validate_visual_statistics(
                        name,
                        feature_stats,
                        actual_visual,
                        self._issue,
                    )
                if not schema_valid:
                    continue
                if actual_visual is None:
                    continue
                if official_operation:
                    _compare_feature_statistics(
                        name,
                        dict(actual_visual),
                        feature_stats,
                        self._issue,
                        mismatch_severity="WARN",
                        mismatch_code="official_visual_stats_difference",
                        mismatch_message=(
                            "Official per-episode sampled visual aggregate differs "
                            f"from decoded media: {name}"
                        ),
                        warning_statistic_keys=frozenset(REQUIRED_STATISTICS),
                    )
                    validated.append(name)
                    unverified_visual.remove(name)
                elif _compare_feature_statistics(
                    name, dict(actual_visual), feature_stats, self._issue
                ):
                    validated.append(name)
                    unverified_visual.remove(name)
                continue
            if not schema_valid:
                continue
            if name == "action" and (relative_expected is not None or relative_invalid):
                if relative_invalid:
                    continue
                actual = {
                    key: np.asarray(value) for key, value in relative_expected.items()
                }
                matches = _compare_feature_statistics(
                    name, actual, feature_stats, self._issue
                )
                quantiles_match = _compare_relative_quantiles(
                    relative_expected, feature_stats, self._issue
                )
                if matches and quantiles_match:
                    validated.append(name)
                continue
            running = self._running.get(name)
            if running is None or name in self._invalid_features:
                self._issue(
                    "FAIL",
                    "stats_feature_unavailable",
                    f"Feature data was unavailable for statistics verification: {name}",
                    None,
                )
                continue
            if official_operation and name in OFFICIAL_BOOKKEEPING_FEATURES:
                _compare_feature_statistics(
                    name,
                    running.values(),
                    feature_stats,
                    self._issue,
                    mismatch_severity="WARN",
                    mismatch_code="official_bookkeeping_stats_difference",
                    mismatch_message=(
                        "Pinned official LeRobot aggregate differs after output "
                        f"bookkeeping reindexing: {name}"
                    ),
                    warning_statistic_keys=frozenset(
                        {"min", "max", "mean", "std"}
                    ),
                )
                validated.append(name)
                continue
            if _compare_feature_statistics(
                name, running.values(), feature_stats, self._issue
            ):
                validated.append(name)

        if unverified_visual:
            self._issue(
                "WARN",
                "visual_stats_not_recomputed",
                "Image/video statistics could not be reproduced from the decoded media",
                None,
            )
        return StatisticsValidationSummary(tuple(validated), tuple(unverified_visual))

    def _visual_feature_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, descriptor in self._features.items()
            if isinstance(name, str)
            and isinstance(descriptor, dict)
            and descriptor.get("dtype") in VISUAL_DTYPES
        )


def _compare_relative_quantiles(
    actual: dict, stored: dict, issue: IssueCallback, *, feature: str = "action"
) -> bool:
    """Check the complete relative distribution, including exact quantiles."""
    valid = True
    for key in (*REQUIRED_STATISTICS, "q01", "q10", "q50", "q90", "q99"):
        try:
            observed = _stored_numeric_array(stored[key])
            expected = np.asarray(actual[key])
            matches = (
                observed.shape == expected.shape
                and np.isfinite(observed).all()
                and (
                    np.array_equal(observed, expected)
                    if key == "count"
                    else np.allclose(observed, expected, rtol=1e-5, atol=1e-7)
                )
            )
        except (KeyError, TypeError, ValueError):
            matches = False
        if not matches:
            issue(
                "FAIL",
                "relative_stats_mismatch",
                f"Relative artifact statistic differs or is missing: {feature}.{key}",
                None,
            )
            valid = False
    return valid


def _feature_width(descriptor: dict[str, Any]) -> int:
    shape = descriptor.get("shape")
    if not isinstance(shape, list) or not shape:
        raise ValueError("numeric feature shape is missing")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 1
        for item in shape
    ):
        raise ValueError("numeric feature shape is invalid")
    return math.prod(shape)


def _numeric_matrix(series: pd.Series, expected_width: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    for value in series:
        array = np.asarray(value)
        if not (
            np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.bool_)
        ):
            raise TypeError("declared numeric feature contains non-numeric values")
        flat = array.astype(np.float64, copy=False).reshape(-1)
        if len(flat) != expected_width:
            raise ValueError("feature width differs from metadata")
        rows.append(flat)
    if not rows:
        return np.empty((0, expected_width), dtype=np.float64)
    return np.stack(rows)


def _read_stats(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise OSError("statistics file is unavailable")
    if path.stat().st_size > MAX_STATS_BYTES:
        raise OSError("statistics file is too large")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("statistics must be an object")
    return value


def _validate_stat_schema(
    feature: str,
    stats: dict[str, Any],
    descriptor: dict[str, Any],
    issue: IssueCallback,
) -> bool:
    valid = True
    arrays: dict[str, np.ndarray] = {}
    for key in REQUIRED_STATISTICS:
        if key not in stats:
            issue(
                "FAIL",
                "stats_field_missing",
                f"Required statistic is missing: {feature}.{key}",
                None,
            )
            valid = False
            continue
        try:
            array = _stored_numeric_array(stats[key])
        except (TypeError, ValueError):
            array = np.asarray([])
        if array.ndim == 0 or array.size == 0 or not np.isfinite(array).all():
            issue(
                "FAIL",
                "stats_value_invalid",
                f"Statistic must be a non-empty finite array: {feature}.{key}",
                None,
            )
            valid = False
        else:
            arrays[key] = array
    if "count" in arrays:
        count = arrays["count"]
        if (
            count.shape != (1,)
            or np.any(count <= 0)
            or np.any(count != np.floor(count))
        ):
            issue(
                "FAIL",
                "stats_count_invalid",
                f"Statistic count must contain positive integers: {feature}",
                None,
            )
            valid = False
    declared_shape = descriptor.get("shape")
    dtype = descriptor.get("dtype")
    if dtype in VISUAL_DTYPES:
        expected_shape = (
            (declared_shape[-1], 1, 1)
            if isinstance(declared_shape, list)
            and len(declared_shape) == 3
            and isinstance(declared_shape[-1], int)
            and not isinstance(declared_shape[-1], bool)
            else None
        )
    else:
        expected_shape = (
            tuple(declared_shape)
            if isinstance(declared_shape, list)
            and declared_shape
            and all(
                isinstance(item, int) and not isinstance(item, bool) and item > 0
                for item in declared_shape
            )
            else None
        )
    value_shapes = [
        arrays[key].shape for key in ("min", "max", "mean", "std") if key in arrays
    ]
    if expected_shape is None or any(shape != expected_shape for shape in value_shapes):
        issue(
            "FAIL",
            "stats_shape_mismatch",
            f"Statistic shapes do not match feature metadata: {feature}",
            None,
        )
        valid = False
    if "std" in arrays and np.any(arrays["std"] < 0):
        issue(
            "FAIL",
            "stats_value_invalid",
            f"Standard deviation must not be negative: {feature}",
            None,
        )
        valid = False
    if "min" in arrays and "max" in arrays and np.any(arrays["min"] > arrays["max"]):
        issue(
            "FAIL",
            "stats_value_invalid",
            f"Statistic minimum exceeds maximum: {feature}",
            None,
        )
        valid = False
    return valid


def _compare_feature_statistics(
    feature: str,
    actual: dict[str, np.ndarray],
    stored: dict[str, Any],
    issue: IssueCallback,
    *,
    mismatch_severity: str = "FAIL",
    mismatch_code: str = "stats_value_mismatch",
    mismatch_message: str | None = None,
    warning_statistic_keys: frozenset[str] = frozenset(),
) -> bool:
    valid = True
    for key in REQUIRED_STATISTICS:
        observed = _stored_numeric_array(stored[key]).reshape(-1)
        expected = actual[key].reshape(-1)
        if key == "count":
            accepted_counts = [expected]
            if "_legacy_pixel_count" in actual:
                accepted_counts.append(
                    np.asarray(actual["_legacy_pixel_count"], dtype=np.float64).reshape(
                        -1
                    )
                )
            matches = observed.shape == (1,) and any(
                candidate.shape == (1,) and observed[0] == candidate[0]
                for candidate in accepted_counts
            )
        else:
            matches = observed.shape == expected.shape and np.allclose(
                observed, expected, rtol=1e-5, atol=1e-7
            )
        if not matches:
            severity = (
                mismatch_severity if key in warning_statistic_keys else "FAIL"
            )
            code = (
                mismatch_code
                if severity == mismatch_severity
                else "stats_value_mismatch"
            )
            issue(
                severity,
                code,
                (
                    f"{mismatch_message} ({feature}.{key})"
                    if mismatch_message
                    else f"Stored statistic differs from dataset values: {feature}.{key}"
                ),
                None,
            )
            valid = False
    return valid


def _validate_visual_statistics(
    feature: str,
    stats: dict[str, Any],
    actual: Mapping[str, Any] | None,
    issue: IssueCallback,
) -> None:
    try:
        count = _stored_numeric_array(stats["count"]).reshape(-1)
    except (TypeError, ValueError):
        return
    accepted_counts: list[np.ndarray] = []
    if actual is not None and "count" in actual:
        accepted_counts.append(
            np.asarray(actual["count"], dtype=np.float64).reshape(-1)
        )
    if actual is not None and "_legacy_pixel_count" in actual:
        accepted_counts.append(
            np.asarray(actual["_legacy_pixel_count"], dtype=np.float64).reshape(-1)
        )
    if count.shape != (1,) or (
        accepted_counts
        and not any(
            candidate.shape == (1,) and count[0] == candidate[0]
            for candidate in accepted_counts
        )
    ):
        issue(
            "FAIL",
            "stats_count_mismatch",
            f"Visual statistic count differs from the deterministic sample count: {feature}",
            None,
        )


def _stored_numeric_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise TypeError("statistics must contain JSON numbers, not coerced values")
    return array.astype(np.float64, copy=False)
