from __future__ import annotations

import json
import math
import os
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from datasetui.transform_errors import CurationTransformError


BATCH_ROWS = 4096
MAX_INFO_BYTES = 16 * 1024 * 1024
NON_NUMERIC_DTYPES = {"image", "language", "string", "video"}
QUANTILES = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}
ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass
class _StableMoments:
    width: int
    count: int = 0
    mean: np.ndarray | None = None
    sum_squared_deviation: np.ndarray | None = None
    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None

    def update(self, values: np.ndarray) -> None:
        if len(values) == 0:
            return
        promoted = np.asarray(values, dtype=np.longdouble)
        batch_count = len(promoted)
        batch_mean = np.mean(promoted, axis=0, dtype=np.longdouble)
        batch_delta = promoted - batch_mean
        batch_squared_deviation = np.sum(
            batch_delta * batch_delta, axis=0, dtype=np.longdouble
        )
        batch_minimum = np.min(promoted, axis=0)
        batch_maximum = np.max(promoted, axis=0)
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.sum_squared_deviation = batch_squared_deviation
            self.minimum = batch_minimum
            self.maximum = batch_maximum
            return
        assert self.mean is not None
        assert self.sum_squared_deviation is not None
        assert self.minimum is not None
        assert self.maximum is not None
        combined_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.sum_squared_deviation += (
            batch_squared_deviation
            + delta * delta * self.count * batch_count / combined_count
        )
        self.mean += delta * batch_count / combined_count
        self.minimum = np.minimum(self.minimum, batch_minimum)
        self.maximum = np.maximum(self.maximum, batch_maximum)
        self.count = combined_count

    def finish(self) -> dict[str, np.ndarray]:
        if self.count == 0:
            raise CurationTransformError("Numeric statistics source is empty")
        assert self.mean is not None
        assert self.sum_squared_deviation is not None
        assert self.minimum is not None
        assert self.maximum is not None
        variance = np.maximum(np.longdouble(0), self.sum_squared_deviation / self.count)
        return {
            "min": np.asarray(self.minimum, dtype=np.float64),
            "max": np.asarray(self.maximum, dtype=np.float64),
            "mean": np.asarray(self.mean, dtype=np.float64),
            "std": np.asarray(np.sqrt(variance), dtype=np.float64),
        }


def recompute_numeric_statistics(
    root: Path,
    on_progress: ProgressCallback | None = None,
    *,
    episode_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    root = Path(root)
    info = _read_info(root / "meta" / "info.json")
    total_frames = info.get("total_frames")
    if (
        isinstance(total_frames, bool)
        or not isinstance(total_frames, int)
        or total_frames < 1
    ):
        raise CurationTransformError("Dataset total_frames is invalid")
    features = info.get("features")
    if not isinstance(features, dict):
        raise CurationTransformError("Dataset feature metadata is invalid")
    numeric_features = _numeric_features(features)
    parquet_files = _parquet_files(root / "data")
    selected = _episode_selection(episode_indices)
    measured_frames = (
        total_frames
        if selected is None
        else _selected_frame_count(parquet_files, selected, total_frames)
    )
    total_measurements = measured_frames * len(numeric_features)
    measured = 0
    _progress(
        on_progress,
        completed=0,
        total=total_measurements,
        current_item=(numeric_features[0][0] if numeric_features else "수치 특성"),
        force=True,
    )
    results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="datasetui-exact-stats-") as scratch:
        scratch_root = Path(scratch)
        for feature_index, (name, descriptor, shape, width) in enumerate(
            numeric_features
        ):
            storage = np.memmap(
                scratch_root / f"feature-{feature_index:04d}.float64",
                mode="w+",
                dtype=np.float64,
                shape=(width, measured_frames),
            )
            moments = _StableMoments(width)
            feature_count = 0
            try:
                for path in parquet_files:
                    with _open_parquet(path) as parquet:
                        _validate_physical_type(parquet.schema_arrow, name, descriptor)
                        columns = [name]
                        if selected is not None and name != "episode_index":
                            columns.append("episode_index")
                        for batch in parquet.iter_batches(
                            batch_size=BATCH_ROWS, columns=columns
                        ):
                            values = _numeric_batch(
                                batch.column(batch.schema.get_field_index(name)),
                                name,
                                width,
                                mask=(
                                    None
                                    if selected is None
                                    else _selection_mask(batch, selected)
                                ),
                            )
                            batch_count = len(values)
                            if feature_count + batch_count > measured_frames:
                                raise CurationTransformError(
                                    "Numeric feature row count exceeds expected selection: "
                                    f"{name}"
                                )
                            storage[:, feature_count : feature_count + batch_count] = (
                                values.T
                            )
                            moments.update(values)
                            feature_count += batch_count
                            _progress(
                                on_progress,
                                completed=measured + feature_count,
                                total=total_measurements,
                                current_item=name,
                            )
                if feature_count != measured_frames:
                    raise CurationTransformError(
                        f"Numeric feature row count differs from expected selection: "
                        f"{name} ({feature_count} != {measured_frames})"
                    )
                storage.flush()
                statistics = moments.finish()
                statistics.update(_exact_quantiles(storage, measured_frames))
                results[name] = {
                    key: _reshape(values, shape) for key, values in statistics.items()
                }
                results[name]["count"] = [feature_count]
            finally:
                del storage
            measured += feature_count
            _progress(
                on_progress,
                completed=measured,
                total=total_measurements,
                current_item=name,
                force=True,
            )
    return results


def _numeric_features(
    features: dict[str, Any],
) -> list[tuple[str, dict[str, Any], tuple[int, ...], int]]:
    result: list[tuple[str, dict[str, Any], tuple[int, ...], int]] = []
    for name, descriptor in features.items():
        if not isinstance(name, str) or not isinstance(descriptor, dict):
            raise CurationTransformError("Dataset feature metadata is invalid")
        dtype = descriptor.get("dtype")
        if dtype in NON_NUMERIC_DTYPES:
            continue
        if not isinstance(dtype, str) or _arrow_type(dtype) is None:
            raise CurationTransformError(
                f"Unsupported declared numeric dtype for feature: {name}"
            )
        raw_shape = descriptor.get("shape")
        if (
            not isinstance(raw_shape, list)
            or not raw_shape
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in raw_shape
            )
        ):
            raise CurationTransformError(f"Numeric feature shape is invalid: {name}")
        shape = tuple(raw_shape)
        result.append((name, descriptor, shape, math.prod(shape)))
    return result


def _read_info(path: Path) -> dict[str, Any]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise CurationTransformError("Dataset info.json is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_INFO_BYTES:
            raise CurationTransformError("Dataset info.json is invalid")
        raw = b""
        while chunk := os.read(descriptor, 64 * 1024):
            raw += chunk
            if len(raw) > MAX_INFO_BYTES:
                raise CurationTransformError("Dataset info.json is invalid")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CurationTransformError("Dataset info.json is invalid") from exc
    if not isinstance(value, dict):
        raise CurationTransformError("Dataset info.json is invalid")
    return value


def _parquet_files(data_root: Path) -> list[Path]:
    if data_root.is_symlink() or not data_root.is_dir():
        raise CurationTransformError("Dataset data directory is unavailable")
    files: list[Path] = []
    for current, directories, names in os.walk(data_root, followlinks=False):
        current_path = Path(current)
        directories.sort()
        for name in directories:
            if (current_path / name).is_symlink():
                raise CurationTransformError("Dataset data contains a symlink")
        for name in sorted(names):
            path = current_path / name
            if path.is_symlink():
                raise CurationTransformError("Dataset data contains a symlink")
            if name.endswith(".parquet"):
                if not path.is_file():
                    raise CurationTransformError("Dataset data file is invalid")
                files.append(path)
    if not files:
        raise CurationTransformError("Dataset contains no Parquet data files")
    return files


@contextmanager
def _open_parquet(path: Path) -> Iterator[pq.ParquetFile]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise CurationTransformError("Dataset Parquet file is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CurationTransformError("Dataset Parquet file is invalid")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            yield pq.ParquetFile(stream)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise CurationTransformError(
                "Dataset Parquet file changed during statistics recomputation"
            )
    finally:
        os.close(descriptor)


def _validate_physical_type(
    schema: pa.Schema, name: str, descriptor: dict[str, Any]
) -> None:
    index = schema.get_field_index(name)
    if index < 0:
        raise CurationTransformError(f"Numeric feature column is missing: {name}")
    physical = schema.field(index).type
    while (
        pa.types.is_list(physical)
        or pa.types.is_large_list(physical)
        or pa.types.is_fixed_size_list(physical)
    ):
        physical = physical.value_type
    expected = _arrow_type(str(descriptor["dtype"]))
    if physical != expected:
        raise CurationTransformError(
            f"Numeric feature dtype differs from metadata: {name} "
            f"({physical} != {expected})"
        )


def _arrow_type(dtype: str) -> pa.DataType | None:
    return {
        "bool": pa.bool_(),
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
    }.get(dtype)


def _episode_selection(episode_indices: Sequence[int] | None) -> set[int] | None:
    if episode_indices is None:
        return None
    selected: set[int] = set()
    for value in episode_indices:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CurationTransformError("Episode statistics selection is invalid")
        if value in selected:
            raise CurationTransformError(
                "Episode statistics selection contains duplicates"
            )
        selected.add(value)
    if not selected:
        raise CurationTransformError("Episode statistics selection cannot be empty")
    return selected


def _selected_frame_count(
    parquet_files: list[Path], selected: set[int], total_frames: int
) -> int:
    selected_count = 0
    all_count = 0
    seen: set[int] = set()
    for path in parquet_files:
        with _open_parquet(path) as parquet:
            index = parquet.schema_arrow.get_field_index("episode_index")
            if index < 0:
                raise CurationTransformError(
                    "Episode selection requires an episode_index column"
                )
            physical = parquet.schema_arrow.field(index).type
            if not pa.types.is_integer(physical):
                raise CurationTransformError("episode_index has a non-integer dtype")
            for batch in parquet.iter_batches(
                batch_size=BATCH_ROWS, columns=["episode_index"]
            ):
                values = batch.column(0).to_pylist()
                all_count += len(values)
                for value in values:
                    if isinstance(value, bool) or not isinstance(value, int):
                        raise CurationTransformError(
                            "episode_index contains an invalid value"
                        )
                    seen.add(value)
                    if value in selected:
                        selected_count += 1
    if all_count != total_frames:
        raise CurationTransformError(
            "Dataset row count differs from total_frames "
            f"({all_count} != {total_frames})"
        )
    missing = selected - seen
    if missing:
        raise CurationTransformError(
            "Unknown episode in numeric statistics selection: "
            + ", ".join(str(value) for value in sorted(missing))
        )
    if selected_count < 1:
        raise CurationTransformError("Selected episodes contain no numeric rows")
    return selected_count


def _selection_mask(batch: pa.RecordBatch, selected: set[int]) -> list[bool]:
    index = batch.schema.get_field_index("episode_index")
    if index < 0:
        raise CurationTransformError(
            "Episode selection requires an episode_index column"
        )
    return [value in selected for value in batch.column(index).to_pylist()]


def _numeric_batch(
    column: pa.Array,
    name: str,
    width: int,
    *,
    mask: list[bool] | None = None,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for row_index, value in enumerate(column.to_pylist()):
        if mask is not None and not mask[row_index]:
            continue
        array = np.asarray(value)
        if not (
            np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.bool_)
        ):
            raise CurationTransformError(
                f"Numeric feature contains a non-numeric value: {name} "
                f"(batch row {row_index})"
            )
        flat = array.reshape(-1)
        if len(flat) != width:
            raise CurationTransformError(
                f"Numeric feature shape differs from metadata: {name} "
                f"(batch row {row_index})"
            )
        promoted = flat.astype(np.float64, copy=False)
        if not np.isfinite(promoted).all():
            raise CurationTransformError(
                f"Numeric feature contains NaN or infinity: {name} "
                f"(batch row {row_index})"
            )
        rows.append(promoted)
    if not rows:
        return np.empty((0, width), dtype=np.float64)
    return np.stack(rows)


def _exact_quantiles(storage: np.memmap, count: int) -> dict[str, np.ndarray]:
    names = tuple(QUANTILES)
    probabilities = np.asarray(tuple(QUANTILES.values()), dtype=np.float64)
    positions = probabilities * (count - 1)
    lower = np.floor(positions).astype(np.int64)
    upper = np.ceil(positions).astype(np.int64)
    fractions = positions - lower
    selected = sorted(set(lower.tolist() + upper.tolist()))
    output = {name: np.empty(storage.shape[0], dtype=np.float64) for name in names}
    for dimension in range(storage.shape[0]):
        values = storage[dimension]
        values.partition(selected)
        quantiles = values[lower] + (values[upper] - values[lower]) * fractions
        for index, name in enumerate(names):
            output[name][dimension] = quantiles[index]
    return output


def _reshape(values: np.ndarray, shape: tuple[int, ...]) -> Any:
    return np.asarray(values, dtype=np.float64).reshape(shape).tolist()


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
        "stage": "statistics",
        "completed": completed,
        "total": total,
        "unit": "rows",
        "current_item": current_item,
    }
    if force:
        event["_force"] = True
    callback(event)
