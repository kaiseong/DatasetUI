from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from datasetui.merge_schema import (
    _compression_policy,
    _open_parquet,
    _parquet_files,
    _without_metadata,
)
from datasetui.transform_errors import CurationTransformError


ProgressCallback = Callable[[dict[str, Any]], None]
NORMALIZATION_ENGINE = "datasetui-merge-input-normalization-v1"
_VECTOR_COLUMNS = ("action", "observation.state")


def plan_merge_normalization(
    roots: Sequence[Path], on_progress: ProgressCallback | None = None
) -> dict[str, Any]:
    """Read and validate merge inputs without modifying them.

    The returned plan contains JSON-serializable source identities.  It can only be
    applied to separate private copies with ``normalize_private_merge_sources``.
    """
    if not roots:
        raise CurationTransformError("Merge normalization requires source datasets")
    resolved = [Path(root).resolve() for root in roots]
    if len(set(resolved)) != len(resolved):
        raise CurationTransformError("Merge normalization sources must be distinct")

    infos = [_read_info(root) for root in resolved]
    canonical_features, widths = _validate_features(infos, resolved)
    has_timestamp = "timestamp" in canonical_features
    expected_fields: list[pa.Field] | None = None
    datasets: list[dict[str, Any]] = []
    total_files = sum(len(_parquet_files(root / "data")) for root in resolved)
    completed = 0
    for dataset_index, root in enumerate(resolved):
        files = []
        for path in _parquet_files(root / "data"):
            relative = path.relative_to(root).as_posix()
            with _open_parquet(path) as parquet:
                fields = list(_without_metadata(parquet.schema_arrow))
                expected_fields = _validate_physical_schema(
                    fields,
                    expected_fields,
                    widths=widths,
                    has_timestamp=has_timestamp,
                    dataset=root,
                    file=relative,
                )
                rows = _validate_values(
                    parquet,
                    path=path,
                    dataset=root,
                    widths=widths,
                    has_timestamp=has_timestamp,
                )
            files.append(
                {
                    "path": relative,
                    "rows": rows,
                    "sha256": _sha256(path),
                }
            )
            completed += 1
            _progress(on_progress, completed, total_files, relative)
        datasets.append(
            {
                "index": dataset_index,
                "root": str(root),
                "info_sha256": _sha256(root / "meta/info.json"),
                "files": files,
            }
        )
    assert expected_fields is not None
    return {
        "engine": NORMALIZATION_ENGINE,
        "datasets": datasets,
        "features": canonical_features,
        "feature_order": list(canonical_features),
        "vector_widths": widths,
        "canonical": {
            **({"timestamp": "float64"} if has_timestamp else {}),
            **{name: "list<float32>" for name in widths},
        },
        "files_checked": total_files,
        "rows_checked": sum(
            file["rows"] for dataset in datasets for file in dataset["files"]
        ),
    }


def normalize_private_merge_sources(
    roots: Sequence[Path],
    plan: dict[str, Any],
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Normalize validated private copies and atomically replace only Parquet/info files."""
    if plan.get("engine") != NORMALIZATION_ENGINE:
        raise CurationTransformError("Merge normalization plan is invalid")
    copies = [Path(root).resolve() for root in roots]
    planned = plan.get("datasets")
    if not isinstance(planned, list) or len(copies) != len(planned):
        raise CurationTransformError("Private merge sources do not match the plan")
    original_roots = [Path(item["root"]).resolve() for item in planned]
    for copy_root, original_root in zip(copies, original_roots, strict=True):
        if copy_root == original_root:
            raise CurationTransformError(
                "Merge normalization refuses to modify an original source dataset"
            )

    widths = plan.get("vector_widths")
    if not isinstance(widths, dict) or any(
        name not in _VECTOR_COLUMNS or not isinstance(width, int)
        for name, width in widths.items()
    ):
        raise CurationTransformError("Merge normalization plan dimensions are invalid")
    total_files = sum(len(item["files"]) for item in planned)
    completed = 0
    rewritten: list[str] = []
    for copy_root, dataset_plan in zip(copies, planned, strict=True):
        _verify_private_copy(copy_root, dataset_plan)
        for file_plan in dataset_plan["files"]:
            path = copy_root / file_plan["path"]
            changed = _normalize_file(path, copy_root, widths)
            if changed:
                rewritten.append(f"{dataset_plan['index']}:{file_plan['path']}")
            completed += 1
            _progress(on_progress, completed, total_files, file_plan["path"])
        _normalize_info(copy_root, plan["features"])
    return {
        "engine": NORMALIZATION_ENGINE,
        "datasets_normalized": len(copies),
        "files_checked": total_files,
        "files_rewritten": rewritten,
        "canonical": copy.deepcopy(plan["canonical"]),
    }


def _validate_features(
    infos: list[dict[str, Any]], roots: list[Path]
) -> tuple[dict[str, Any], dict[str, int]]:
    feature_sets = []
    for info, root in zip(infos, roots, strict=True):
        features = info.get("features")
        if not isinstance(features, dict):
            raise CurationTransformError(f"Dataset features are invalid: {root}")
        feature_sets.append(features)
    expected_order = list(feature_sets[0])
    for root, features in zip(roots[1:], feature_sets[1:], strict=True):
        if list(features) != expected_order:
            raise CurationTransformError(
                f"Dataset feature order differs: dataset={root}; "
                f"expected={expected_order}; actual={list(features)}"
            )
    widths: dict[str, int] = {}
    for name in _VECTOR_COLUMNS:
        first = feature_sets[0].get(name)
        if first is None:
            continue
        if not isinstance(first, dict):
            raise CurationTransformError(f"Feature metadata is invalid: {name}")
        if first.get("dtype") != "float32":
            continue
        shape = first.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 1
            or isinstance(shape[0], bool)
            or not isinstance(shape[0], int)
            or shape[0] <= 0
        ):
            raise CurationTransformError(
                f"Feature shape is invalid: dataset={roots[0]}; column={name}; "
                f"expected=[positive width]; actual={shape}"
            )
        widths[name] = shape[0]
    timestamp = feature_sets[0].get("timestamp")
    if timestamp is not None and (
        not isinstance(timestamp, dict)
        or timestamp.get("dtype") not in {"float32", "float64"}
    ):
        actual = timestamp.get("dtype") if isinstance(timestamp, dict) else None
        raise CurationTransformError(
            f"Feature dtype is incompatible: dataset={roots[0]}; column=timestamp; "
            f"expected=float32 or float64; actual={actual}"
        )
    canonical = copy.deepcopy(feature_sets[0])
    if timestamp is not None:
        canonical["timestamp"]["dtype"] = "float64"
    for root, features in zip(roots, feature_sets, strict=True):
        for name in expected_order:
            actual = copy.deepcopy(features[name])
            expected = copy.deepcopy(canonical[name])
            if name == "timestamp" and timestamp is not None and isinstance(actual, dict):
                if actual.get("dtype") not in {"float32", "float64"}:
                    raise CurationTransformError(
                        f"Feature dtype is incompatible: dataset={root}; "
                        f"column=timestamp; expected=float32 or float64; "
                        f"actual={actual.get('dtype')}"
                    )
                actual["dtype"] = "float64"
            if actual != expected:
                raise CurationTransformError(
                    f"Dataset feature metadata differs: dataset={root}; column={name}; "
                    f"expected={expected}; actual={features[name]}"
                )
    return canonical, widths


def _validate_physical_schema(
    fields: list[pa.Field],
    expected_fields: list[pa.Field] | None,
    *,
    widths: dict[str, int],
    has_timestamp: bool,
    dataset: Path,
    file: str,
) -> list[pa.Field]:
    names = [field.name for field in fields]
    normalized_names = set(widths)
    if has_timestamp:
        normalized_names.add("timestamp")
    if expected_fields is not None and names != [field.name for field in expected_fields]:
        raise CurationTransformError(
            f"Parquet column order differs: dataset={dataset}; file={file}; "
            f"expected={[field.name for field in expected_fields]}; actual={names}"
        )
    for field in fields:
        if field.name in widths:
            _validate_vector_type(field.type, widths[field.name], dataset, file, field.name)
        elif field.name == "timestamp" and has_timestamp:
            if not (pa.types.is_float32(field.type) or pa.types.is_float64(field.type)):
                raise CurationTransformError(
                    f"Parquet dtype is incompatible: dataset={dataset}; file={file}; "
                    f"column=timestamp; expected=float32 or float64; actual={field.type}"
                )
        if expected_fields is not None:
            expected = expected_fields[names.index(field.name)]
            if field.nullable != expected.nullable:
                raise CurationTransformError(
                    f"Parquet nullability differs: dataset={dataset}; file={file}; "
                    f"column={field.name}; expected={expected.nullable}; actual={field.nullable}"
                )
            if field.name not in normalized_names and field.type != expected.type:
                raise CurationTransformError(
                    f"Parquet dtype differs: dataset={dataset}; file={file}; "
                    f"column={field.name}; expected={expected.type}; actual={field.type}"
                )
    required = [*widths]
    if has_timestamp:
        required.append("timestamp")
    if any(name not in names for name in required):
        missing = [name for name in required if name not in names]
        raise CurationTransformError(
            f"Required Parquet columns are missing: dataset={dataset}; file={file}; "
            f"columns={missing}"
        )
    return fields if expected_fields is None else expected_fields


def _validate_vector_type(
    dtype: pa.DataType, width: int, dataset: Path, file: str, column: str
) -> None:
    supported = pa.types.is_list(dtype) or pa.types.is_fixed_size_list(dtype)
    child = dtype.value_type if supported else None
    fixed_width_ok = not pa.types.is_fixed_size_list(dtype) or dtype.list_size == width
    if not supported or not pa.types.is_float32(child) or not fixed_width_ok:
        raise CurationTransformError(
            f"Parquet dtype is incompatible: dataset={dataset}; file={file}; "
            f"column={column}; expected=list<float32> with width {width}; actual={dtype}"
        )


def _validate_values(
    parquet: pq.ParquetFile,
    *,
    path: Path,
    dataset: Path,
    widths: dict[str, int],
    has_timestamp: bool,
) -> int:
    row_offset = 0
    columns = [*widths]
    if has_timestamp:
        columns.append("timestamp")
    for batch in parquet.iter_batches(batch_size=65536, columns=columns):
        for name in widths:
            values = batch.column(batch.schema.get_field_index(name)).to_pylist()
            for local_row, value in enumerate(values):
                row = row_offset + local_row
                if value is None:
                    raise CurationTransformError(
                        f"Null array is not allowed: dataset={dataset}; file={path}; "
                        f"column={name}; row={row}"
                    )
                if len(value) != widths[name]:
                    raise CurationTransformError(
                        f"Vector length differs: dataset={dataset}; file={path}; "
                        f"column={name}; row={row}; expected={widths[name]}; "
                        f"actual={len(value)}"
                    )
                for element_index, element in enumerate(value):
                    if element is None:
                        raise CurationTransformError(
                            f"Null vector element is not allowed: dataset={dataset}; "
                            f"file={path}; column={name}; row={row}; "
                            f"element={element_index}"
                        )
                    if not math.isfinite(element):
                        raise CurationTransformError(
                            f"Vector element must be finite: dataset={dataset}; "
                            f"file={path}; column={name}; row={row}; "
                            f"element={element_index}; actual={element}"
                        )
        if has_timestamp:
            timestamps = batch.column(batch.schema.get_field_index("timestamp")).to_pylist()
            for local_row, value in enumerate(timestamps):
                row = row_offset + local_row
                if value is None or not math.isfinite(value):
                    raise CurationTransformError(
                        f"Timestamp must be finite: dataset={dataset}; file={path}; "
                        f"column=timestamp; row={row}; actual={value}"
                    )
        row_offset += batch.num_rows
    return row_offset


def _verify_private_copy(root: Path, dataset_plan: dict[str, Any]) -> None:
    if root.is_symlink() or not root.is_dir():
        raise CurationTransformError(f"Private merge source is unavailable: {root}")
    if _sha256(root / "meta/info.json") != dataset_plan["info_sha256"]:
        raise CurationTransformError(f"Private merge metadata differs from plan: {root}")
    actual_files = [
        path.relative_to(root).as_posix() for path in _parquet_files(root / "data")
    ]
    expected_files = [item["path"] for item in dataset_plan["files"]]
    if actual_files != expected_files:
        raise CurationTransformError(
            f"Private merge Parquet inventory differs: dataset={root}; "
            f"expected={expected_files}; actual={actual_files}"
        )
    for item in dataset_plan["files"]:
        path = root / item["path"]
        if _sha256(path) != item["sha256"]:
            raise CurationTransformError(
                f"Private merge file differs from plan: dataset={root}; file={item['path']}"
            )


def _normalize_file(path: Path, root: Path, widths: dict[str, int]) -> bool:
    with _open_parquet(path) as source:
        target_schema = _canonical_schema(source.schema_arrow, widths)
        if source.schema_arrow.equals(target_schema, check_metadata=True):
            return False
        compression = _compression_policy(source)
        row_groups = source.metadata.num_row_groups
    mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.normalize-", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with _open_parquet(path) as source:
            with pq.ParquetWriter(temporary, target_schema, compression=compression) as writer:
                for index in range(source.metadata.num_row_groups):
                    table = source.read_row_group(index)
                    arrays = []
                    for field in target_schema:
                        column = table.column(field.name).combine_chunks()
                        if field.name in widths:
                            column = pa.array(column.to_pylist(), type=field.type, safe=True)
                        elif field.name == "timestamp":
                            column = column.cast(pa.float64(), safe=True)
                        arrays.append(column)
                    writer.write_table(
                        pa.Table.from_arrays(arrays, schema=target_schema),
                        row_group_size=len(table),
                    )
        _fsync(temporary)
        _verify_rewrite(path, temporary, row_groups)
        os.chmod(temporary, mode, follow_symlinks=False)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return True


def _canonical_schema(schema: pa.Schema, widths: dict[str, int]) -> pa.Schema:
    fields = []
    for field in schema:
        dtype = field.type
        if field.name in widths:
            dtype = pa.list_(pa.float32())
        elif field.name == "timestamp":
            dtype = pa.float64()
        fields.append(
            pa.field(field.name, dtype, nullable=field.nullable, metadata=field.metadata)
        )
    return pa.schema(
        fields,
        metadata=_canonical_huggingface_metadata(schema.metadata, widths),
    )


def _canonical_huggingface_metadata(
    metadata: dict[bytes, bytes] | None,
    widths: dict[str, int],
) -> dict[bytes, bytes] | None:
    if not metadata or b"huggingface" not in metadata:
        return metadata
    updated = dict(metadata)
    try:
        payload = json.loads(updated[b"huggingface"])
        features = payload["info"]["features"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CurationTransformError("Parquet Hugging Face feature metadata is invalid") from exc
    timestamp = features.get("timestamp")
    if timestamp is not None:
        if not isinstance(timestamp, dict):
            raise CurationTransformError("Parquet Hugging Face timestamp metadata is invalid")
        timestamp["dtype"] = "float64"
    for name in widths:
        descriptor = features.get(name)
        if descriptor is None:
            continue
        if not isinstance(descriptor, dict) or descriptor.get("feature", {}).get("dtype") != "float32":
            continue
        descriptor["_type"] = "List"
        descriptor.pop("length", None)
        descriptor["feature"] = {"dtype": "float32", "_type": "Value"}
    updated[b"huggingface"] = json.dumps(payload, separators=(",", ":")).encode()
    return updated


def _verify_rewrite(original: Path, candidate: Path, expected_row_groups: int) -> None:
    with _open_parquet(original) as before, _open_parquet(candidate) as after:
        if before.metadata.num_row_groups != expected_row_groups or after.metadata.num_row_groups != expected_row_groups:
            raise CurationTransformError(f"Parquet row groups changed: file={original}")
        if before.metadata.num_rows != after.metadata.num_rows:
            raise CurationTransformError(f"Parquet row count changed: file={original}")
        for index in range(expected_row_groups):
            left = before.read_row_group(index)
            right = after.read_row_group(index)
            for name in left.column_names:
                left_column = left.column(name).combine_chunks()
                right_column = right.column(name).combine_chunks()
                if name in _VECTOR_COLUMNS and (
                    pa.types.is_list(right_column.type)
                    and pa.types.is_float32(right_column.type.value_type)
                ):
                    equal = left_column.to_pylist() == right_column.to_pylist()
                elif name == "timestamp":
                    equal = left_column.cast(pa.float64(), safe=True).equals(right_column)
                else:
                    equal = left_column.equals(right_column)
                if not equal:
                    raise CurationTransformError(
                        f"Parquet values changed during normalization: file={original}; "
                        f"column={name}; row_group={index}"
                    )


def _normalize_info(root: Path, canonical_features: dict[str, Any]) -> None:
    path = root / "meta/info.json"
    info = _read_info(root)
    features = info.get("features")
    if not isinstance(features, dict):
        raise CurationTransformError(f"Dataset features are invalid: {root}")
    if "timestamp" in canonical_features:
        features["timestamp"] = copy.deepcopy(canonical_features["timestamp"])
    _atomic_json(path, info)


def _read_info(root: Path) -> dict[str, Any]:
    path = root / "meta/info.json"
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            info = json.load(stream)
    except (OSError, ValueError) as exc:
        raise CurationTransformError(f"Dataset info.json is unavailable or invalid: {root}") from exc
    if not isinstance(info, dict):
        raise CurationTransformError(f"Dataset info.json is invalid: {root}")
    return info


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.normalize-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode, follow_symlinks=False)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise CurationTransformError(f"Dataset file is unavailable: {path}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise CurationTransformError(f"Dataset file is invalid: {path}")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _progress(
    callback: ProgressCallback | None, completed: int, total: int, item: str
) -> None:
    if callback:
        callback(
            {
                "stage": "merge_preflight",
                "completed": completed,
                "total": total,
                "unit": "files",
                "current_item": item,
            }
        )
