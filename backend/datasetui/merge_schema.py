from __future__ import annotations

import hashlib
import math
import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from datasetui.transform_errors import CurationTransformError


REPAIR_ENGINE = "datasetui-merge-schema-restore-v1"


def inspect_compatible_data_schema(source_roots: Sequence[Path]) -> pa.Schema:
    if not source_roots:
        raise CurationTransformError("Merge schema inspection requires source datasets")
    expected: pa.Schema | None = None
    expected_location: str | None = None
    for root in source_roots:
        for path in _parquet_files(Path(root) / "data"):
            with _open_parquet(path) as parquet:
                schema = _without_metadata(parquet.schema_arrow)
            if expected is None:
                expected = schema
                expected_location = str(path)
                continue
            if not _schemas_equal(expected, schema):
                raise CurationTransformError(
                    "Merge source data schemas are incompatible: "
                    f"{expected_location} != {path}"
                )
    if expected is None:
        raise CurationTransformError("Merge sources contain no Parquet data files")
    return expected


def restore_merged_data_schema(
    *, source_roots: Sequence[Path], output_root: Path
) -> dict[str, Any]:
    expected = inspect_compatible_data_schema(source_roots)
    output_root = Path(output_root)
    output_files = _parquet_files(output_root / "data")
    plans: list[tuple[Path, dict[str, pa.DataType]]] = []
    total_rows = 0
    total_row_groups = 0
    for path in output_files:
        with _open_parquet(path) as parquet:
            repairs = _repair_plan(expected, parquet.schema_arrow, path)
            total_rows += parquet.metadata.num_rows
            total_row_groups += parquet.metadata.num_row_groups
            _validate_repair_values(parquet, repairs, path)
        plans.append((path, repairs))

    repaired_files: list[str] = []
    repaired_columns: set[str] = set()
    for path, repairs in plans:
        if not repairs:
            continue
        _rewrite_file(path, expected, repairs)
        repaired_files.append(path.relative_to(output_root).as_posix())
        repaired_columns.update(repairs)

    return {
        "engine": REPAIR_ENGINE,
        "source_schema_sha256": _schema_sha256(expected),
        "output_files_checked": len(output_files),
        "output_rows_checked": total_rows,
        "output_row_groups_checked": total_row_groups,
        "repaired_files": repaired_files,
        "repaired_columns": sorted(repaired_columns),
        "repair_kind": "variable-list-to-fixed-size-list",
    }


def _repair_plan(
    expected: pa.Schema, actual: pa.Schema, path: Path
) -> dict[str, pa.DataType]:
    if expected.names != actual.names:
        raise CurationTransformError(
            f"Merged output columns differ from source schema: {path}"
        )
    repairs: dict[str, pa.DataType] = {}
    for expected_field, actual_field in zip(expected, actual, strict=True):
        if expected_field.nullable != actual_field.nullable:
            raise CurationTransformError(
                "Merged output nullability differs from source schema: "
                f"{path} · {expected_field.name}"
            )
        if expected_field.type == actual_field.type:
            continue
        if (
            pa.types.is_fixed_size_list(expected_field.type)
            and pa.types.is_list(actual_field.type)
            and expected_field.type.value_type == actual_field.type.value_type
        ):
            repairs[expected_field.name] = expected_field.type
            continue
        raise CurationTransformError(
            "Merged output contains an unsupported physical schema change: "
            f"{path} · {expected_field.name} "
            f"({actual_field.type} != {expected_field.type})"
        )
    return repairs


def _validate_repair_values(
    parquet: pq.ParquetFile, repairs: dict[str, pa.DataType], path: Path
) -> None:
    if not repairs:
        return
    for row_group in range(parquet.metadata.num_row_groups):
        table = parquet.read_row_group(row_group, columns=list(repairs))
        for name, expected_type in repairs.items():
            source = table.column(name).combine_chunks()
            _restore_array(source, expected_type, path=path, column=name)


def _restore_array(
    source: pa.Array, expected_type: pa.DataType, *, path: Path, column: str
) -> pa.Array:
    assert pa.types.is_fixed_size_list(expected_type)
    list_size = expected_type.list_size
    if source.type.value_type != expected_type.value_type:
        raise CurationTransformError(
            f"Merged output list child dtype differs from source: {path} · {column}"
        )
    offsets = source.offsets.to_numpy(zero_copy_only=False)
    lengths = offsets[1:] - offsets[:-1]
    for row_index, length in enumerate(lengths.tolist()):
        if source.is_null()[row_index].as_py():
            continue
        if length != list_size:
            raise CurationTransformError(
                "Merged output list length differs from source fixed size: "
                f"{path} · {column} · row {row_index} ({length} != {list_size})"
            )
    try:
        restored = pa.array(source.to_pylist(), type=expected_type, safe=True)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as exc:
        raise CurationTransformError(
            f"Merged output list cannot be safely restored: {path} · {column}"
        ) from exc
    if not _values_equal(source.to_pylist(), restored.to_pylist()):
        raise CurationTransformError(
            f"Merged output values changed during schema restoration: {path} · {column}"
        )
    return restored


def _rewrite_file(
    path: Path, expected: pa.Schema, repairs: dict[str, pa.DataType]
) -> None:
    original_mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.schema-", suffix=".tmp", dir=path.parent
    )
    os.close(temporary_fd)
    temporary = Path(temporary_name)
    try:
        with _open_parquet(path) as parquet:
            if _repair_plan(expected, parquet.schema_arrow, path) != repairs:
                raise CurationTransformError(
                    f"Merged output schema changed before restoration: {path}"
                )
            target_schema = _target_schema(expected, parquet.schema_arrow)
            compression = _compression_policy(parquet)
            with pq.ParquetWriter(
                temporary,
                target_schema,
                compression=compression,
            ) as writer:
                for row_group in range(parquet.metadata.num_row_groups):
                    table = parquet.read_row_group(row_group)
                    columns = []
                    for field in target_schema:
                        column = table.column(field.name).combine_chunks()
                        if field.name in repairs:
                            column = _restore_array(
                                column,
                                repairs[field.name],
                                path=path,
                                column=field.name,
                            )
                        columns.append(column)
                    repaired = pa.Table.from_arrays(columns, schema=target_schema)
                    writer.write_table(repaired, row_group_size=len(repaired))
        descriptor = os.open(temporary, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        with _open_parquet(temporary) as candidate:
            if not _schemas_equal(expected, candidate.schema_arrow):
                raise CurationTransformError(
                    f"Merged output schema restoration failed: {path}"
                )
            _verify_row_groups(path, candidate, repairs)
        os.chmod(temporary, original_mode, follow_symlinks=False)
        os.replace(temporary, path)
        directory = os.open(
            path.parent,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY,
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _target_schema(expected: pa.Schema, actual: pa.Schema) -> pa.Schema:
    fields = [
        pa.field(
            expected_field.name,
            expected_field.type,
            nullable=expected_field.nullable,
            metadata=actual.field(expected_field.name).metadata,
        )
        for expected_field in expected
    ]
    return pa.schema(fields, metadata=actual.metadata)


def _verify_row_groups(
    original: Path,
    candidate: pq.ParquetFile,
    repairs: dict[str, pa.DataType],
) -> None:
    with _open_parquet(original) as source:
        if source.metadata.num_rows != candidate.metadata.num_rows:
            raise CurationTransformError(
                f"Merged output row count changed during schema restoration: {original}"
            )
        if source.metadata.num_row_groups != candidate.metadata.num_row_groups:
            raise CurationTransformError(
                "Merged output row groups changed during schema restoration: "
                f"{original}"
            )
        for index in range(source.metadata.num_row_groups):
            if (
                source.metadata.row_group(index).num_rows
                != candidate.metadata.row_group(index).num_rows
            ):
                raise CurationTransformError(
                    "Merged output row group size changed during schema restoration: "
                    f"{original} · {index}"
                )
            before = source.read_row_group(index)
            after = candidate.read_row_group(index)
            for name in before.column_names:
                before_column = before.column(name)
                after_column = after.column(name)
                values_match = (
                    _values_equal(before_column.to_pylist(), after_column.to_pylist())
                    if name in repairs
                    else before_column.equals(after_column)
                )
                if not values_match:
                    raise CurationTransformError(
                        "Merged output values changed during schema restoration: "
                        f"{original} · {name} · row group {index}"
                    )


def _compression_policy(parquet: pq.ParquetFile) -> Any:
    codecs_by_path: dict[str, set[str]] = {}
    for row_group in range(parquet.metadata.num_row_groups):
        metadata = parquet.metadata.row_group(row_group)
        for column in range(metadata.num_columns):
            chunk = metadata.column(column)
            codecs_by_path.setdefault(chunk.path_in_schema, set()).add(
                chunk.compression.lower()
            )
    if not codecs_by_path:
        raise CurationTransformError(
            "Merged output contains no Parquet columns to repair"
        )
    if any(len(codecs) != 1 for codecs in codecs_by_path.values()):
        raise CurationTransformError(
            "Merged output changes Parquet compression between row groups"
        )
    resolved = {
        path: _compression_name(next(iter(codecs)))
        for path, codecs in codecs_by_path.items()
    }
    unique = set(resolved.values())
    if len(unique) == 1:
        return next(iter(unique))
    return resolved


def _compression_name(name: str) -> str | None:
    if name == "uncompressed":
        return None
    return name


def _without_metadata(schema: pa.Schema) -> pa.Schema:
    return pa.schema(
        [pa.field(field.name, field.type, nullable=field.nullable) for field in schema]
    )


def _schemas_equal(expected: pa.Schema, actual: pa.Schema) -> bool:
    return _without_metadata(expected).equals(
        _without_metadata(actual), check_metadata=False
    )


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(str(_without_metadata(schema)).encode()).hexdigest()


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if isinstance(left, float) and isinstance(right, float):
        return left == right or (math.isnan(left) and math.isnan(right))
    return left == right


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
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise CurationTransformError("Dataset Parquet file is invalid")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            yield pq.ParquetFile(stream)
    finally:
        os.close(descriptor)
