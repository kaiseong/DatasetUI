from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from datasetui.merge.schema import (
    inspect_compatible_data_schema,
    REPAIR_ENGINE,
    restore_merged_data_schema,
)
from datasetui.transform_errors import CurationTransformError



def _schema(
    *,
    state_type: pa.DataType | None = None,
    metadata: dict[bytes, bytes] | None = None,
    extra: bool = False,
) -> pa.Schema:
    fields = [
        pa.field(
            "observation.state",
            state_type or pa.list_(pa.float32(), 2),
        ),
        pa.field("action", pa.list_(pa.float32(), 2)),
        pa.field("episode_index", pa.int64()),
        pa.field("index", pa.int64()),
        pa.field("task_index", pa.int64()),
    ]
    if extra:
        fields.append(pa.field("unexpected", pa.int64()))
    return pa.schema(fields, metadata=metadata)


def _table(
    schema: pa.Schema,
    *,
    start: int,
    count: int,
    state_width: int = 2,
) -> pa.Table:
    values = list(range(start, start + count))
    arrays = [
        pa.array(
            [
                [float(value + item / 10) for item in range(state_width)]
                for value in values
            ],
            type=schema.field("observation.state").type,
        ),
        pa.array(
            [[float(-value), float(value)] for value in values],
            type=schema.field("action").type,
        ),
        pa.array([value // 3 for value in values], type=pa.int64()),
        pa.array([1000 + value for value in values], type=pa.int64()),
        pa.array([value % 2 for value in values], type=pa.int64()),
    ]
    if "unexpected" in schema.names:
        arrays.append(pa.array(values, type=pa.int64()))
    return pa.Table.from_arrays(arrays, schema=schema)


def _write_file(
    root: Path,
    schema: pa.Schema,
    row_groups: list[pa.Table],
    *,
    file_index: int = 0,
    compression: object = "zstd",
) -> Path:
    path = root / f"data/chunk-000/file-{file_index:03d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(path, schema, compression=compression) as writer:
        for table in row_groups:
            writer.write_table(table, row_group_size=len(table))
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_values(path: Path) -> dict[str, list]:
    table = pq.read_table(path)
    return {name: table.column(name).to_pylist() for name in table.column_names}


@pytest.mark.parametrize(
    "compression",
    [
        "zstd",
        None,
        {
            "observation.state.list.element": "zstd",
            "action.list.element": "snappy",
            "episode_index": "gzip",
            "index": "zstd",
            "task_index": "snappy",
        },
    ],
)
def test_restores_only_known_list_to_fixed_size_loss_and_preserves_values(
    tmp_path: Path, compression: object
) -> None:
    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    output = tmp_path / "output"
    source_schema_a = _schema(metadata={b"origin": b"a"})
    source_schema_b = _schema(metadata={b"origin": b"b"})
    source_files = [
        _write_file(
            source_a, source_schema_a, [_table(source_schema_a, start=0, count=2)]
        ),
        _write_file(
            source_b, source_schema_b, [_table(source_schema_b, start=2, count=3)]
        ),
    ]
    output_schema = _schema(
        state_type=pa.list_(pa.float32()), metadata={b"writer": b"upstream"}
    )
    output_path = _write_file(
        output,
        output_schema,
        [
            _table(output_schema, start=100, count=2),
            _table(output_schema, start=200, count=3),
        ],
        compression=compression,
    )
    output_path.chmod(0o640)
    source_hashes = [_sha256(path) for path in source_files]
    output_values = _read_values(output_path)
    output_parquet = pq.ParquetFile(output_path)
    output_compression = [
        output_parquet.metadata.row_group(0).column(index).compression
        for index in range(output_parquet.metadata.row_group(0).num_columns)
    ]

    result = restore_merged_data_schema(
        source_roots=[source_a, source_b], output_root=output
    )

    assert [_sha256(path) for path in source_files] == source_hashes
    assert _read_values(output_path) == output_values
    parquet = pq.ParquetFile(output_path)
    assert parquet.schema_arrow.field("observation.state").type == pa.list_(
        pa.float32(), 2
    )
    assert parquet.schema_arrow.field("action").type == pa.list_(pa.float32(), 2)
    assert parquet.schema_arrow.metadata == {b"writer": b"upstream"}
    assert [
        parquet.metadata.row_group(0).column(index).compression
        for index in range(parquet.metadata.row_group(0).num_columns)
    ] == output_compression
    assert output_path.stat().st_mode & 0o777 == 0o640
    assert [
        parquet.metadata.row_group(index).num_rows
        for index in range(parquet.metadata.num_row_groups)
    ] == [2, 3]
    assert result == {
        "engine": REPAIR_ENGINE,
        "source_schema_sha256": result["source_schema_sha256"],
        "output_files_checked": 1,
        "output_rows_checked": 5,
        "output_row_groups_checked": 2,
        "repaired_files": ["data/chunk-000/file-000.parquet"],
        "repaired_columns": ["observation.state"],
        "repair_kind": "variable-list-to-fixed-size-list",
    }


def test_identical_output_schema_is_not_rewritten(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    schema = _schema()
    _write_file(source, schema, [_table(schema, start=0, count=2)])
    output_path = _write_file(output, schema, [_table(schema, start=10, count=2)])
    before = _sha256(output_path)

    result = restore_merged_data_schema(source_roots=[source], output_root=output)

    assert _sha256(output_path) == before
    assert result["repaired_files"] == []
    assert result["repaired_columns"] == []


def test_source_schema_metadata_is_ignored_but_physical_mismatch_fails(
    tmp_path: Path,
) -> None:
    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    compatible_a = _schema(metadata={b"origin": b"a"})
    compatible_b = _schema(metadata={b"origin": b"b"})
    _write_file(source_a, compatible_a, [_table(compatible_a, start=0, count=1)])
    _write_file(source_b, compatible_b, [_table(compatible_b, start=1, count=1)])

    inspected = inspect_compatible_data_schema([source_a, source_b])

    assert inspected.metadata is None
    incompatible = _schema(state_type=pa.list_(pa.float64(), 2))
    _write_file(source_b, incompatible, [_table(incompatible, start=2, count=1)])
    with pytest.raises(
        CurationTransformError, match="source data schemas are incompatible"
    ):
        inspect_compatible_data_schema([source_a, source_b])


@pytest.mark.parametrize(
    "output_schema",
    [
        _schema(state_type=pa.list_(pa.float64())),
        _schema(extra=True),
    ],
)
def test_unexpected_output_schema_change_fails_without_rewriting(
    tmp_path: Path, output_schema: pa.Schema
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    expected = _schema()
    _write_file(source, expected, [_table(expected, start=0, count=2)])
    output_path = _write_file(
        output, output_schema, [_table(output_schema, start=10, count=2)]
    )
    before = _sha256(output_path)

    with pytest.raises(CurationTransformError):
        restore_merged_data_schema(source_roots=[source], output_root=output)

    assert _sha256(output_path) == before


def test_wrong_variable_list_length_fails_without_partial_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    expected = _schema()
    actual = _schema(state_type=pa.list_(pa.float32()))
    _write_file(source, expected, [_table(expected, start=0, count=2)])
    output_path = _write_file(
        output,
        actual,
        [_table(actual, start=10, count=2, state_width=3)],
    )
    before = _sha256(output_path)

    with pytest.raises(CurationTransformError, match="list length differs"):
        restore_merged_data_schema(source_roots=[source], output_root=output)

    assert _sha256(output_path) == before
    assert list(output_path.parent.glob(f".{output_path.name}.schema-*.tmp")) == []
