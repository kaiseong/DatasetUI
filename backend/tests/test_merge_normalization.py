from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from datasetui.merge_normalization import (
    NORMALIZATION_ENGINE,
    normalize_private_merge_sources,
    plan_merge_normalization,
)
from datasetui.transform_errors import CurationTransformError

def test_identical_float64_vectors_are_preserved_without_narrowing(tmp_path):
    root = tmp_path / "source"
    path = _dataset(root, fixed=True, timestamp_type=pa.float32())
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    table = pq.read_table(path)
    metadata = dict(table.schema.metadata)
    hf = json.loads(metadata[b"huggingface"])
    for name in ("action", "observation.state"):
        dtype = pa.list_(pa.float64(), 2)
        index = table.schema.get_field_index(name)
        table = table.set_column(index, name, table[name].cast(dtype))
        info["features"][name]["dtype"] = "float64"
        hf["info"]["features"][name]["feature"]["dtype"] = "float64"
    metadata[b"huggingface"] = json.dumps(hf).encode()
    pq.write_table(table.replace_schema_metadata(metadata), path)
    info_path.write_text(json.dumps(info))
    before = _sha(path)
    plan = plan_merge_normalization([root])
    assert plan["vector_widths"] == {}
    copied = tmp_path / "copy"
    shutil.copytree(root, copied)
    normalize_private_merge_sources([copied], plan)
    actual = pq.read_table(copied / "data/chunk-000/file-000.parquet")
    for name in ("action", "observation.state"):
        assert actual.schema.field(name).type == pa.list_(pa.float64(), 2)
        assert actual[name].equals(table[name])
    assert _sha(path) == before



def _hf_metadata(*, fixed: bool, timestamp: str) -> dict[bytes, bytes]:
    vector = {
        "feature": {"dtype": "float32", "_type": "Value"},
        "_type": "List",
    }
    if fixed:
        vector["length"] = 2
    payload = {
        "info": {
            "features": {
                "action": vector,
                "observation.state": dict(vector),
                "timestamp": {"dtype": timestamp, "_type": "Value"},
                "index": {"dtype": "int64", "_type": "Value"},
            }
        }
    }
    return {
        b"huggingface": json.dumps(payload).encode(),
        b"preserved": b"yes",
    }


def _dataset(
    root: Path,
    *,
    fixed: bool,
    timestamp_type: pa.DataType,
    bad_action: list[list[float] | None] | None = None,
    timestamp_values: list[float] | None = None,
) -> Path:
    (root / "meta").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "rby1",
        "features": {
            "action": {"dtype": "float32", "shape": [2], "names": None},
            "observation.state": {
                "dtype": "float32",
                "shape": [2],
                "names": None,
            },
            "timestamp": {
                "dtype": "float64" if pa.types.is_float64(timestamp_type) else "float32",
                "shape": [1],
                "names": None,
            },
            "index": {"dtype": "int64", "shape": [1], "names": None},
        },
        "custom": {"keep": True},
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    vector_type = pa.list_(pa.float32(), 2) if fixed else pa.list_(pa.float32())
    schema = pa.schema(
        [
            pa.field("action", vector_type),
            pa.field("observation.state", vector_type),
            pa.field("timestamp", timestamp_type),
            pa.field("index", pa.int64()),
        ],
        metadata=_hf_metadata(
            fixed=fixed,
            timestamp="float64" if pa.types.is_float64(timestamp_type) else "float32",
        ),
    )
    path = root / "data/chunk-000/file-000.parquet"
    path.parent.mkdir(parents=True)
    actions = bad_action or [[1.0, 2.0], [3.0, 4.0]]
    timestamps = timestamp_values or [0.0, 1.0 / 30.0]
    table = pa.Table.from_arrays(
        [
            pa.array(actions, type=vector_type),
            pa.array([[5.0, 6.0], [7.0, 8.0]], type=vector_type),
            pa.array(timestamps, type=timestamp_type),
            pa.array([0, 1], type=pa.int64()),
        ],
        schema=schema,
    )
    pq.write_table(table, path, compression="zstd", row_group_size=1)
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mixed_fixed_and_variable_float_inputs_normalize_private_copies_losslessly(
    tmp_path: Path,
) -> None:
    originals = [tmp_path / "original-a", tmp_path / "original-b"]
    source_files = [
        _dataset(originals[0], fixed=True, timestamp_type=pa.float64()),
        _dataset(originals[1], fixed=False, timestamp_type=pa.float32()),
    ]
    original_hashes = {
        path: _sha(path)
        for root, path in zip(originals, source_files, strict=True)
        for path in (path, root / "meta/info.json")
    }

    plan = plan_merge_normalization(originals)

    assert plan["engine"] == NORMALIZATION_ENGINE
    json.dumps(plan)
    assert plan["rows_checked"] == 4
    assert {_sha(path) for path in source_files} == {
        original_hashes[path] for path in source_files
    }
    copies = [tmp_path / "copy-a", tmp_path / "copy-b"]
    for source, destination in zip(originals, copies, strict=True):
        shutil.copytree(source, destination)
    result = normalize_private_merge_sources(copies, plan)

    assert result["datasets_normalized"] == 2
    assert len(result["files_rewritten"]) == 2
    for path, expected in original_hashes.items():
        assert _sha(path) == expected
    for index, copy in enumerate(copies):
        info = json.loads((copy / "meta/info.json").read_text())
        assert info["features"]["timestamp"]["dtype"] == "float64"
        assert info["custom"] == {"keep": True}
        parquet = pq.ParquetFile(copy / "data/chunk-000/file-000.parquet")
        assert parquet.metadata.row_group(0).column(0).compression == "ZSTD"
        schema = parquet.schema_arrow
        assert schema.field("action").type == pa.list_(pa.float32())
        assert schema.field("observation.state").type == pa.list_(pa.float32())
        assert schema.field("timestamp").type == pa.float64()
        assert schema.metadata[b"preserved"] == b"yes"
        hf = json.loads(schema.metadata[b"huggingface"])
        assert hf["info"]["features"]["timestamp"]["dtype"] == "float64"
        assert "length" not in hf["info"]["features"]["action"]
        data = parquet.read()
        assert data.column("action").to_pylist() == [[1.0, 2.0], [3.0, 4.0]]
        if index == 0:
            assert data.column("timestamp").to_pylist()[1] == 1.0 / 30.0


@pytest.mark.parametrize(
    ("bad_action", "timestamps", "message"),
    [
        ([[1.0, 2.0], [3.0]], None, "Vector length differs.*row=1"),
        ([[1.0, 2.0], [3.0, None]], None, "Null vector element.*row=1"),
        ([[1.0, 2.0], [3.0, float("nan")]], None, "Vector element must be finite.*row=1"),
        (None, [0.0, float("nan")], "Timestamp must be finite.*row=1"),
    ],
)
def test_preflight_rejects_bad_rows_with_location_without_modifying_source(
    tmp_path: Path,
    bad_action: list[list[float] | None] | None,
    timestamps: list[float] | None,
    message: str,
) -> None:
    root = tmp_path / "source"
    path = _dataset(
        root,
        fixed=False,
        timestamp_type=pa.float32(),
        bad_action=bad_action,
        timestamp_values=timestamps,
    )
    before = (_sha(path), _sha(root / "meta/info.json"))

    with pytest.raises(CurationTransformError, match=message):
        plan_merge_normalization([root])

    assert (_sha(path), _sha(root / "meta/info.json")) == before


def test_preflight_rejects_unsupported_timestamp_dtype(tmp_path: Path) -> None:
    root = tmp_path / "source"
    path = _dataset(root, fixed=False, timestamp_type=pa.float32())
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"]["timestamp"]["dtype"] = "int64"
    info_path.write_text(json.dumps(info))
    before = _sha(path)

    with pytest.raises(CurationTransformError, match="column=timestamp"):
        plan_merge_normalization([root])

    assert _sha(path) == before


def test_normalization_refuses_original_roots(tmp_path: Path) -> None:
    root = tmp_path / "source"
    path = _dataset(root, fixed=False, timestamp_type=pa.float32())
    plan = plan_merge_normalization([root])
    before = _sha(path)

    with pytest.raises(CurationTransformError, match="refuses to modify an original"):
        normalize_private_merge_sources([root], plan)

    assert _sha(path) == before
