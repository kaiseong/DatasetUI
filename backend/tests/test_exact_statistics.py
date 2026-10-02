from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import datasetui.statistics.exact as exact_statistics

from datasetui.statistics.exact import recompute_numeric_statistics
from datasetui.transform_errors import CurationTransformError


def _write_dataset(
    root: Path,
    *,
    features: dict,
    tables: list[pa.Table],
    total_frames: int | None = None,
) -> None:
    (root / "meta").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "total_frames": (
            sum(table.num_rows for table in tables)
            if total_frames is None
            else total_frames
        ),
        "features": features,
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    for index, table in enumerate(tables):
        path = root / f"data/chunk-000/file-{index:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _expected(values: np.ndarray) -> dict:
    promoted = np.asarray(values, dtype=np.float64)
    return {
        "min": np.min(promoted, axis=0),
        "max": np.max(promoted, axis=0),
        "mean": np.mean(promoted, axis=0),
        "std": np.std(promoted, axis=0),
        "q01": np.quantile(promoted, 0.01, axis=0),
        "q10": np.quantile(promoted, 0.10, axis=0),
        "q50": np.quantile(promoted, 0.50, axis=0),
        "q90": np.quantile(promoted, 0.90, axis=0),
        "q99": np.quantile(promoted, 0.99, axis=0),
        "count": np.asarray([len(promoted)]),
    }


def test_exact_global_statistics_use_every_row_across_unequal_files(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "dataset"
    signal = np.asarray([1, 2, 10, 20, 30, 40, 50], dtype=np.float64)
    action = np.asarray([[value, -value] for value in signal], dtype=np.float32)
    features = {
        "signal": {"dtype": "float64", "shape": [1]},
        "action": {"dtype": "float32", "shape": [2]},
        "observation.images.top": {
            "dtype": "video",
            "shape": [64, 64, 3],
        },
        "language_events": {"dtype": "language", "shape": [1]},
        "label": {"dtype": "string", "shape": [1]},
    }
    tables = []
    for start, end in ((0, 2), (2, 7)):
        tables.append(
            pa.table(
                {
                    "signal": pa.array(signal[start:end], type=pa.float64()),
                    "action": pa.array(
                        action[start:end].tolist(),
                        type=pa.list_(pa.float32(), 2),
                    ),
                    "label": pa.array(["x"] * (end - start), type=pa.string()),
                }
            )
        )
    _write_dataset(root, features=features, tables=tables)
    before = _tree_hashes(root)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(exact_statistics.tempfile, "tempdir", str(scratch))
    events: list[dict] = []

    actual = recompute_numeric_statistics(root, on_progress=events.append)

    assert set(actual) == {"signal", "action"}
    for name, values in (("signal", signal[:, None]), ("action", action)):
        expected = _expected(values)
        for statistic, expected_value in expected.items():
            np.testing.assert_allclose(actual[name][statistic], expected_value)
    assert before == _tree_hashes(root)
    assert list(scratch.iterdir()) == []
    assert events[0] == {
        "stage": "statistics",
        "completed": 0,
        "total": 14,
        "unit": "rows",
        "current_item": "signal",
        "_force": True,
    }
    assert events[-1]["completed"] == events[-1]["total"] == 14
    assert events[-1]["current_item"] == "action"


def test_large_offset_small_variance_std_is_stable_for_float64_and_float32(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    float64_values = np.asarray(
        [1e12, 1e12 + 0.25, 1e12 + 0.5, 1e12 + 0.75], dtype=np.float64
    )
    float32_values = np.asarray(
        [1e6, 1e6 + 0.125, 1e6 + 0.25, 1e6 + 0.375], dtype=np.float32
    )
    _write_dataset(
        root,
        features={
            "signal64": {"dtype": "float64", "shape": [1]},
            "signal32": {"dtype": "float32", "shape": [1]},
        },
        tables=[
            pa.table(
                {
                    "signal64": pa.array(float64_values, type=pa.float64()),
                    "signal32": pa.array(float32_values, type=pa.float32()),
                }
            )
        ],
    )

    actual = recompute_numeric_statistics(root)

    assert actual["signal64"]["std"][0] == pytest.approx(
        float(np.std(float64_values.astype(np.longdouble))), rel=1e-12
    )
    assert actual["signal32"]["std"][0] == pytest.approx(
        float(np.std(float32_values.astype(np.longdouble))), rel=1e-12
    )
    assert actual["signal64"]["std"][0] > 0
    assert actual["signal32"]["std"][0] > 0


def test_episode_selection_streams_only_matching_rows(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    values = np.asarray([1, 2, 10, 20, 30], dtype=np.float32)
    episodes = np.asarray([0, 0, 1, 1, 1], dtype=np.int64)
    _write_dataset(
        root,
        features={
            "signal": {"dtype": "float32", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
        },
        tables=[
            pa.table(
                {
                    "signal": pa.array(values[:2], type=pa.float32()),
                    "episode_index": pa.array(episodes[:2], type=pa.int64()),
                }
            ),
            pa.table(
                {
                    "signal": pa.array(values[2:], type=pa.float32()),
                    "episode_index": pa.array(episodes[2:], type=pa.int64()),
                }
            ),
        ],
    )
    events: list[dict] = []

    actual = recompute_numeric_statistics(
        root, on_progress=events.append, episode_indices=[1]
    )

    for statistic, expected in _expected(values[2:, None]).items():
        np.testing.assert_allclose(actual["signal"][statistic], expected)
    assert actual["episode_index"]["min"] == [1.0]
    assert actual["episode_index"]["max"] == [1.0]
    assert actual["episode_index"]["count"] == [3]
    assert events[-1]["completed"] == events[-1]["total"] == 6
    with pytest.raises(CurationTransformError, match="Unknown episode"):
        recompute_numeric_statistics(root, episode_indices=[2])


def test_multidimensional_shape_is_preserved_in_every_statistic(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    values = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ],
        dtype=np.float32,
    )
    nested_type = pa.list_(pa.list_(pa.float32(), 2), 2)
    _write_dataset(
        root,
        features={"matrix": {"dtype": "float32", "shape": [2, 2]}},
        tables=[pa.table({"matrix": pa.array(values.tolist(), type=nested_type)})],
    )

    actual = recompute_numeric_statistics(root)["matrix"]

    for statistic, expected in _expected(values).items():
        np.testing.assert_allclose(actual[statistic], expected)
    assert np.asarray(actual["mean"]).shape == (2, 2)


def test_scans_one_declared_column_in_small_batches_and_uses_disk_memmap(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "dataset"
    values = np.arange(18, dtype=np.float32).reshape(9, 2)
    _write_dataset(
        root,
        features={"action": {"dtype": "float32", "shape": [2]}},
        tables=[
            pa.table(
                {"action": pa.array(values.tolist(), type=pa.list_(pa.float32(), 2))}
            )
        ],
    )
    monkeypatch.setattr(exact_statistics, "BATCH_ROWS", 2)
    original_iter_batches = pq.ParquetFile.iter_batches
    calls: list[tuple[int | None, list[str] | None]] = []

    def tracked_batches(self, *args, **kwargs):
        calls.append((kwargs.get("batch_size"), kwargs.get("columns")))
        return original_iter_batches(self, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", tracked_batches)
    original_memmap = np.memmap
    memmap_shapes: list[tuple[int, ...]] = []

    def tracked_memmap(*args, **kwargs):
        memmap_shapes.append(tuple(kwargs["shape"]))
        return original_memmap(*args, **kwargs)

    monkeypatch.setattr(exact_statistics.np, "memmap", tracked_memmap)
    monkeypatch.setattr(
        exact_statistics.pq,
        "read_table",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("read_table must not load the dataset")
        ),
    )

    actual = recompute_numeric_statistics(root)

    assert actual["action"]["count"] == [9]
    assert calls == [(2, ["action"])]
    assert memmap_shapes == [(2, 9)]


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_non_finite_values_fail_explicitly(tmp_path: Path, invalid: float) -> None:
    root = tmp_path / "dataset"
    _write_dataset(
        root,
        features={"signal": {"dtype": "float64", "shape": [1]}},
        tables=[pa.table({"signal": pa.array([1.0, invalid], type=pa.float64())})],
    )

    with pytest.raises(CurationTransformError, match="NaN or infinity: signal"):
        recompute_numeric_statistics(root)


def test_missing_numeric_column_fails_explicitly(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_dataset(
        root,
        features={"signal": {"dtype": "float32", "shape": [1]}},
        tables=[pa.table({"other": pa.array([1.0], type=pa.float32())})],
    )

    with pytest.raises(CurationTransformError, match="column is missing: signal"):
        recompute_numeric_statistics(root)


def test_declared_dtype_is_not_silently_cast_to_match_data(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_dataset(
        root,
        features={"signal": {"dtype": "float32", "shape": [1]}},
        tables=[pa.table({"signal": pa.array([1.0], type=pa.float64())})],
    )

    with pytest.raises(CurationTransformError, match="dtype differs from metadata"):
        recompute_numeric_statistics(root)


def test_declared_shape_mismatch_fails_explicitly(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_dataset(
        root,
        features={"action": {"dtype": "float32", "shape": [2]}},
        tables=[
            pa.table(
                {"action": pa.array([[1.0, 2.0, 3.0]], type=pa.list_(pa.float32(), 3))}
            )
        ],
    )

    with pytest.raises(CurationTransformError, match="shape differs from metadata"):
        recompute_numeric_statistics(root)


def test_total_frame_count_mismatch_fails_explicitly(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_dataset(
        root,
        features={"signal": {"dtype": "float32", "shape": [1]}},
        tables=[pa.table({"signal": pa.array([1.0, 2.0], type=pa.float32())})],
        total_frames=3,
    )

    with pytest.raises(CurationTransformError, match="row count differs.*2 != 3"):
        recompute_numeric_statistics(root)


@pytest.mark.parametrize("batch_rows", [3, 4096])
def test_single_pass_episode_statistics_equal_per_episode_runs(
    tmp_path: Path, monkeypatch, batch_rows: int
) -> None:
    # Episodes straddle files and record batches; values are irregular so any
    # change in batch slicing or accumulation order shows up in the bits.
    monkeypatch.setattr(exact_statistics, "BATCH_ROWS", batch_rows)
    root = tmp_path / "dataset"
    rng = np.random.default_rng(7)
    lengths = [5, 1, 9, 4, 7]
    episodes = np.repeat(np.arange(len(lengths)), lengths)
    state = rng.normal(size=(len(episodes), 3)).astype(np.float32) * 1e3
    signal = rng.normal(size=len(episodes)) / 7
    features = {
        "observation.state": {"dtype": "float32", "shape": [3]},
        "signal": {"dtype": "float64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "observation.images.top": {"dtype": "video", "shape": [8, 8, 3]},
    }
    tables = [
        pa.table(
            {
                "observation.state": pa.array(
                    state[start:end].tolist(), type=pa.list_(pa.float32(), 3)
                ),
                "signal": pa.array(signal[start:end], type=pa.float64()),
                "episode_index": pa.array(episodes[start:end], type=pa.int64()),
            }
        )
        for start, end in ((0, 8), (8, 11), (11, len(episodes)))
    ]
    _write_dataset(root, features=features, tables=tables)
    events: list[dict] = []

    actual = exact_statistics.recompute_numeric_statistics_by_episode(
        root, list(range(len(lengths))), on_progress=events.append
    )

    for episode in range(len(lengths)):
        expected = recompute_numeric_statistics(root, episode_indices=[episode])
        assert json.dumps(actual[episode]) == json.dumps(expected)
    assert list(actual) == list(range(len(lengths)))
    assert events[-1]["completed"] == events[-1]["total"] == len(episodes) * 3
    subset = exact_statistics.recompute_numeric_statistics_by_episode(root, [3, 1])
    assert json.dumps(subset[3]) == json.dumps(actual[3])
    with pytest.raises(CurationTransformError, match="Unknown episode"):
        exact_statistics.recompute_numeric_statistics_by_episode(root, [0, 9])


def test_single_pass_episode_statistics_check_total_frames(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_dataset(
        root,
        features={"episode_index": {"dtype": "int64", "shape": [1]}},
        tables=[pa.table({"episode_index": pa.array([0, 0, 1], type=pa.int64())})],
        total_frames=4,
    )
    with pytest.raises(CurationTransformError, match="differs from total_frames"):
        exact_statistics.recompute_numeric_statistics_by_episode(root, [0, 1])
