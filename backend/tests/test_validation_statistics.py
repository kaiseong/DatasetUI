from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from datasetui.validation_statistics import NumericStatisticsValidator


def test_stats_writer_preserves_declared_tensor_shape(tmp_path: Path) -> None:
    from datasetui.transforms import _write_stats

    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(
        json.dumps(
            {
                "total_frames": 2,
                "total_episodes": 1, "codebase_version": "v2.1", "fps": 10,
                "features": {"action": {"dtype": "float32", "shape": [2, 2]}},
            }
        )
    )
    frames = pd.DataFrame({"action": [np.zeros((2, 2)), np.ones((2, 2))]})
    import pyarrow as pa
    import pyarrow.parquet as pq

    (tmp_path / "data").mkdir()
    (meta / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 2}) + "\n")
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 0],
                "action": pa.array(
                    [value.tolist() for value in frames.action],
                    type=pa.list_(pa.list_(pa.float32(), 2), 2),
                )
            }
        ),
        tmp_path / "data/file.parquet",
    )
    _write_stats(meta / "stats.json", [frames])
    actual = json.loads((meta / "stats.json").read_text())["action"]
    assert actual["mean"] == [[0.5, 0.5], [0.5, 0.5]]
    assert actual["count"] == [2]


def _info(*, visual: bool = False) -> dict:
    features = {
        "action": {"dtype": "float32", "shape": [2]},
        "timestamp": {"dtype": "float32", "shape": [1]},
    }
    if visual:
        features["observation.images.top"] = {
            "dtype": "video",
            "shape": [24, 32, 3],
        }
    return {"features": features, "total_frames": 1}


def _stats(values: np.ndarray) -> dict:
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "count": [len(values)],
    }


def _write_stats(root: Path, value: dict) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "meta/stats.json").write_text(json.dumps(value), encoding="utf-8")


def _bind_official_policy(root: Path, info: dict, *, operation="split_dataset") -> None:
    from datasetui.official_operations import _bind_official_statistics

    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    episodes = root / "meta/episodes/chunk-000"
    episodes.mkdir(parents=True)
    pd.DataFrame({"episode_index": [0], "length": [1]}).to_parquet(
        episodes / "file-000.parquet", index=False
    )
    data = root / "data/chunk-000"
    data.mkdir(parents=True)
    pd.DataFrame({"index": [0]}).to_parquet(data / "file-000.parquet", index=False)
    _bind_official_statistics(root, operation=operation)


def test_official_policy_only_downgrades_known_bookkeeping_mismatches(
    tmp_path: Path,
) -> None:
    info = {
        "features": {
            "action": {"dtype": "float32", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
        }
    }
    stored = {
        "action": _stats(np.asarray([[999.0]])),
        "index": _stats(np.asarray([[999.0]])),
    }
    _write_stats(tmp_path, stored)
    _bind_official_policy(tmp_path, info)
    issues = []
    validator = NumericStatisticsValidator(info, lambda *item: issues.append(item))
    validator.add_episode(
        pd.DataFrame({"action": [1.0], "index": [0]}), episode_index=0
    )

    validator.validate(tmp_path, complete_dataset=True)

    assert any(
        item[0] == "FAIL" and item[1] == "stats_value_mismatch" for item in issues
    )
    assert any(
        item[0] == "WARN"
        and item[1] == "official_bookkeeping_stats_difference"
        for item in issues
    )


def test_tampered_official_marker_fails_and_does_not_relax_validation(
    tmp_path: Path,
) -> None:
    info = {"features": {"index": {"dtype": "int64", "shape": [1]}}}
    _write_stats(tmp_path, {"index": _stats(np.asarray([[999.0]]))})
    _bind_official_policy(tmp_path, info)
    marker_path = tmp_path / "meta/datasetui_provenance.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["stats_sha256"] = "0" * 64
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    issues = []
    validator = NumericStatisticsValidator(info, lambda *item: issues.append(item))
    validator.add_episode(pd.DataFrame({"index": [0]}), episode_index=0)

    validator.validate(tmp_path, complete_dataset=True)

    assert any(
        item[0] == "FAIL" and item[1] == "official_statistics_provenance_invalid"
        for item in issues
    )
    assert any(
        item[0] == "FAIL" and item[1] == "stats_value_mismatch" for item in issues
    )


def test_official_bookkeeping_count_mismatch_remains_a_failure(tmp_path: Path) -> None:
    info = {"features": {"index": {"dtype": "int64", "shape": [1]}}}
    stats = _stats(np.asarray([[0.0]]))
    stats["count"] = [999]
    _write_stats(tmp_path, {"index": stats})
    _bind_official_policy(tmp_path, info)
    issues = []
    validator = NumericStatisticsValidator(info, lambda *item: issues.append(item))
    validator.add_episode(pd.DataFrame({"index": [0]}), episode_index=0)

    validator.validate(tmp_path, complete_dataset=True)

    assert any(
        item[0] == "FAIL" and item[1] == "stats_value_mismatch" for item in issues
    )


def test_official_policy_rejects_changed_video_content(tmp_path: Path) -> None:
    info = {"features": {"index": {"dtype": "int64", "shape": [1]}}}
    _write_stats(tmp_path, {"index": _stats(np.asarray([[0.0]]))})
    video = tmp_path / "videos/camera/chunk-000/file-000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"original-video")
    _bind_official_policy(tmp_path, info)
    video.write_bytes(b"changed-video")
    issues = []
    validator = NumericStatisticsValidator(info, lambda *item: issues.append(item))
    validator.add_episode(pd.DataFrame({"index": [0]}), episode_index=0)

    validator.validate(tmp_path, complete_dataset=True)

    assert any(
        item[0] == "FAIL" and item[1] == "official_statistics_provenance_invalid"
        for item in issues
    )


def test_official_policy_rejects_changed_parquet_content(tmp_path: Path) -> None:
    info = {"features": {"index": {"dtype": "int64", "shape": [1]}}}
    _write_stats(tmp_path, {"index": _stats(np.asarray([[0.0]]))})
    _bind_official_policy(tmp_path, info)
    data = tmp_path / "data/chunk-000/file-000.parquet"
    pd.DataFrame({"index": [1]}).to_parquet(data, index=False)
    issues = []
    validator = NumericStatisticsValidator(info, lambda *item: issues.append(item))
    validator.add_episode(pd.DataFrame({"index": [0]}), episode_index=0)

    validator.validate(tmp_path, complete_dataset=True)

    assert any(
        item[0] == "FAIL" and item[1] == "official_statistics_provenance_invalid"
        for item in issues
    )


def test_official_policy_rejects_symlinked_episode_metadata(tmp_path: Path) -> None:
    info = {"features": {"index": {"dtype": "int64", "shape": [1]}}}
    _write_stats(tmp_path, {"index": _stats(np.asarray([[0.0]]))})
    _bind_official_policy(tmp_path, info)
    episode = tmp_path / "meta/episodes/chunk-000/file-000.parquet"
    episode.unlink()
    episode.symlink_to(tmp_path / "meta/stats.json")
    issues = []
    validator = NumericStatisticsValidator(info, lambda *item: issues.append(item))
    validator.add_episode(pd.DataFrame({"index": [0]}), episode_index=0)

    validator.validate(tmp_path, complete_dataset=True)

    assert any(
        item[0] == "FAIL" and item[1] == "official_statistics_provenance_invalid"
        for item in issues
    )


def test_bound_official_visual_difference_is_verified_with_warning(
    tmp_path: Path,
) -> None:
    feature = "observation.images.top"
    info = {
        "features": {
            feature: {"dtype": "video", "shape": [24, 32, 3]},
        }
    }
    stored = {
        feature: {
            "min": [[[0.0]], [[0.0]], [[0.0]]],
            "max": [[[1.0]], [[1.0]], [[1.0]]],
            "mean": [[[0.5]], [[0.5]], [[0.5]]],
            "std": [[[0.1]], [[0.1]], [[0.1]]],
            "count": [1],
        }
    }
    actual = {
        feature: {
            "min": np.zeros((3, 1, 1)),
            "max": np.zeros((3, 1, 1)),
            "mean": np.zeros((3, 1, 1)),
            "std": np.zeros((3, 1, 1)),
            "count": np.asarray([2]),
        }
    }
    _write_stats(tmp_path, stored)
    _bind_official_policy(tmp_path, info)
    issues = []
    validator = NumericStatisticsValidator(info, lambda *item: issues.append(item))

    summary = validator.validate(
        tmp_path, complete_dataset=True, visual_statistics=actual
    )

    assert not any(item[0] == "FAIL" for item in issues)
    assert any(item[1] == "official_visual_stats_difference" for item in issues)
    assert summary.unverified_visual_features == ()
    assert summary.validated_features == (feature,)


def test_recomputes_numeric_stats_episode_wise_with_bounded_state(
    tmp_path: Path,
) -> None:
    issues = []
    validator = NumericStatisticsValidator(_info(), lambda *item: issues.append(item))
    first = pd.DataFrame({"action": [[1.0, 2.0], [3.0, 4.0]], "timestamp": [0.0, 0.1]})
    second = pd.DataFrame({"action": [[5.0, 6.0], [7.0, 8.0]], "timestamp": [0.0, 0.1]})
    validator.add_episode(first, episode_index=0)
    validator.add_episode(second, episode_index=1)
    action = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
    timestamp = np.asarray([[0.0], [0.1], [0.0], [0.1]])
    _write_stats(tmp_path, {"action": _stats(action), "timestamp": _stats(timestamp)})

    summary = validator.validate(tmp_path, complete_dataset=True)

    assert issues == []
    assert summary.validated_features == ("action", "timestamp")
    running = validator._running["action"]
    assert not hasattr(running, "samples")
    assert running.count == 4


def test_std_is_stable_for_large_offsets_with_small_variance(tmp_path: Path) -> None:
    issues = []
    info = {"features": {"signal": {"dtype": "float64", "shape": [1]}}}
    validator = NumericStatisticsValidator(info, lambda *item: issues.append(item))
    values = np.asarray([[1e9], [1e9 + 1], [1e9 + 2]], dtype=np.float64)
    validator.add_episode(pd.DataFrame({"signal": values[:, 0]}), episode_index=0)
    _write_stats(tmp_path, {"signal": _stats(values)})

    summary = validator.validate(tmp_path, complete_dataset=True)

    assert issues == []
    assert summary.validated_features == ("signal",)


def test_numeric_stat_shape_must_match_declared_multidimensional_shape(
    tmp_path: Path,
) -> None:
    info = {"features": {"matrix": {"dtype": "float32", "shape": [2, 2]}}}
    values = np.asarray(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        dtype=np.float32,
    )
    validator = NumericStatisticsValidator(info, lambda *_: None)
    validator.add_episode(pd.DataFrame({"matrix": list(values)}), episode_index=0)
    flattened = {
        name: np.asarray(value).reshape(-1).tolist()
        for name, value in _stats(values).items()
    }
    _write_stats(tmp_path, {"matrix": flattened})
    issues = []
    validator = NumericStatisticsValidator(info, lambda *item: issues.append(item))
    validator.add_episode(pd.DataFrame({"matrix": list(values)}), episode_index=0)

    validator.validate(tmp_path, complete_dataset=True)

    assert any(item[1] == "stats_shape_mismatch" for item in issues)


def test_stored_stat_strings_and_boole_are_not_coerced_to_numbers(
    tmp_path: Path,
) -> None:
    info = {"features": {"signal": {"dtype": "float32", "shape": [1]}}}
    validator = NumericStatisticsValidator(info, lambda *_: None)
    validator.add_episode(pd.DataFrame({"signal": [1.0]}), episode_index=0)
    _write_stats(
        tmp_path,
        {
            "signal": {
                "min": ["1"],
                "max": [1.0],
                "mean": [1.0],
                "std": [False],
                "count": [1],
            }
        },
    )
    issues = []
    validator = NumericStatisticsValidator(info, lambda *item: issues.append(item))
    validator.add_episode(pd.DataFrame({"signal": [1.0]}), episode_index=0)

    validator.validate(tmp_path, complete_dataset=True)

    assert [item[1] for item in issues].count("stats_value_invalid") == 2


def test_rejects_wrong_values_missing_fields_and_non_finite_values(
    tmp_path: Path,
) -> None:
    issues = []
    validator = NumericStatisticsValidator(_info(), lambda *item: issues.append(item))
    validator.add_episode(
        pd.DataFrame({"action": [[1.0, 2.0]], "timestamp": [0.0]}),
        episode_index=0,
    )
    _write_stats(
        tmp_path,
        {
            "action": {
                "min": [1.0, 2.0],
                "max": [1.0, 2.0],
                "mean": [999.0, 2.0],
                "std": [0.0, 0.0],
                "count": [1],
            },
            "timestamp": {"min": [float("nan")], "count": [1]},
        },
    )

    validator.validate(tmp_path, complete_dataset=True)

    codes = [item[1] for item in issues]
    assert "stats_value_invalid" in codes
    assert "stats_field_missing" in codes
    assert "stats_value_mismatch" in codes


def test_rejects_numeric_strings_instead_of_silently_coercing(tmp_path: Path) -> None:
    issues = []
    validator = NumericStatisticsValidator(_info(), lambda *item: issues.append(item))

    validator.add_episode(
        pd.DataFrame({"action": [["1", "2"]], "timestamp": [0.0]}),
        episode_index=0,
    )

    assert any(item[1] == "stats_source_invalid" for item in issues)


def test_visual_stats_are_required_but_reported_as_not_recomputed(
    tmp_path: Path,
) -> None:
    issues = []
    validator = NumericStatisticsValidator(
        _info(visual=True), lambda *item: issues.append(item)
    )
    frame = pd.DataFrame({"action": [[1.0, 2.0]], "timestamp": [0.0]})
    validator.add_episode(frame, episode_index=0)
    _write_stats(
        tmp_path,
        {
            "action": _stats(np.asarray([[1.0, 2.0]])),
            "timestamp": _stats(np.asarray([[0.0]])),
        },
    )

    summary = validator.validate(tmp_path, complete_dataset=True)

    assert summary.unverified_visual_features == ("observation.images.top",)
    assert any(item[1] == "stats_feature_missing" for item in issues)
    assert any(item[1] == "visual_stats_not_recomputed" for item in issues)


def test_incomplete_episode_scan_cannot_validate_statistics(tmp_path: Path) -> None:
    issues = []
    validator = NumericStatisticsValidator(_info(), lambda *item: issues.append(item))

    summary = validator.validate(tmp_path, complete_dataset=False)

    assert summary.validated_features == ()
    assert any(item[1] == "stats_recompute_incomplete" for item in issues)


def test_visual_stat_shape_and_count_are_validated(tmp_path: Path) -> None:
    issues = []
    validator = NumericStatisticsValidator(
        _info(visual=True), lambda *item: issues.append(item)
    )
    frame = pd.DataFrame({"action": [[1.0, 2.0]], "timestamp": [0.0]})
    validator.add_episode(frame, episode_index=0)
    _write_stats(
        tmp_path,
        {
            "action": _stats(np.asarray([[1.0, 2.0]])),
            "timestamp": _stats(np.asarray([[0.0]])),
            "observation.images.top": {
                "min": [0.0],
                "max": [1.0],
                "mean": [0.5],
                "std": [0.1],
                "count": [999],
            },
        },
    )

    validator.validate(
        tmp_path,
        complete_dataset=True,
        visual_statistics={
            "observation.images.top": {
                "min": np.zeros(3),
                "max": np.ones(3),
                "mean": np.full(3, 0.5),
                "std": np.full(3, 0.1),
                "count": np.asarray([100]),
            }
        },
    )

    codes = [item[1] for item in issues]
    assert "stats_shape_mismatch" in codes
    assert "stats_count_mismatch" in codes
