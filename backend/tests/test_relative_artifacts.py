import copy
import json

import numpy as np
import pytest

from test_transforms import _write_v21

from datasetui.curation.writer import write_dataset
from datasetui.dataset_io.source import DatasetSource
from datasetui.models import RelativeActionConfig
from datasetui.relative.artifacts import read_relative_profile
from datasetui.transform_errors import CurationTransformError
from datasetui.validation.run import validate_dataset_root





@pytest.mark.parametrize("horizon", [0, -1, 1025, True, 1.5, "50"])
def test_relative_chunk_size_is_strict(horizon):
    with pytest.raises(ValueError):
        RelativeActionConfig(enabled=True, dimensions=["joint_0"], chunk_size=horizon)


def test_legacy_disabled_config_has_default_chunk():
    assert RelativeActionConfig().chunk_size == 50


def _fake_profile(info, episodes, config, *, on_progress=None):
    arrays = []
    horizon = config["chunk_size"]
    for data in episodes:
        actions = np.stack(data.action).astype(np.float32)
        states = np.stack(data["observation.state"]).astype(np.float32)
        arrays.extend(
            actions[t : t + horizon] - states[t] for t in range(len(data) - horizon + 1)
        )
    values = np.concatenate(arrays).astype(np.float64)
    stats = {
        key: getattr(np, key)(values, axis=0).tolist()
        for key in ("min", "max", "mean", "std")
    }
    stats["count"] = [len(values)]
    for name, q in [
        ("q01", 0.01),
        ("q10", 0.1),
        ("q50", 0.5),
        ("q90", 0.9),
        ("q99", 0.99),
    ]:
        stats[name] = np.quantile(values, q, axis=0).tolist()
    return {
        "enabled": True,
        "dimensions": ["joint_0"],
        "absolute_dimensions": [],
        "mask": [True],
        "action_names": ["joint_0"],
        "chunk_size": horizon,
        "action_horizon": horizon,
        "stored_action": "absolute",
        "training_normalization": True,
        "statistics_scope": "chunk-relative",
        "engine": "test-official-adapter",
        "official_function": "to_relative_actions",
        "upstream_commit": "test-pin",
        "processor_config": {
            "enabled": True,
            "action_names": ["datasetui_dim_0000"],
            "exclude_joints": [],
        },
        "statistics": stats,
    }


@pytest.fixture
def relative_output(tmp_path, monkeypatch):
    import datasetui.relative.actions as relative_actions

    monkeypatch.setattr(
        relative_actions, "compute_relative_action_profile", _fake_profile
    )
    source = tmp_path / "source"
    _write_v21(source)
    original = {
        str(p.relative_to(source)): p.read_bytes()
        for p in source.rglob("*")
        if p.is_file()
    }
    info = json.loads((source / "meta/info.json").read_text())
    destination = tmp_path / "relative"
    built = write_dataset(
        source=DatasetSource(source, info),
        destination=destination,
        source_indices=[0, 1],
        trim_config={"enabled": False},
        annotations={},
        relative_action={"enabled": True, "dimensions": ["joint_0"], "chunk_size": 3},
    )
    assert original == {
        str(p.relative_to(source)): p.read_bytes()
        for p in source.rglob("*")
        if p.is_file()
    }
    return source, destination, built


def test_materialization_preserves_absolute_rows_and_writes_relative_training_stats(
    relative_output,
):
    source, destination, built = relative_output
    info = json.loads((source / "meta/info.json").read_text())
    original = DatasetSource(source, info)
    output = DatasetSource(
        destination, json.loads((destination / "meta/info.json").read_text())
    )
    for episode in range(2):
        for key in ("action", "observation.state"):
            np.testing.assert_array_equal(
                np.stack(original.episode(episode)[0][key]),
                np.stack(output.episode(episode)[0][key]),
            )
    absolute = json.loads((destination / "meta/stats.absolute.json").read_text())
    mixed = json.loads((destination / "meta/stats.json").read_text())
    assert absolute["action"] != mixed["action"]
    assert mixed["action"] == built["relative_action"]["statistics"]
    assert mixed["observation.state"] == absolute["observation.state"]
    assert read_relative_profile(destination)["chunk_size"] == 3
    report = validate_dataset_root(destination, mode="full")
    assert report["passed"], report["issues"]


@pytest.mark.parametrize(
    "tamper",
    [
        "mask",
        "quantile",
        "missing_quantile",
        "profile_stats",
        "horizon",
        "invalid_json",
        "absolute_backup",
        "processor_class",
    ],
)
def test_validation_rejects_bad_relative_artifacts(relative_output, tamper):
    _, root, _ = relative_output
    profile_path = root / "meta/relative_action.json"
    profile = json.loads(profile_path.read_text())
    if tamper == "absolute_backup":
        path = root / "meta/stats.absolute.json"
        stats = json.loads(path.read_text())
        stats["action"]["q99"] = [12345]
        path.write_text(json.dumps(stats))
    elif tamper == "processor_class":
        path = root / "meta/datasetui_relative_preprocessor.json"
        value = json.loads(path.read_text())
        value["steps"][0]["class"] = "unsafe.module.Class"
        path.write_text(json.dumps(value))
    elif tamper in ("quantile", "missing_quantile"):
        path = root / "meta/stats.json"
        stats = json.loads(path.read_text())
        if tamper == "quantile":
            stats["action"]["q99"] = [12345]
        else:
            del stats["action"]["q99"]
        path.write_text(json.dumps(stats))
    elif tamper == "invalid_json":
        profile_path.write_text("{")
    else:
        profile = copy.deepcopy(profile)
        if tamper == "mask":
            profile["mask"] = [False]
        elif tamper == "profile_stats":
            profile["statistics"]["mean"] = [12345]
        else:
            profile["action_horizon"] = 99
        profile_path.write_text(json.dumps(profile))
    report = validate_dataset_root(root, mode="full")
    assert not report["passed"]
    assert any(
        item["code"] in {"relative_profile_invalid", "relative_stats_mismatch"}
        for item in report["issues"]
    )


def test_symlinked_relative_marker_fails_closed(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/relative_action.json").symlink_to(tmp_path / "missing")
    with pytest.raises((OSError, CurationTransformError)):
        read_relative_profile(tmp_path)


def test_relative_only_preserves_all_data_files_byte_for_byte(relative_output):
    source, output, _ = relative_output
    assert {
        str(p.relative_to(source)): p.read_bytes()
        for p in (source / "data").rglob("*.parquet")
    } == {
        str(p.relative_to(output)): p.read_bytes()
        for p in (output / "data").rglob("*.parquet")
    }
    assert not (output / "meta/datasetui_relative.py").exists()
    assert (output / "meta/datasetui_relative_preprocessor.json").is_file()


def test_relative_source_cannot_silently_lose_profile(relative_output, tmp_path):
    _, root, _ = relative_output
    from datasetui.relative.artifacts import reject_relative_profile

    for operation in ("Merge", "v2.1 conversion"):
        with pytest.raises(CurationTransformError, match="discard"):
            reject_relative_profile(root, operation=operation)
    info = json.loads((root / "meta/info.json").read_text())
    with pytest.raises(CurationTransformError, match="discard"):
        write_dataset(
            source=DatasetSource(root, info),
            destination=tmp_path / "invalid",
            source_indices=[0],
            trim_config={},
            annotations={},
            relative_action={},
        )
