import json

import pandas as pd
import pytest

from datasetui import transforms
from datasetui.deferred_statistics import read_deferred_statistics
from datasetui.models import TrimConfig
from datasetui.output_statistics import write_output_statistics
from datasetui.validation import validate_dataset_root
from test_transforms import _write_v21


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.delenv("DATASETUI_PROCESSOR_ENGINE", raising=False)
    root = tmp_path / "source"
    _write_v21(root)
    write_output_statistics(root)
    return transforms._DatasetSource(
        root, json.loads((root / "meta/info.json").read_text())
    )


def build(source, destination, *, recompute=False, trim=True):
    return transforms._write_dataset(
        source=source,
        destination=destination,
        source_indices=[0],
        trim_config={
            "enabled": trim,
            "recompute_statistics": recompute,
            "episode_overrides": {0: {"start_frame": 2, "end_frame": 8}},
        },
        annotations={},
    )


def test_skip_avoids_recomputation_preserves_source_and_updates_structure(
    source, tmp_path, monkeypatch
):
    before = {
        str(p.relative_to(source.root)): p.read_bytes()
        for p in source.root.rglob("*")
        if p.is_file()
    }
    monkeypatch.setattr(
        transforms, "_write_stats", lambda *a, **k: pytest.fail("statistics recomputed")
    )
    output = tmp_path / "trimmed"
    result = build(source, output)
    assert result["statistics"]["status"] == "deferred"
    assert read_deferred_statistics(output)
    assert (output / "meta/stats.json").read_bytes() == before["meta/stats.json"]
    info = json.loads((output / "meta/info.json").read_text())
    assert info["total_frames"] == 6 and info["total_episodes"] == 1
    frame = pd.read_parquet(next((output / "data").rglob("*.parquet")))
    assert frame.frame_index.tolist() == list(range(6))
    assert "norm_stats" in (output / "README.md").read_text()
    assert {
        str(p.relative_to(source.root)): p.read_bytes()
        for p in source.root.rglob("*")
        if p.is_file()
    } == before
    validation = validate_dataset_root(output, mode="export_gate")
    assert validation["passed"], validation["issues"]
    assert any(i["code"] == "stats_deferred" for i in validation["issues"])
    assert (
        next(c for c in validation["checks"] if c["id"] == "statistics")["status"]
        == "warning"
    )


def test_opt_in_and_legacy_option_recompute(source, tmp_path):
    assert TrimConfig(enabled=True).recompute_statistics is True
    output = tmp_path / "recomputed"
    build(source, output, recompute=True)
    assert not read_deferred_statistics(output)
    assert json.loads((output / "meta/stats.json").read_text())["action"]["count"] == [
        6
    ]
    result = validate_dataset_root(output, mode="export_gate")
    assert result["passed"], result["issues"]


@pytest.mark.parametrize(
    "corruption", ["marker", "info", "stats", "nonfinite", "index"]
)
def test_deferred_statistics_does_not_hide_corruption(source, tmp_path, corruption):
    output = tmp_path / "trimmed"
    build(source, output)
    if corruption == "marker":
        (output / "meta/datasetui_statistics.json").unlink()
    elif corruption in {"info", "stats"}:
        path = output / "meta" / f"{corruption}.json"
        value = json.loads(path.read_text())
        if corruption == "info":
            value["total_frames"] += 1
        else:
            value["action"]["mean"] = [123.0]
        path.write_text(json.dumps(value))
    else:
        path = next((output / "data").rglob("*.parquet"))
        frame = pd.read_parquet(path)
        if corruption == "nonfinite":
            frame.at[0, "action"] = [float("nan")]
        else:
            frame.loc[0, "frame_index"] = 99
        frame.to_parquet(path, index=False)
    result = validate_dataset_root(output, mode="export_gate")
    assert not result["passed"], result


def test_deferred_state_survives_selection_and_can_be_recomputed(
    source, tmp_path, monkeypatch
):
    output = tmp_path / "trimmed"
    build(source, output)
    inherited = transforms._DatasetSource(
        output, json.loads((output / "meta/info.json").read_text())
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            transforms,
            "_write_stats",
            lambda *a, **k: pytest.fail("selection recomputed statistics"),
        )
        selected = tmp_path / "selected"
        build(inherited, selected, trim=False)
    assert read_deferred_statistics(selected)
    write_output_statistics(selected)
    assert not read_deferred_statistics(selected)
    assert (
        "Distribution statistics deferred" not in (selected / "README.md").read_text()
    )
    assert validate_dataset_root(selected, mode="export_gate")["passed"]


def test_deferred_merge_does_not_aggregate_stale_statistics(
    source, tmp_path, monkeypatch
):
    from datasetui.merge import _MergedSource
    from datasetui.merge_writer import write_preserved_merge

    output = tmp_path / "trimmed"
    build(source, output)
    inherited = transforms._DatasetSource(
        output, json.loads((output / "meta/info.json").read_text())
    )
    merged = _MergedSource([inherited, source], "rby1")
    monkeypatch.setattr(
        "datasetui.merge_writer._write_stats",
        lambda *a, **k: pytest.fail("merge recomputed statistics"),
    )
    destination = tmp_path / "merged"
    result = write_preserved_merge(source=merged, destination=destination)
    assert result["statistics"]["status"] == "deferred"
    assert read_deferred_statistics(destination)
    validation = validate_dataset_root(destination, mode="export_gate")
    assert validation["passed"], validation["issues"]


def test_relative_statistics_are_still_required():
    from datasetui.deferred_statistics import should_defer

    assert not should_defer(
        {"enabled": True, "recompute_statistics": False}, {"enabled": True}
    )


def test_stationary_v3_skip_preserves_video_bytes_and_segment_ranges(
    tmp_path, monkeypatch
):
    from test_trim_source_codec import _source

    monkeypatch.delenv("DATASETUI_PROCESSOR_ENGINE", raising=False)
    root = tmp_path / "source"
    _source(root, "v3.0")
    write_output_statistics(root)
    source = transforms._DatasetSource(
        root, json.loads((root / "meta/info.json").read_text())
    )
    source_videos = sorted(p.read_bytes() for p in (root / "videos").rglob("*.mp4"))
    monkeypatch.setattr(
        transforms, "_write_stats", lambda *a, **k: pytest.fail("statistics recomputed")
    )
    monkeypatch.setattr(
        transforms,
        "_slice_video",
        lambda *a, **k: pytest.fail("stationary video reencoded"),
    )
    destination = tmp_path / "trimmed"
    transforms._write_dataset(
        source=source,
        destination=destination,
        source_indices=[0],
        annotations={},
        trim_config={
            "enabled": True,
            "method": "stationary",
            "recompute_statistics": False,
            "episode_overrides": {0: {"start_frame": 2, "end_frame": 8}},
        },
    )
    assert (
        sorted(p.read_bytes() for p in (destination / "videos").rglob("*.mp4"))
        == source_videos
    )
    metadata = pd.read_parquet(next((destination / "meta/episodes").rglob("*.parquet")))
    assert metadata.length.tolist() == [6]
    for column in metadata:
        if column.endswith("/from_timestamp"):
            assert metadata[column].tolist() == pytest.approx([0.2])
        if column.endswith("/to_timestamp"):
            assert metadata[column].tolist() == pytest.approx([0.8])
    result = validate_dataset_root(destination, mode="export_gate")
    assert result["passed"], result["issues"]
    video = next((destination / "videos").rglob("*.mp4"))
    video.write_bytes(b"broken video")
    assert not validate_dataset_root(destination, mode="export_gate")["passed"]


def test_recipe_option_roundtrips_into_immutable_snapshot(client, database):
    from test_curation_api import _profile, _register

    dataset = _register(database)
    profile = _profile(client, "Optional statistics")
    response = client.post(
        f"/api/v1/datasets/{dataset['id']}/recipes",
        json={
            "profile_id": profile["id"],
            "name": "Trim without statistics",
            "selection_mode": "all",
            "trim_config": {"enabled": True, "recompute_statistics": False},
        },
    )
    assert response.status_code == 201, response.text
    recipe = response.json()
    assert recipe["trim_config"]["recompute_statistics"] is False
    response = client.post(
        f"/api/v1/recipes/{recipe['id']}/runs",
        json={
            "profile_id": profile["id"],
            "output_name": "trim-no-stats",
            "idempotency_key": "optional-stats-snapshot",
        },
    )
    assert response.status_code == 202, response.text
    job = response.json()
    snapshot = database.get_curation_snapshot(job["payload"]["snapshot_id"])
    assert snapshot["trim_config"]["recompute_statistics"] is False
