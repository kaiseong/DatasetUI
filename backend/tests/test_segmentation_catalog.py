import pytest

from datasetui.segmentation.catalog import dataset_catalog, selected_scope
from datasetui.segmentation.frames import read_snapshot_frame
from test_segmentation import _registered


def test_catalog_does_not_load_source_or_video(tmp_path, monkeypatch):
    settings, database, _, dataset = _registered(tmp_path)
    monkeypatch.setattr(
        "datasetui.segmentation.source.load_source", lambda *a, **k: pytest.fail("full source")
    )
    monkeypatch.setattr(
        "datasetui.segmentation.frames.verified_file_sha256",
        lambda *a: pytest.fail("video hash"),
    )
    result = dataset_catalog(database, settings, dataset["id"])
    assert result["total_episodes"] == 1
    assert result["video_keys"] == ["observation.images.top", "observation.images.side"]
    assert "fingerprint" not in result
    assert len(result["metadata_revision"]) == 64


def test_selected_scope_reads_only_selected_video(tmp_path, monkeypatch):
    settings, database, _, dataset = _registered(tmp_path)
    monkeypatch.setattr(
        "datasetui.segmentation.source.load_source", lambda *a, **k: pytest.fail("full source")
    )
    result = selected_scope(
        database, settings, dataset["id"], 0, "observation.images.top"
    )
    assert result["length"] == 4
    assert "fingerprint" not in result
    assert read_snapshot_frame(
        database,
        settings,
        dataset["id"],
        result["frame_token"],
        0,
        "observation.images.top",
        1,
    )
    with pytest.raises(ValueError):
        read_snapshot_frame(
            database,
            settings,
            dataset["id"],
            result["frame_token"],
            1,
            "observation.images.top",
            0,
        )


def test_catalog_rejects_unknown_selection(tmp_path):
    settings, database, _, dataset = _registered(tmp_path)
    with pytest.raises(ValueError):
        selected_scope(database, settings, dataset["id"], 9, "observation.images.top")
    with pytest.raises(ValueError):
        selected_scope(database, settings, dataset["id"], 0, "../bad")


def test_snapshot_rejects_metadata_mutation(tmp_path):
    import json
    from datasetui.database import RecipeRevisionMismatchError

    settings, database, root, dataset = _registered(tmp_path)
    result = selected_scope(
        database, settings, dataset["id"], 0, "observation.images.top"
    )
    path = root / "meta/episodes.jsonl"
    row = json.loads(path.read_text().splitlines()[0])
    row["length"] = 3
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(RecipeRevisionMismatchError):
        read_snapshot_frame(
            database,
            settings,
            dataset["id"],
            result["frame_token"],
            0,
            "observation.images.top",
            0,
        )


def test_unselected_camera_is_not_opened(tmp_path, monkeypatch):
    settings, database, root, dataset = _registered(tmp_path)
    import datasetui.segmentation.catalog as catalog

    original = catalog.verified_file_sha256
    opened = []

    def hashed(path):
        opened.append(str(path))
        assert "observation.images.side" not in str(path)
        assert "/data/" not in str(path.relative_to(root))
        return original(path)

    monkeypatch.setattr(catalog, "verified_file_sha256", hashed)
    selected_scope(database, settings, dataset["id"], 0, "observation.images.top")
    assert any(path.endswith(".mp4") for path in opened)
