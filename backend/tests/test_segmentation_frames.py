from __future__ import annotations

import json
import os
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import datasetui.segmentation.frames as segmentation_frames
from datasetui.database import Database, RecipeRevisionMismatchError
from datasetui.datasets import inspect_dataset
from datasetui.segmentation.source import dataset_scope
from datasetui.segmentation.frames import create_frame_snapshot, read_snapshot_frame
from test_segmentation import _registered, _write_video
from test_transforms import _settings
from test_validation_conversion import _write_v3


def _rescan(database: Database, settings, relative_path: str) -> dict:
    candidate = inspect_dataset(
        area_root=settings.nas_root / "raw",
        storage_area="raw",
        relative_path=relative_path,
    )
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw",
        records=[candidate.as_record()],
        scan_generation=generation,
    )
    return database.list_datasets()[0]


def test_snapshot_reads_scope_bound_frame_and_rejects_bad_selections(
    tmp_path: Path,
) -> None:
    settings, database, _, dataset = _registered(tmp_path)
    scope = dataset_scope(database, settings, dataset["id"])

    token = create_frame_snapshot(database, settings, dataset["id"], scope)

    with Image.open(
        BytesIO(
            read_snapshot_frame(
                database,
                settings,
                dataset["id"],
                token,
                0,
                "observation.images.top",
                2,
            )
        )
    ) as image:
        assert image.size == (16, 16)
        assert image.format == "PNG"

    with database.connect() as connection:
        row = connection.execute(
            "SELECT refs_json FROM segmentation_frame_snapshots WHERE token = ?",
            (token,),
        ).fetchone()
    references = json.loads(row["refs_json"])
    assert len(references) == 2
    assert all(len(reference["sha256"]) == 64 for reference in references)

    with pytest.raises(ValueError, match="token"):
        read_snapshot_frame(
            database,
            settings,
            dataset["id"],
            "not-a-uuid",
            0,
            "observation.images.top",
            0,
        )
    with pytest.raises(ValueError, match="unavailable"):
        read_snapshot_frame(
            database,
            settings,
            dataset["id"],
            str(uuid.uuid4()),
            0,
            "observation.images.top",
            0,
        )
    with pytest.raises(ValueError, match="outside"):
        read_snapshot_frame(
            database,
            settings,
            dataset["id"],
            token,
            0,
            "observation.images.top",
            4,
        )
    with pytest.raises(ValueError, match="selection"):
        read_snapshot_frame(
            database,
            settings,
            dataset["id"],
            token,
            0,
            "unknown.camera",
            0,
        )


def test_snapshot_detects_same_size_video_change_with_restored_mtime(
    tmp_path: Path,
) -> None:
    settings, database, root, dataset = _registered(tmp_path)
    scope = dataset_scope(database, settings, dataset["id"])
    token = create_frame_snapshot(database, settings, dataset["id"], scope)
    path = root / "videos/chunk-000/observation.images.top/episode_000000.mp4"
    before = path.stat()
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))

    assert path.stat().st_size == before.st_size
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    with pytest.raises(RecipeRevisionMismatchError):
        read_snapshot_frame(
            database,
            settings,
            dataset["id"],
            token,
            0,
            "observation.images.top",
            0,
        )


def test_snapshot_detects_video_mutation_during_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, database, root, dataset = _registered(tmp_path)
    scope = dataset_scope(database, settings, dataset["id"])
    token = create_frame_snapshot(database, settings, dataset["id"], scope)
    path = root / "videos/chunk-000/observation.images.top/episode_000000.mp4"

    def mutate_while_decoding(video_path: Path, index: int) -> np.ndarray:
        assert video_path == path
        assert index == 0
        before = video_path.stat()
        content = bytearray(video_path.read_bytes())
        content[-1] ^= 1
        video_path.write_bytes(content)
        os.utime(video_path, ns=(before.st_atime_ns, before.st_mtime_ns))
        return np.zeros((16, 16, 3), dtype=np.uint8)

    monkeypatch.setattr(segmentation_frames, "_decode_one", mutate_while_decoding)
    with pytest.raises(RecipeRevisionMismatchError):
        read_snapshot_frame(
            database,
            settings,
            dataset["id"],
            token,
            0,
            "observation.images.top",
            0,
        )


def test_snapshot_is_invalid_after_registry_fingerprint_changes(tmp_path: Path) -> None:
    settings, database, root, dataset = _registered(tmp_path)
    registry_fingerprint = database.get_dataset(dataset["id"])["fingerprint"]
    scope = dataset_scope(database, settings, dataset["id"])
    token = create_frame_snapshot(database, settings, dataset["id"], scope)
    # RTX retains a metadata-only registry marker; change that marker rather
    # than assuming a README edit changes every registry implementation.
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["robot_type"] = "changed-robot"
    info_path.write_text(json.dumps(info), encoding="utf-8")

    rescanned = _rescan(database, settings, "lab/source")
    assert rescanned["fingerprint"] != registry_fingerprint
    with pytest.raises(RecipeRevisionMismatchError):
        read_snapshot_frame(
            database,
            settings,
            dataset["id"],
            token,
            0,
            "observation.images.top",
            0,
        )


def test_snapshot_uses_stored_shared_video_offset_not_live_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    root = settings.nas_root / "raw/lab/v3"
    _write_v3(root)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["observation.images.top"] = {
        "dtype": "video",
        "shape": [16, 16, 3],
        "names": ["height", "width", "channel"],
    }
    info_path.write_text(json.dumps(info), encoding="utf-8")
    metadata_path = root / "meta/episodes/chunk-000/file-000.parquet"
    metadata = pd.read_parquet(metadata_path)
    metadata["videos/observation.images.top/chunk_index"] = [0, 0]
    metadata["videos/observation.images.top/file_index"] = [0, 0]
    metadata["videos/observation.images.top/from_timestamp"] = [0.0, 0.4]
    metadata.to_parquet(metadata_path, index=False)
    video_path = root / "videos/observation.images.top/chunk-000/file-000.mp4"
    _write_video(video_path, [(0, 0, 180)] * 4 + [(20, 180, 20)] * 4)
    database = Database(settings.database_path)
    database.initialize()
    dataset = _rescan(database, settings, "lab/v3")
    scope = dataset_scope(database, settings, dataset["id"])
    hash_calls: list[Path] = []
    original_hash = segmentation_frames.verified_file_sha256

    def count_hash(path: Path) -> str:
        hash_calls.append(path)
        return original_hash(path)

    monkeypatch.setattr(segmentation_frames, "verified_file_sha256", count_hash)
    token = create_frame_snapshot(database, settings, dataset["id"], scope)
    assert hash_calls == [video_path]

    # Reshape the live episode metadata without rescanning. Snapshot reads must
    # retain the verified offset captured with the scope rather than reloading it.
    metadata["videos/observation.images.top/from_timestamp"] = [0.0, 0.0]
    metadata.to_parquet(metadata_path, index=False)
    with Image.open(
        BytesIO(
            read_snapshot_frame(
                database,
                settings,
                dataset["id"],
                token,
                1,
                "observation.images.top",
                0,
            )
        )
    ) as image:
        pixel = np.asarray(image)[8, 8]
    assert int(pixel[1]) > int(pixel[2])


def test_snapshot_rejects_corrupt_relative_path(tmp_path: Path) -> None:
    settings, database, _, dataset = _registered(tmp_path)
    scope = dataset_scope(database, settings, dataset["id"])
    token = create_frame_snapshot(database, settings, dataset["id"], scope)
    with database.connect() as connection:
        row = connection.execute(
            "SELECT refs_json FROM segmentation_frame_snapshots WHERE token = ?",
            (token,),
        ).fetchone()
        references = json.loads(row["refs_json"])
        references[0]["video_path"] = "../outside.mp4"
        connection.execute(
            "UPDATE segmentation_frame_snapshots SET refs_json = ? WHERE token = ?",
            (json.dumps(references), token),
        )

    with pytest.raises(ValueError, match="path"):
        read_snapshot_frame(
            database,
            settings,
            dataset["id"],
            token,
            0,
            "observation.images.top",
            0,
        )


def test_snapshot_supports_legacy_metadata_registry_fingerprint(tmp_path: Path):
    import hashlib

    settings, database, root, dataset = _registered(tmp_path)
    metadata_fingerprint = hashlib.sha256(
        (root / "meta/info.json").read_bytes()
    ).hexdigest()
    with database.connect() as connection:
        connection.execute(
            "UPDATE datasets SET fingerprint=? WHERE id=?",
            (metadata_fingerprint, dataset["id"]),
        )
    scope = dataset_scope(database, settings, dataset["id"])
    assert scope["fingerprint"] != metadata_fingerprint
    token = create_frame_snapshot(database, settings, dataset["id"], scope)
    frame = read_snapshot_frame(
        database, settings, dataset["id"], token, 0, "observation.images.top", 0
    )
    assert frame.startswith(b"\x89PNG")
    assert database.get_dataset(dataset["id"])["fingerprint"] == metadata_fingerprint
