from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot_dataset_editor.dataset.validation import validate_dataset


def test_total_episode_count_detects_missing_trailing_episode(
    v21_fixture: Path, tmp_path: Path
) -> None:
    dataset = tmp_path / "missing-tail"
    shutil.copytree(v21_fixture, dataset)
    episodes = (dataset / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    (dataset / "meta" / "episodes.jsonl").write_text(episodes[0] + "\n", encoding="utf-8")

    result = validate_dataset(dataset)
    assert result.valid is False
    assert any("total_episodes" in error for error in result.errors)


def test_schema_drift_is_checked_in_every_data_file(v21_fixture: Path, tmp_path: Path) -> None:
    dataset = tmp_path / "later-drift"
    shutil.copytree(v21_fixture, dataset)
    second = sorted((dataset / "data").rglob("*.parquet"))[1]
    original = pq.read_table(second)
    drifted = original.drop(["action"])
    pq.write_table(drifted, second)

    result = validate_dataset(dataset)
    assert result.valid is False
    assert any("action" in error and second.name in error for error in result.errors)


def test_corrupt_episode_metadata_parquet_is_reported(v30_fixture: Path, tmp_path: Path) -> None:
    dataset = tmp_path / "corrupt-meta"
    shutil.copytree(v30_fixture, dataset)
    episodes = next((dataset / "meta" / "episodes").rglob("*.parquet"))
    episodes.write_bytes(b"not parquet metadata")

    result = validate_dataset(dataset)
    assert result.valid is False
    assert any("meta/episodes" in error and "Corrupt" in error for error in result.errors)


def test_nan_timestamp_is_invalid(v21_fixture: Path, tmp_path: Path) -> None:
    dataset = tmp_path / "nan-time"
    shutil.copytree(v21_fixture, dataset)
    first = sorted((dataset / "data").rglob("*.parquet"))[0]
    table = pq.read_table(first)
    values = table.column("timestamp").to_pylist()
    values[1] = float("nan")
    replacement = pa.array(values, type=table.schema.field("timestamp").type)
    index = table.schema.get_field_index("timestamp")
    table = table.set_column(index, "timestamp", replacement)
    pq.write_table(table, first)

    result = validate_dataset(dataset)
    assert result.valid is False
    assert any("finite" in error.lower() or "nan" in error.lower() for error in result.errors)


def test_missing_data_file_is_detected_by_footer_row_totals(
    v21_fixture: Path, tmp_path: Path
) -> None:
    dataset = tmp_path / "missing-data-file"
    shutil.copytree(v21_fixture, dataset)
    sorted((dataset / "data").rglob("*.parquet"))[1].unlink()

    result = validate_dataset(dataset)
    assert result.valid is False
    assert any("data rows" in error and "total_frames" in error for error in result.errors)


def test_primitive_type_drift_is_detected_in_later_file(
    v21_fixture: Path, tmp_path: Path
) -> None:
    dataset = tmp_path / "type-drift"
    shutil.copytree(v21_fixture, dataset)
    second = sorted((dataset / "data").rglob("*.parquet"))[1]
    table = pq.read_table(second)
    action_index = table.schema.get_field_index("action")
    action_values = table.column("action").to_pylist()
    action_width = len(action_values[0])
    table = table.set_column(
        action_index,
        "action",
        pa.array(action_values, type=pa.list_(pa.float64(), action_width)),
    )
    pq.write_table(table, second)

    result = validate_dataset(dataset)
    assert result.valid is False
    assert any("type drift" in error.lower() and "action" in error for error in result.errors)


def test_video_dtype_participates_in_missing_media_validation(
    v21_fixture: Path, tmp_path: Path
) -> None:
    dataset = tmp_path / "video-dtype"
    shutil.copytree(v21_fixture, dataset)
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["video_path"] = (
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )
    info["features"]["observation.images.missing"] = {
        "dtype": "video",
        "shape": [2, 2, 3],
        "names": None,
    }
    info_path.write_text(json.dumps(info), encoding="utf-8")

    result = validate_dataset(dataset)
    assert result.valid is False
    assert any("Missing video file" in error for error in result.errors)


def test_parquet_referenced_image_path_must_exist(v21_fixture: Path, tmp_path: Path) -> None:
    dataset = tmp_path / "missing-image"
    shutil.copytree(v21_fixture, dataset)
    parquet_path = sorted((dataset / "data").rglob("*.parquet"))[0]
    table = pq.read_table(parquet_path)
    media = table["observation.images.top"].to_pylist()
    image_path = dataset / "images" / "observation.images.top" / "episode_000000" / "frame_000000.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(media[0]["bytes"])
    media[0] = {"bytes": None, "path": image_path.name}
    index = table.schema.get_field_index("observation.images.top")
    table = table.set_column(index, "observation.images.top", pa.array(media, type=table.schema.field(index).type))
    pq.write_table(table, parquet_path)
    image_path.unlink()

    result = validate_dataset(dataset)
    assert result.valid is False
    assert any("Missing image file" in error and "frame_000000.png" in error for error in result.errors)
