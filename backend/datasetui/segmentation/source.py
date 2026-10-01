"""Loading and verifying a registered source dataset for segmentation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from datasetui.config import Settings
from datasetui.content_integrity import (
    ContentIntegrityError,
    dataset_content_fingerprint,
)
from datasetui.database import Database, RecipeRevisionMismatchError
from datasetui.segmentation.errors import SegmentationError
from datasetui.segmentation.media import MAX_IMAGE_PIXELS, _decode_one
from datasetui.segmentation.paths import _safe_directory_path, _safe_regular_path
from datasetui.transforms import (
    _DatasetSource,
    _read_regular_bytes,
    _safe_dataset_root,
)


def load_source(
    database: Database,
    settings: Settings,
    dataset_id: str,
    fingerprint: str | None = None,
    *,
    verify_content: bool = True,
) -> tuple[Path, _DatasetSource]:
    record = database.get_dataset(str(dataset_id))
    # Segmentation approvals bind full content independently of older registry
    # deployments whose fingerprint intentionally covers only meta/info.json.
    expected = fingerprint
    if not record["available"] or record["readiness"] != "ready":
        raise RecipeRevisionMismatchError(str(dataset_id))
    root = _safe_dataset_root(
        settings.nas_root, record["storage_area"], record["relative_path"]
    )
    if verify_content:
        try:
            actual = dataset_content_fingerprint(root, reuse_file_digests=True)
        except ContentIntegrityError as exc:
            raise RecipeRevisionMismatchError(str(dataset_id)) from exc
        if expected is not None and actual != expected:
            raise RecipeRevisionMismatchError(str(dataset_id))
    try:
        info_path = _safe_regular_path(root, root / "meta/info.json")
        _safe_directory_path(root, root / "data")
        info = json.loads(_read_regular_bytes(info_path, max_bytes=2 * 1024 * 1024))
        if not isinstance(info, dict):
            raise ValueError
        source = _DatasetSource(root, info)
        if verify_content:
            source.segmentation_fingerprint = actual
    except Exception as exc:
        raise SegmentationError("Dataset source could not be loaded") from exc
    return root, source


def dataset_scope(
    database: Database, settings: Settings, dataset_id: str
) -> dict[str, Any]:
    _, source = load_source(database, settings, dataset_id)
    episodes = []
    for episode_index in range(int(source.info["total_episodes"])):
        data, _ = source.episode(episode_index)
        episodes.append({"episode_index": episode_index, "length": len(data)})
    return {
        "fingerprint": source.segmentation_fingerprint,
        "video_keys": list(source.video_keys),
        "episodes": episodes,
    }


def source_frame(
    database: Database,
    settings: Settings,
    dataset_id: str,
    episode_index: int,
    video_key: str,
    frame_index: int,
) -> bytes:
    root, source = load_source(database, settings, dataset_id, verify_content=False)
    data, metadata = _episode(source, episode_index)
    _validate_video_selection(source, video_key, frame_index, len(data))
    path, start = source.video_source(episode_index, video_key, metadata)
    path = _safe_regular_path(root, path)
    frame = _decode_one(path, start + frame_index)
    if frame.shape[0] * frame.shape[1] > MAX_IMAGE_PIXELS:
        raise SegmentationError("Video frame exceeds the image pixel limit")
    image = Image.fromarray(frame, mode="RGB")
    from io import BytesIO

    output = BytesIO()
    image.save(output, format="PNG", compress_level=6)
    return output.getvalue()


def _episode(
    source: _DatasetSource, episode_index: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    total = int(source.info.get("total_episodes", 0))
    if episode_index < 0 or episode_index >= total:
        raise SegmentationError("Episode is outside the dataset")
    return source.episode(episode_index)


def _validate_video_selection(
    source: _DatasetSource, video_key: str, frame_index: int, frame_count: int
) -> None:
    if video_key not in source.video_keys:
        raise SegmentationError("Video key is not available")
    if frame_index < 0 or frame_index >= frame_count:
        raise SegmentationError("Frame is outside the episode")
