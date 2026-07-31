"""Metadata-only loader: builds DatasetDocument without reading row data."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .document import (
    DatasetDocument,
    DatasetVersion,
    EpisodeRef,
    FeatureSpec,
    MediaRef,
    ValidationResult,
)
from .version import detect_version, uses_legacy_episode_layout


def _load_info(dataset_path: Path) -> dict[str, Any]:
    """Load and return info.json contents."""
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path, encoding="utf-8") as f:
        return json.load(f)


def _parse_features(info: dict[str, Any]) -> tuple[FeatureSpec, ...]:
    """Parse features from info.json into frozen specs."""
    features_raw = info.get("features", {})
    specs: list[FeatureSpec] = []
    for name, spec in features_raw.items():
        dtype = spec.get("dtype", "unknown")
        shape = tuple(spec.get("shape", []))
        names_raw = spec.get("names")
        names = tuple(names_raw) if isinstance(names_raw, list) else None
        specs.append(FeatureSpec(name=name, dtype=dtype, shape=shape, names=names))
    return tuple(specs)


def _load_episodes_v21(dataset_path: Path) -> tuple[EpisodeRef, ...]:
    """Load episodes from meta/episodes.jsonl (v2.1 format)."""
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    episodes: list[EpisodeRef] = []
    with open(episodes_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ep_idx = row["episode_index"]
            length = row["length"]
            tasks = tuple(row.get("tasks", ()))
            episodes.append(EpisodeRef(
                index=ep_idx,
                length=length,
                chunk_index=0,
                file_index=ep_idx,
                tasks=tasks,
                frame_start=0,
                frame_end=0,
            ))
    return tuple(episodes)


def _load_episodes_v30(dataset_path: Path) -> tuple[EpisodeRef, ...]:
    """Load episodes from meta/episodes Parquet files (v3.0 format).

    Uses only ParquetFile metadata + read_row_group for the episodes metadata
    file (which is small metadata about episodes, not frame data).
    """
    episodes_dir = dataset_path / "meta" / "episodes"
    episodes: list[EpisodeRef] = []

    if not episodes_dir.exists():
        return ()

    parquet_files = sorted(episodes_dir.rglob("*.parquet"))
    for pq_path in parquet_files:
        pf = pq.ParquetFile(pq_path)
        # Episodes parquet is small metadata — read via row groups
        for rg_idx in range(pf.metadata.num_row_groups):
            rg_table = pf.read_row_group(rg_idx)
            for i in range(rg_table.num_rows):
                row = {col: rg_table.column(col)[i].as_py() for col in rg_table.column_names}
                ep_idx = row.get("episode_index", i)
                length = row.get("length", 0)
                tasks_raw = row.get("tasks", [])
                tasks = tuple(tasks_raw) if isinstance(tasks_raw, list) else ()
                chunk_idx = row.get("data/chunk_index", 0)
                file_idx = row.get("data/file_index", 0)
                episodes.append(EpisodeRef(
                    index=ep_idx,
                    length=length,
                    chunk_index=chunk_idx,
                    file_index=file_idx,
                    tasks=tasks,
                    frame_start=0,
                    frame_end=0,
                ))
    return tuple(episodes)




def _assign_frame_ranges(episodes: tuple[EpisodeRef, ...]) -> tuple[EpisodeRef, ...]:
    """Return episode-index order with cumulative half-open frame ranges."""
    cursor = 0
    ranged: list[EpisodeRef] = []
    for episode in sorted(episodes, key=lambda item: item.index):
        frame_end = cursor + episode.length
        ranged.append(replace(episode, frame_start=cursor, frame_end=frame_end))
        cursor = frame_end
    return tuple(ranged)
def _parse_media(info: dict[str, Any], episodes: tuple[EpisodeRef, ...]) -> tuple[MediaRef, ...]:
    """Parse media references from info.json video_path template."""
    video_path = info.get("video_path")
    if not video_path:
        return ()

    features = info.get("features", {})
    video_keys = [
        name
        for name, spec in features.items()
        if isinstance(spec, dict) and spec.get("dtype") in {"image", "video"}
    ]
    if not video_keys:
        return ()

    media: list[MediaRef] = []
    ep_indices = tuple(ep.index for ep in episodes)
    for key in video_keys:
        media.append(MediaRef(
            video_key=key,
            path_template=video_path,
            episodes_with_video=ep_indices,
        ))
    return tuple(media)


def _detect_annotations(info: dict[str, Any]) -> tuple[bool, tuple[str, ...] | None]:
    """Detect if v3.1 language annotation columns are present."""
    features = info.get("features", {})
    annotation_keys: list[str] = []
    for name, spec in features.items():
        if isinstance(spec, dict) and spec.get("dtype") == "language":
            annotation_keys.append(name)
    if annotation_keys:
        return True, tuple(sorted(annotation_keys))
    return False, None


def _load_provenance(dataset_path: Path) -> str | None:
    """Read PROVENANCE.md if it exists."""
    prov_path = dataset_path / "PROVENANCE.md"
    if prov_path.exists():
        return prov_path.read_text(encoding="utf-8")
    return None


def _load_tasks_v21(dataset_path: Path) -> tuple[str, ...]:
    """Load tasks from meta/tasks.jsonl."""
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        return ()
    tasks: list[str] = []
    with open(tasks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tasks.append(row.get("task", ""))
    return tuple(tasks)


def _load_tasks_v30(dataset_path: Path) -> tuple[str, ...]:
    """Load tasks from meta/tasks.parquet."""
    tasks_path = dataset_path / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return ()
    pf = pq.ParquetFile(tasks_path)
    tasks: list[str] = []
    for rg_idx in range(pf.metadata.num_row_groups):
        rg_table = pf.read_row_group(rg_idx)
        if "task" in rg_table.column_names:
            for val in rg_table.column("task"):
                tasks.append(val.as_py())
    return tuple(tasks)


def load_document(dataset_path: str | Path) -> DatasetDocument:
    """Load a DatasetDocument from a dataset directory using metadata only.

    Never calls pq.read_table() — uses only ParquetFile metadata and
    row-group reads for small metadata Parquet files.
    """
    dataset_path = Path(dataset_path).resolve()
    info = _load_info(dataset_path)
    version = detect_version(info)

    features = _parse_features(info)

    # Load episodes based on version
    if uses_legacy_episode_layout(version):
        episodes = _load_episodes_v21(dataset_path)
        tasks = _load_tasks_v21(dataset_path)
    else:
        episodes = _load_episodes_v30(dataset_path)
        tasks = _load_tasks_v30(dataset_path)

    episodes = _assign_frame_ranges(episodes)
    media = _parse_media(info, episodes)
    has_annotations, annotation_styles = _detect_annotations(info)
    provenance = _load_provenance(dataset_path)

    total_frames = info.get("total_frames", sum(ep.length for ep in episodes))
    fps = info.get("fps", 0)

    return DatasetDocument(
        source_path=str(dataset_path),
        version=version,
        features=features,
        episodes=episodes,
        total_frames=total_frames,
        fps=fps,
        media=media,
        tasks=tasks,
        has_annotations=has_annotations,
        annotation_styles=annotation_styles,
        provenance=provenance,
        validation=ValidationResult(valid=True, errors=(), warnings=()),
        indexed_at=datetime.now(timezone.utc).isoformat(),
    )
