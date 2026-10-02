from __future__ import annotations

import copy
import math
import os
import shutil
import stat
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from datasetui.dataset_io.files import safe_child, write_json, write_json_lines
from datasetui.dataset_io.stats import write_stats
from datasetui.dataset_io.tables import (
    episode_task_names,
    language_column_types,
    write_parquet,
    write_v3_tasks,
)
from datasetui.job_progress import report_progress
from datasetui.transform_errors import CurationTransformError



ProgressCallback = Callable[[dict[str, Any]], None]
_COPY_CHUNK_BYTES = 1024 * 1024


def _open_regular_file(path: Path, root: Path) -> int:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise CurationTransformError("Source episode video is unavailable") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise CurationTransformError("Source episode video is unavailable")
    opened: list[int] = []
    try:
        current = os.open(
            root, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
        )
        opened.append(current)
        for component in relative.parts[:-1]:
            current = os.open(
                component,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY,
                dir_fd=current,
            )
            opened.append(current)
        descriptor = os.open(
            relative.parts[-1],
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=current,
        )
    except OSError as exc:
        raise CurationTransformError("Source episode video is unavailable") from exc
    finally:
        for directory in reversed(opened):
            os.close(directory)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise CurationTransformError("Source episode video is unavailable")
    return descriptor


def _regular_file_size(path: Path, root: Path) -> int:
    descriptor = _open_regular_file(path, root)
    try:
        return os.fstat(descriptor).st_size
    finally:
        os.close(descriptor)


def _copy_regular_file(
    source: Path,
    destination: Path,
    source_root: Path,
    *,
    copied: int,
    total: int,
    on_progress: ProgressCallback | None,
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = _open_regular_file(source, source_root)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CurationTransformError("Source episode video is unavailable")
        with os.fdopen(descriptor, "rb", closefd=False) as input_stream:
            with destination.open("xb") as output_stream:
                while True:
                    chunk = input_stream.read(_COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    output_stream.write(chunk)
                    copied += len(chunk)
                    report_progress(
                        on_progress,
                        stage="video",
                        completed=copied,
                        total=total,
                        unit="bytes",
                        current_item="원본 영상 복사 (재인코딩 없음)",
                    )
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise CurationTransformError("Source episode video changed during merge")
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)
    return copied


def _metadata_without_paths(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if not key.startswith("_")
        and not key.startswith("data/")
        and not key.startswith("videos/")
        and not key.startswith("stats/")
        and not key.startswith("meta/episodes/")
        and key
        not in {
            "episode_index",
            "length",
            "tasks",
            "dataset_from_index",
            "dataset_to_index",
        }
    }


def write_preserved_merge(
    *,
    source: Any,
    destination: Path,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Write a merge while copying every source video bitstream unchanged."""
    (destination / "meta").mkdir(parents=True)
    (destination / "data").mkdir()
    if source.video_keys:
        (destination / "videos").mkdir()
    readme = source.root / "README.md"
    if readme.is_file() and not readme.is_symlink():
        shutil.copy2(readme, destination / "README.md")

    episodes: list[tuple[pd.DataFrame, dict[str, Any], int, int]] = []
    lineage: list[dict[str, Any]] = []
    global_index = 0
    report_progress(
        on_progress,
        stage="read",
        completed=0,
        total=source.total_episodes,
        unit="episodes",
        force=True,
    )
    for output_index in range(source.total_episodes):
        data, metadata = source.episode(output_index)
        data = data.copy().reset_index(drop=True)
        data["episode_index"] = np.full(len(data), output_index, dtype=np.int64)
        data["index"] = np.arange(
            global_index, global_index + len(data), dtype=np.int64
        )
        global_index += len(data)
        source_index, local_index = source._episodes[output_index]
        metadata = {
            **metadata,
            "_source_episode_index": output_index,
            "_merge_source_index": source_index,
            "_merge_local_episode": local_index,
        }
        episodes.append((data, metadata, 0, len(data)))
        lineage.append(
            {
                "source_episode_index": output_index,
                "output_episode_index": output_index,
                "source_start_frame": 0,
                "source_end_frame": len(data),
                "output_length": len(data),
                "trim_method": "disabled",
                "task_overridden": False,
                "persistent_annotations": 0,
                "event_annotations": 0,
            }
        )
        report_progress(
            on_progress,
            stage="read",
            completed=output_index + 1,
            total=source.total_episodes,
            unit="episodes",
            current_item=f"에피소드 {output_index}",
        )

    tasks = [
        {"task_index": index, "task": text}
        for index, text in sorted(source.tasks.items())
    ]
    language_types = language_column_types(episodes)
    if source.version == "v3.0":
        _write_v3_preserved(
            source, destination, episodes, tasks, language_types, on_progress
        )
    else:
        _write_v2_preserved(
            source, destination, episodes, tasks, language_types, on_progress
        )
    from datasetui.deferred_statistics import read_deferred_statistics, preserve_deferred_statistics

    deferred_inputs = [read_deferred_statistics(item.root) for item in source.sources]
    defer_statistics = any(deferred_inputs)
    used_legacy_aggregate = False
    legacy_aggregate_eligible = source.version in {"v2.0", "v2.1"} and not defer_statistics
    if legacy_aggregate_eligible:
        from datasetui.official_operations import (
            write_legacy_aggregated_statistics,
        )

        used_legacy_aggregate = write_legacy_aggregated_statistics(
            destination,
            [
                (
                    source.sources[int(metadata["_merge_source_index"])].root,
                    int(metadata["_merge_local_episode"]),
                    data,
                )
                for data, metadata, _, _ in episodes
            ],
        )
    if not used_legacy_aggregate and not defer_statistics:
        if legacy_aggregate_eligible:
            report_progress(
                on_progress,
                stage="statistics",
                completed=0,
                total=0,
                unit="items",
                current_item=(
                    "에피소드 통계가 없거나 불완전하여 전체 통계를 재계산합니다"
                ),
                force=True,
            )
        write_stats(
            destination / "meta" / "stats.json",
            [item[0] for item in episodes],
            on_progress=on_progress,
        )
    from datasetui.output_metadata import copy_modality_metadata

    copy_modality_metadata([item.root for item in source.sources], destination)
    deferred_result = None
    if defer_statistics:
        deferred_result = preserve_deferred_statistics(source.sources[0].root, destination)
        report_progress(on_progress, stage="statistics", completed=1, total=1,
                         unit="items", current_item="분포 통계 미재계산 상태 유지 · 학습 전 norm_stats 계산 필요")
    return {
        "lineage": lineage,
        "relative_action": {
            "enabled": False,
            "dimensions": [],
            "statistics": {},
        },
        "statistics_reused": used_legacy_aggregate,
        "statistics": deferred_result or {
            "policy": (
                "lerobot-official-aggregate-v1"
                if used_legacy_aggregate
                else "exact-global-numeric-sampled-rgb-v1"
            ),
            "source": (
                "legacy-episode-statistics"
                if used_legacy_aggregate
                else "full-output-recompute"
            ),
            "fallback": legacy_aggregate_eligible and not used_legacy_aggregate,
        },
    }


def _write_v2_preserved(
    source: Any,
    root: Path,
    episodes: list[tuple[pd.DataFrame, dict[str, Any], int, int]],
    tasks: list[dict[str, Any]],
    language_types: dict[str, Any],
    on_progress: ProgressCallback | None,
) -> None:
    info = copy.deepcopy(source.info)
    info.pop("total_chunks", None)
    info.update(
        data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        total_episodes=len(episodes),
        total_frames=sum(len(item[0]) for item in episodes),
        total_tasks=len(tasks),
        total_videos=len(episodes) * len(source.video_keys),
        splits={"train": f"0:{len(episodes)}"},
    )
    write_json(root / "meta/info.json", info)
    write_json_lines(root / "meta/tasks.jsonl", tasks)
    chunk_size = int(info.get("chunks_size", 1000))
    episode_rows: list[dict[str, Any]] = []
    report_progress(
        on_progress,
        stage="write",
        completed=0,
        total=len(episodes),
        unit="episodes",
        force=True,
    )
    for index, (data, metadata, _, _) in enumerate(episodes):
        relative = info["data_path"].format(
            episode_chunk=index // chunk_size, episode_index=index
        )
        path = safe_child(root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_parquet(data, path, language_types)
        episode_rows.append(
            {
                **_metadata_without_paths(metadata),
                "episode_index": index,
                "tasks": episode_task_names(data, tasks),
                "length": len(data),
            }
        )
        report_progress(
            on_progress,
            stage="write",
            completed=index + 1,
            total=len(episodes),
            unit="episodes",
            current_item=f"에피소드 {index}",
        )
    write_json_lines(root / "meta/episodes.jsonl", episode_rows)

    copies: list[tuple[Path, Path, Path]] = []
    for index, (_, metadata, _, _) in enumerate(episodes):
        for key in source.video_keys:
            source_path, _ = source.video_source(index, key, metadata)
            destination = safe_child(
                root,
                info["video_path"].format(
                    episode_chunk=index // chunk_size,
                    episode_index=index,
                    video_key=key,
                ),
            )
            source_number = int(metadata["_merge_source_index"])
            copies.append(
                (source_path, destination, source.sources[source_number].root)
            )
    _copy_videos(copies, on_progress)


def _write_v3_preserved(
    source: Any,
    root: Path,
    episodes: list[tuple[pd.DataFrame, dict[str, Any], int, int]],
    tasks: list[dict[str, Any]],
    language_types: dict[str, Any],
    on_progress: ProgressCallback | None,
) -> None:
    info = copy.deepcopy(source.info)
    info.update(
        total_episodes=len(episodes),
        total_frames=sum(len(item[0]) for item in episodes),
        total_tasks=len(tasks),
        total_chunks=max(1, (len(episodes) + 999) // 1000),
        chunks_size=1000,
        splits={"train": f"0:{len(episodes)}"},
        data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )
    write_json(root / "meta/info.json", info)
    write_v3_tasks(root, tasks)

    video_locations: dict[tuple[int, str, Path], tuple[int, int]] = {}
    next_file: dict[str, int] = {key: 0 for key in source.video_keys}
    copies: list[tuple[Path, Path, Path]] = []
    metadata_rows: list[dict[str, Any]] = []
    dataset_offset = 0
    report_progress(
        on_progress,
        stage="write",
        completed=0,
        total=len(episodes),
        unit="episodes",
        force=True,
    )
    for index, (data, metadata, _, _) in enumerate(episodes):
        chunk, file_index = index // 1000, index % 1000
        data_path = root / f"data/chunk-{chunk:03d}/file-{file_index:03d}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        write_parquet(data, data_path, language_types)
        row: dict[str, Any] = {
            **_metadata_without_paths(metadata),
            "episode_index": index,
            "tasks": episode_task_names(data, tasks),
            "length": len(data),
            "data/chunk_index": chunk,
            "data/file_index": file_index,
            "dataset_from_index": dataset_offset,
            "dataset_to_index": dataset_offset + len(data),
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        source_number = int(metadata["_merge_source_index"])
        for key in source.video_keys:
            prefix = f"videos/{key}"
            from_field = f"{prefix}/from_timestamp"
            to_field = f"{prefix}/to_timestamp"
            if from_field not in metadata or to_field not in metadata:
                raise CurationTransformError(
                    "Source v3 video segment timestamps are missing"
                )
            try:
                from_timestamp = float(metadata[from_field])
                to_timestamp = float(metadata[to_field])
            except (TypeError, ValueError) as exc:
                raise CurationTransformError(
                    "Source v3 video segment timestamps are invalid"
                ) from exc
            if (
                not math.isfinite(from_timestamp)
                or not math.isfinite(to_timestamp)
                or from_timestamp < 0
                or to_timestamp <= from_timestamp
            ):
                raise CurationTransformError(
                    "Source v3 video segment timestamps are invalid"
                )
            source_path, _ = source.video_source(index, key, metadata)
            identity = (source_number, key, source_path)
            location = video_locations.get(identity)
            if location is None:
                ordinal = next_file[key]
                next_file[key] += 1
                location = ordinal // 1000, ordinal % 1000
                video_locations[identity] = location
                output_path = root / info["video_path"].format(
                    video_key=key,
                    chunk_index=location[0],
                    file_index=location[1],
                )
                copies.append(
                    (source_path, output_path, source.sources[source_number].root)
                )
            row[f"{prefix}/chunk_index"] = location[0]
            row[f"{prefix}/file_index"] = location[1]
            row[from_field] = metadata[from_field]
            row[to_field] = metadata[to_field]
        metadata_rows.append(row)
        dataset_offset += len(data)
        report_progress(
            on_progress,
            stage="write",
            completed=index + 1,
            total=len(episodes),
            unit="episodes",
            current_item=f"에피소드 {index}",
        )
    metadata_path = root / "meta/episodes/chunk-000/file-000.parquet"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metadata_rows).to_parquet(metadata_path, index=False)
    info["total_videos"] = len(copies)
    write_json(root / "meta/info.json", info)
    _copy_videos(copies, on_progress)


def _copy_videos(
    copies: list[tuple[Path, Path, Path]], on_progress: ProgressCallback | None
) -> None:
    if not copies:
        return
    total = sum(_regular_file_size(source, root) for source, _, root in copies)
    copied = 0
    report_progress(
        on_progress,
        stage="video",
        completed=0,
        total=total,
        unit="bytes",
        current_item="원본 영상 복사 (재인코딩 없음)",
        force=True,
    )
    for source, destination, source_root in copies:
        copied = _copy_regular_file(
            source,
            destination,
            source_root,
            copied=copied,
            total=total,
            on_progress=on_progress,
        )
    report_progress(
        on_progress,
        stage="video",
        completed=copied,
        total=total,
        unit="bytes",
        current_item="원본 영상 복사 (재인코딩 없음)",
        force=True,
    )
