from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from datasetui.exact_statistics import (
    _open_parquet,
    _parquet_files,
    _read_info,
    recompute_numeric_statistics,
)
from datasetui.merge_schema import _compression_policy, _values_equal
from datasetui.transform_errors import CurationTransformError
from datasetui.visual_statistics import recompute_visual_statistics_with_episodes


EPISODE_STATISTICS_ENGINE = "datasetui-exact-episode-statistics-v1"
STATISTIC_ORDER = (
    "min",
    "max",
    "mean",
    "std",
    "count",
    "q01",
    "q10",
    "q50",
    "q90",
    "q99",
)


def write_episode_statistics(
    root: Path, *, on_progress=None, visual_stats_by_episode=None
) -> dict[str, Any]:
    root = Path(root)
    info = _read_info(root / "meta/info.json")
    version = info.get("codebase_version")
    if version not in {"v2.0", "v2.1", "v3.0"}:
        raise CurationTransformError(
            "Unsupported dataset version for episode statistics"
        )
    episode_indices = _episode_indices(root, version)
    if len(episode_indices) != info.get("total_episodes"):
        raise CurationTransformError(
            "Episode metadata count differs from total_episodes"
        )
    if visual_stats_by_episode is None:
        _, visual_stats_by_episode = recompute_visual_statistics_with_episodes(
            root, on_progress=on_progress
        )
    if set(visual_stats_by_episode) != set(episode_indices):
        raise CurationTransformError(
            "Precomputed visual statistics episode indices differ from metadata"
        )
    expected_video_keys = {
        name
        for name, feature in info.get("features", {}).items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    }
    if any(
        not isinstance(visual_stats_by_episode[episode_index], dict)
        or set(visual_stats_by_episode[episode_index]) != expected_video_keys
        for episode_index in episode_indices
    ):
        raise CurationTransformError(
            "Precomputed visual statistics features differ from metadata"
        )
    stats_by_episode: dict[int, dict[str, Any]] = {}
    for completed, episode_index in enumerate(episode_indices, start=1):

        def child_progress(event: dict[str, Any]) -> None:
            if on_progress is None:
                return
            update = dict(event)
            item = update.get("current_item")
            update["current_item"] = (
                f"에피소드 {episode_index} · {item}"
                if item
                else f"에피소드 {episode_index}"
            )
            on_progress(update)

        stats = recompute_numeric_statistics(
            root,
            on_progress=child_progress,
            episode_indices=[episode_index],
        )
        stats.update(visual_stats_by_episode[episode_index])
        stats_by_episode[episode_index] = stats
        if on_progress is not None:
            on_progress(
                {
                    "stage": "statistics",
                    "completed": completed,
                    "total": len(episode_indices),
                    "unit": "episodes",
                    "current_item": f"에피소드 {episode_index} 통계 완료",
                    "_force": True,
                }
            )

    if version in {"v2.0", "v2.1"}:
        output_paths = [_write_v2_episode_statistics(root, stats_by_episode)]
        output_format = "jsonl"
    else:
        output_paths = _write_v3_episode_statistics(root, stats_by_episode)
        output_format = "parquet-columns"
    return {
        "engine": EPISODE_STATISTICS_ENGINE,
        "version": version,
        "episodes": len(episode_indices),
        "format": output_format,
        "paths": [path.relative_to(root).as_posix() for path in output_paths],
    }


def _episode_indices(root: Path, version: str) -> list[int]:
    if version in {"v2.0", "v2.1"}:
        path = root / "meta/episodes.jsonl"
        rows = _read_json_lines(path)
        indices = [_episode_index(row) for row in rows]
    else:
        indices = []
        for path in _parquet_files(root / "meta/episodes"):
            with _open_parquet(path) as parquet:
                if parquet.schema_arrow.get_field_index("episode_index") < 0:
                    raise CurationTransformError(
                        "Episode metadata is missing episode_index"
                    )
                for batch in parquet.iter_batches(columns=["episode_index"]):
                    indices.extend(
                        _episode_index(value) for value in batch.column(0).to_pylist()
                    )
    if len(indices) != len(set(indices)):
        raise CurationTransformError("Episode metadata contains duplicate indices")
    if sorted(indices) != list(range(len(indices))):
        raise CurationTransformError("Episode metadata indices are not contiguous")
    return sorted(indices)


def _episode_index(value: Any) -> int:
    if isinstance(value, dict):
        value = value.get("episode_index")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CurationTransformError("Episode metadata index is invalid")
    return value


def _write_v2_episode_statistics(
    root: Path, stats_by_episode: dict[int, dict[str, Any]]
) -> Path:
    path = root / "meta/episodes_stats.jsonl"
    rows = [
        {"episode_index": episode_index, "stats": stats_by_episode[episode_index]}
        for episode_index in sorted(stats_by_episode)
    ]
    text = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )
    _atomic_write_bytes(path, text.encode("utf-8"))
    return path


def _write_v3_episode_statistics(
    root: Path, stats_by_episode: dict[int, dict[str, Any]]
) -> list[Path]:
    paths = _parquet_files(root / "meta/episodes")
    metadata_indices: set[int] = set()
    for path in paths:
        with _open_parquet(path) as parquet:
            for batch in parquet.iter_batches(columns=["episode_index"]):
                metadata_indices.update(
                    _episode_index(value) for value in batch.column(0).to_pylist()
                )
    if metadata_indices != set(stats_by_episode):
        raise CurationTransformError(
            "Episode metadata and computed statistics indices differ"
        )
    for path in paths:
        _rewrite_v3_metadata_file(path, stats_by_episode)
    return paths


def _rewrite_v3_metadata_file(
    path: Path, stats_by_episode: dict[int, dict[str, Any]]
) -> None:
    original_mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.episode-stats-", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with _open_parquet(path) as parquet:
            compression = _compression_policy(parquet)
            row_groups = [
                _episode_table_with_stats(
                    parquet.read_row_group(index), stats_by_episode
                )
                for index in range(parquet.metadata.num_row_groups)
            ]
            if not row_groups:
                raise CurationTransformError("Episode metadata contains no row groups")
            with pq.ParquetWriter(
                temporary,
                row_groups[0].schema,
                compression=compression,
            ) as writer:
                for table in row_groups:
                    if not table.schema.equals(row_groups[0].schema):
                        raise CurationTransformError(
                            "Episode statistics schema differs between row groups"
                        )
                    writer.write_table(table, row_group_size=len(table))
        _verify_v3_metadata_rewrite(path, temporary, stats_by_episode)
        os.chmod(temporary, original_mode, follow_symlinks=False)
        _fsync_file(temporary)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _episode_table_with_stats(
    table: pa.Table, stats_by_episode: dict[int, dict[str, Any]]
) -> pa.Table:
    if "episode_index" not in table.column_names:
        raise CurationTransformError("Episode metadata is missing episode_index")
    episode_indices = [
        _episode_index(value) for value in table.column("episode_index").to_pylist()
    ]
    base_names = [name for name in table.column_names if not name.startswith("stats/")]
    arrays = [table.column(name).combine_chunks() for name in base_names]
    fields = [table.schema.field(name) for name in base_names]
    stat_columns = _statistic_columns(stats_by_episode)
    for column_name in stat_columns:
        values = [
            _statistic_value(stats_by_episode[episode_index], column_name)
            for episode_index in episode_indices
        ]
        value_type = _statistic_arrow_type(column_name, values)
        arrays.append(pa.array(values, type=value_type, safe=True))
        fields.append(pa.field(column_name, value_type))
    return pa.Table.from_arrays(
        arrays,
        schema=pa.schema(fields, metadata=table.schema.metadata),
    )


def _statistic_columns(stats_by_episode: dict[int, dict[str, Any]]) -> list[str]:
    expected: set[str] | None = None
    for episode_index, stats in stats_by_episode.items():
        if any("/" in feature for feature in stats):
            raise CurationTransformError(
                "Episode statistic feature names cannot contain slashes"
            )
        columns = {
            f"stats/{feature}/{statistic}"
            for feature, feature_stats in stats.items()
            for statistic in feature_stats
        }
        if expected is None:
            expected = columns
        elif columns != expected:
            raise CurationTransformError(
                f"Episode statistic keys differ: {episode_index}"
            )
    if expected is None:
        return []
    order = {name: index for index, name in enumerate(STATISTIC_ORDER)}
    return sorted(
        expected,
        key=lambda column: (
            column.rsplit("/", 1)[0],
            order.get(column.rsplit("/", 1)[1], len(order)),
            column,
        ),
    )


def _statistic_value(stats: dict[str, Any], column_name: str) -> Any:
    _, feature, statistic = column_name.split("/", 2)
    try:
        return stats[feature][statistic]
    except KeyError as exc:
        raise CurationTransformError(
            f"Episode statistic is missing: {feature}/{statistic}"
        ) from exc


def _statistic_arrow_type(column_name: str, values: list[Any]) -> pa.DataType:
    statistic = column_name.rsplit("/", 1)[1]
    inferred = pa.array(values)
    if statistic == "count":
        expected_leaf = pa.int64()
    else:
        expected_leaf = pa.float64()
    physical = inferred.type
    while pa.types.is_list(physical) or pa.types.is_large_list(physical):
        physical = physical.value_type
    if physical != expected_leaf:
        raise CurationTransformError(
            f"Episode statistic has an invalid dtype: {column_name}"
        )
    return inferred.type


def _verify_v3_metadata_rewrite(
    original_path: Path,
    candidate_path: Path,
    stats_by_episode: dict[int, dict[str, Any]],
) -> None:
    with (
        _open_parquet(original_path) as original,
        _open_parquet(candidate_path) as candidate,
    ):
        if original.metadata.num_row_groups != candidate.metadata.num_row_groups:
            raise CurationTransformError(
                "Episode metadata row groups changed during statistics write"
            )
        for row_group in range(original.metadata.num_row_groups):
            before = original.read_row_group(row_group)
            after = candidate.read_row_group(row_group)
            if len(before) != len(after):
                raise CurationTransformError(
                    "Episode metadata row count changed during statistics write"
                )
            for name in before.column_names:
                if name.startswith("stats/"):
                    continue
                if not before.column(name).equals(after.column(name)):
                    raise CurationTransformError(
                        f"Episode metadata value changed during statistics write: {name}"
                    )
            expected = _episode_table_with_stats(before, stats_by_episode)
            for name in after.column_names:
                if not name.startswith("stats/"):
                    continue
                if not _values_equal(
                    after.column(name).to_pylist(), expected.column(name).to_pylist()
                ):
                    raise CurationTransformError(
                        f"Episode statistic value changed during write: {name}"
                    )


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise CurationTransformError("Episode metadata is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise CurationTransformError("Episode metadata is invalid")
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CurationTransformError("Episode metadata is invalid") from exc
    finally:
        os.close(descriptor)
    if any(not isinstance(row, dict) for row in rows):
        raise CurationTransformError("Episode metadata is invalid")
    return rows


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise CurationTransformError("Episode statistics path is unsafe")
    if path.exists() and not path.is_file():
        raise CurationTransformError("Episode statistics path is invalid")
    existing_mode = (
        stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
        if path.exists() and not path.is_symlink()
        else 0o644
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.episode-stats-", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, existing_mode, follow_symlinks=False)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
