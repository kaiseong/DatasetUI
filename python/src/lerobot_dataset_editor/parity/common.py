"""Small, dependency-free helpers shared by parity services."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

MAX_PARQUET_BATCH_ROWS = 8192


def load_info(source: Path) -> dict[str, Any]:
    value = json.loads((source / "meta" / "info.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("meta/info.json must contain an object")
    return value


def data_paths(source: Path) -> list[Path]:
    directory = source / "data"
    return sorted(directory.rglob("*.parquet")) if directory.is_dir() else []


def read_data(source: Path, columns: Iterable[str] | None = None) -> pa.Table:
    paths = data_paths(source)
    if not paths:
        return pa.table({})
    requested = list(columns) if columns is not None else None
    tables: list[pa.Table] = []
    for path in paths:
        schema_names = pq.read_schema(path).names
        selected = [name for name in requested if name in schema_names] if requested else None
        tables.append(pq.read_table(path, columns=selected))
    return pa.concat_tables(tables, promote_options="default") if len(tables) > 1 else tables[0]


def episode_records(source: Path, extra_columns: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Read lightweight episode metadata without touching frame parquet files."""
    legacy = source / "meta" / "episodes.jsonl"
    if legacy.is_file():
        rows = [json.loads(line) for line in legacy.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        return sorted((row for row in rows if isinstance(row, dict)),
                      key=lambda row: int(row.get("episode_index", -1)))
    rows: list[dict[str, Any]] = []
    required = ("episode_index", "length", "data/chunk_index", "data/file_index",
                "dataset_from_index", "dataset_to_index", *extra_columns)
    for path in sorted((source / "meta" / "episodes").rglob("*.parquet")):
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow.names
        columns = [name for name in dict.fromkeys(required) if name in schema]
        if "episode_index" in columns:
            rows.extend(parquet.read(columns=columns).to_pylist())
    if rows:
        return sorted(rows, key=lambda row: int(row.get("episode_index", -1)))
    # Some valid v2 exports omit episodes.jsonl.  Their data template is still
    # episode-addressable, so build descriptors from info plus Parquet metadata
    # (which does not decode or materialize frame columns).
    info = load_info(source)
    synthetic: list[dict[str, Any]] = []
    for episode in range(max(0, int(info.get("total_episodes", 0)))):
        row: dict[str, Any] = {"episode_index": episode}
        path = _format_data_path(source, info, row)
        if path is not None:
            row["length"] = pq.ParquetFile(path).metadata.num_rows
            synthetic.append(row)
    return synthetic


def _format_data_path(source: Path, info: dict[str, Any], row: dict[str, Any]) -> Path | None:
    template = info.get("data_path")
    if not isinstance(template, str):
        return None
    episode = int(row["episode_index"])
    chunk = int(row.get("data/chunk_index", row.get("episode_chunk", episode // 1000)))
    file_index = int(row.get("data/file_index", episode))
    try:
        relative = template.format(episode_index=episode, episode_chunk=episode // 1000,
                                   chunk_index=chunk, file_index=file_index)
    except (KeyError, ValueError):
        return None
    path = source / relative
    return path if path.is_file() else None


def read_sampled_episodes(source: Path, records: list[dict[str, Any]],
                          columns: Iterable[str], maximum_frames: int, *,
                          metrics: dict[str, int] | None = None,
                          all_records: list[dict[str, Any]] | None = None) -> pa.Table:
    """Read endpoint-inclusive frame samples from only the selected episodes.

    v3's episode metadata maps each episode to a shared data file and a global
    row interval.  We translate the desired local frame indices into file-local
    offsets and read only intersecting Parquet row groups.  v2's one-file-per-
    episode layout follows the same path with local offsets.  Conversion to
    Python objects therefore happens only after Arrow ``take`` has reduced the
    data to at most ``len(records) * maximum_frames`` rows.
    """
    if not records:
        return pa.table({})
    info = load_info(source)
    requested = list(dict.fromkeys(columns))
    file_starts: dict[Path, int] = {}
    for row in all_records if all_records is not None else episode_records(source):
        path = _format_data_path(source, info, row)
        if path is not None and "dataset_from_index" in row:
            start = int(row["dataset_from_index"])
            file_starts[path] = min(file_starts.get(path, start), start)
    grouped: dict[Path, list[tuple[dict[str, Any], list[int]]]] = {}
    for row in records:
        path = _format_data_path(source, info, row)
        if path is None:
            continue
        length = max(0, int(row.get("length", 0)))
        indices = evenly_sample_indices(length, maximum_frames)
        grouped.setdefault(path, []).append((row, indices))

    sampled: list[pa.Table] = []
    for path, episodes in grouped.items():
        parquet = pq.ParquetFile(path)
        available = parquet.schema_arrow.names
        selected_columns = [name for name in requested if name in available]
        if not selected_columns:
            continue
        # v3 indices are global dataset offsets.  A shared file starts at the
        # smallest metadata offset assigned to that file; v2 offsets are local.
        file_start = file_starts.get(path, 0)
        wanted: list[int] = []
        for row, local_indices in episodes:
            start = int(row.get("dataset_from_index", file_start)) - file_start
            wanted.extend(start + index for index in local_indices)
        wanted = sorted(set(index for index in wanted if 0 <= index < parquet.metadata.num_rows))
        if not wanted:
            continue
        row_group_start = 0
        cursor = 0
        for group_index in range(parquet.metadata.num_row_groups):
            group_rows = parquet.metadata.row_group(group_index).num_rows
            row_group_end = row_group_start + group_rows
            while cursor < len(wanted) and wanted[cursor] < row_group_start:
                cursor += 1
            end_cursor = cursor
            while end_cursor < len(wanted) and wanted[end_cursor] < row_group_end:
                end_cursor += 1
            if end_cursor > cursor:
                group_cursor = cursor
                batch_start = row_group_start
                for batch in parquet.iter_batches(row_groups=[group_index],
                                                   batch_size=MAX_PARQUET_BATCH_ROWS,
                                                   columns=selected_columns):
                    batch_end = batch_start + batch.num_rows
                    batch_cursor = group_cursor
                    while batch_cursor < end_cursor and wanted[batch_cursor] < batch_end:
                        batch_cursor += 1
                    if batch_cursor > group_cursor:
                        offsets = pa.array([index - batch_start
                                            for index in wanted[group_cursor:batch_cursor]],
                                           type=pa.int64())
                        sampled.append(pa.Table.from_batches([batch.take(offsets)]))
                        if metrics is not None:
                            metrics["retained_rows"] = metrics.get("retained_rows", 0) + len(offsets)
                    if metrics is not None:
                        metrics["peak_batch_rows"] = max(metrics.get("peak_batch_rows", 0),
                                                         batch.num_rows)
                    group_cursor = batch_cursor
                    batch_start = batch_end
                    if group_cursor == end_cursor:
                        break
            cursor = end_cursor
            row_group_start = row_group_end
            if cursor == len(wanted):
                break
    if not sampled:
        return pa.table({})
    return pa.concat_tables(sampled, promote_options="default") if len(sampled) > 1 else sampled[0]


def episode_table(source: Path, episode_index: int, columns: Iterable[str] | None = None, *,
                  records: list[dict[str, Any]] | None = None) -> pa.Table:
    records = records if records is not None else episode_records(source)
    record = next((row for row in records if int(row.get("episode_index", -1)) == episode_index), None)
    if record is None:
        return pa.table({})
    requested = list(columns) if columns is not None else ["episode_index"]
    return read_sampled_episodes(source, [record], requested, int(record.get("length", 0)),
                                 all_records=records)


def finite(value: Any) -> Any:
    """Recursively turn Arrow/numpy-ish values into strict JSON values."""
    if isinstance(value, dict):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return finite(value.item())
    return str(value)


def evenly_sample(values: list[Any], maximum: int) -> list[Any]:
    if len(values) <= maximum:
        return values
    if maximum <= 1:
        return values[:1]
    last = len(values) - 1
    # JavaScript Math.round is half-up for the non-negative indices used by
    # the pinned Space; Python round uses bankers rounding and would select a
    # different middle frame for cases such as 6 -> 3 samples.
    return [values[math.floor(index * last / (maximum - 1) + 0.5)]
            for index in range(maximum)]


def evenly_sample_indices(length: int, maximum: int) -> list[int]:
    """Return pinned endpoint-inclusive indices without allocating ``range(length)``."""
    if length <= 0 or maximum <= 0:
        return []
    if length <= maximum:
        return list(range(length))
    if maximum == 1:
        return [0]
    last = length - 1
    return [math.floor(index * last / (maximum - 1) + 0.5)
            for index in range(maximum)]


def resolve_under(root: Path, relative: str | Path) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("path escapes dataset root")
    return candidate


def dimension_names(info: dict[str, Any], key: str, width: int) -> list[str]:
    spec = info.get("features", {}).get(key, {})
    names = spec.get("names") if isinstance(spec, dict) else None
    if isinstance(names, list) and len(names) == width:
        return [str(value) for value in names]
    return [str(index) for index in range(width)]
