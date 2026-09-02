"""Optional SARM/SRM progress sidecar reader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .common import MAX_PARQUET_BATCH_ROWS, evenly_sample_indices, finite


def read_progress(source: Path, episode_index: int, duration: float | None = None, *,
                  metrics: dict[str, int] | None = None) -> dict[str, Any]:
    path = next((source / name for name in ("sarm_progress.parquet", "srm_progress.parquet")
                 if (source / name).is_file()), None)
    if path is None:
        return {"available": False, "series": [], "source": None}
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow.names
    priority = ["progress_sparse", "progress_dense", "progress"]
    progress_columns = [name for name in schema if name.startswith("progress")]
    ordered = [key for key in priority if key in progress_columns]
    ordered.extend(sorted(set(progress_columns) - set(priority)))
    selected_progress = ordered[:1]
    if not selected_progress:
        return {"available": True, "series": [], "source": path.name}
    identity = [name for name in ("episode_index", "index", "frame_index") if name in schema]
    projected = list(dict.fromkeys([*identity, *selected_progress]))

    matching_count = 0
    count_columns = ["episode_index"] if "episode_index" in schema else []
    for batch in parquet.iter_batches(columns=count_columns or projected[:1],
                                       batch_size=MAX_PARQUET_BATCH_ROWS):
        if metrics is not None:
            metrics["peak_batch_rows"] = max(metrics.get("peak_batch_rows", 0), batch.num_rows)
        if "episode_index" in batch.schema.names:
            matching_count += sum(int(value.as_py()) == episode_index
                                  for value in batch.column("episode_index"))
        else:
            matching_count += batch.num_rows
    wanted = evenly_sample_indices(matching_count, 4000)
    rows: list[dict[str, Any]] = []
    match_index = 0
    wanted_cursor = 0
    for batch in parquet.iter_batches(columns=projected, batch_size=MAX_PARQUET_BATCH_ROWS):
        episode_column = (batch.column("episode_index")
                          if "episode_index" in batch.schema.names else None)
        take: list[int] = []
        for row_index in range(batch.num_rows):
            if episode_column is not None and int(episode_column[row_index].as_py()) != episode_index:
                continue
            if wanted_cursor < len(wanted) and match_index == wanted[wanted_cursor]:
                take.append(row_index)
                wanted_cursor += 1
            match_index += 1
        if take:
            rows.extend(batch.take(take).to_pylist())
            if metrics is not None:
                metrics["retained_rows"] = metrics.get("retained_rows", 0) + len(take)
        if wanted_cursor == len(wanted):
            break
    order_key = next((key for key in ("index", "frame_index") if rows and key in rows[0]), None)
    if order_key:
        rows.sort(key=lambda row: row.get(order_key, 0))
    columns = selected_progress
    max_time = float(duration) if duration is not None else float(max(len(rows) - 1, 0))
    result = []
    for key in columns:
        label = "progress | " + key.removeprefix("progress_") if key.startswith("progress") else key
        points = [{"timestamp": (index / max(len(rows)-1, 1)) * max_time,
                   "value": float(row[key])}
                  for index, row in enumerate(rows) if row.get(key) is not None]
        result.append({"key": key, "name": label, "points": points})
    return finite({"available": True, "source": path.name, "series": result})
