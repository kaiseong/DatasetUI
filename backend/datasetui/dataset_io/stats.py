"""Writes meta/stats.json for a newly written dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from datasetui.dataset_io.files import write_json
from datasetui.job_progress import ProgressCallback, report_progress



def write_stats(
    path: Path,
    episodes: list[pd.DataFrame],
    *,
    on_progress: ProgressCallback | None = None,
) -> None:
    root = path.parent.parent
    if (root / "meta/info.json").is_file():
        from datasetui.statistics.output import write_output_statistics

        write_output_statistics(root, on_progress=on_progress)
        return
    # Array-only diagnostic compatibility; production outputs always have info.
    combined = pd.concat(episodes, ignore_index=True)
    stats: dict[str, Any] = {}
    columns = list(combined.columns)
    report_progress(
        on_progress,
        stage="statistics",
        completed=0,
        total=len(columns),
        unit="items",
        current_item="수치 특성",
        force=True,
    )
    for column_index, name in enumerate(columns, start=1):
        try:
            matrix = np.stack(
                [
                    np.asarray(value, dtype=np.float64).reshape(-1)
                    for value in combined[name]
                ]
            )
        except (TypeError, ValueError):
            report_progress(
                on_progress,
                stage="statistics",
                completed=column_index,
                total=len(columns),
                unit="items",
                current_item=name,
            )
            continue
        if matrix.size == 0 or not np.isfinite(matrix).all():
            report_progress(
                on_progress,
                stage="statistics",
                completed=column_index,
                total=len(columns),
                unit="items",
                current_item=name,
            )
            continue
        stats[name] = {
            "min": np.min(matrix, axis=0).tolist(),
            "max": np.max(matrix, axis=0).tolist(),
            "mean": np.mean(matrix, axis=0).tolist(),
            "std": np.std(matrix, axis=0).tolist(),
            "q01": np.quantile(matrix, 0.01, axis=0).tolist(),
            "q10": np.quantile(matrix, 0.10, axis=0).tolist(),
            "q50": np.quantile(matrix, 0.50, axis=0).tolist(),
            "q90": np.quantile(matrix, 0.90, axis=0).tolist(),
            "q99": np.quantile(matrix, 0.99, axis=0).tolist(),
            "count": [len(matrix)],
        }
        report_progress(
            on_progress,
            stage="statistics",
            completed=column_index,
            total=len(columns),
            unit="items",
            current_item=name,
        )
    write_json(path, stats)
