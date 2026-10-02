"""DatasetSource: a validated view of one LeRobot dataset on disk."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from datasetui.dataset_io.files import read_json_lines, require_regular_file, safe_child
from datasetui.dataset_io.tables import (
    read_parquet,
    safe_parquet_files,
    task_rows_from_frame,
)
from datasetui.transform_errors import CurationTransformError


class DatasetSource:
    def __init__(self, root: Path, info: dict[str, Any]):
        self.root = root
        self.info = info
        self.version = str(info["codebase_version"])
        self.fps = float(info["fps"])
        self.video_keys = [
            key
            for key, value in info.get("features", {}).items()
            if isinstance(value, dict) and value.get("dtype") == "video"
        ]
        if any(
            key in {"", ".", ".."} or "/" in key or "\\" in key
            for key in self.video_keys
        ):
            raise CurationTransformError(
                "Dataset contains an unsafe video feature name"
            )
        self.tasks = self._load_tasks()
        self.episode_metadata = self._load_episode_metadata()
        self._v3_data_cache: tuple[Path, pd.DataFrame] | None = None

    def _load_tasks(self) -> dict[int, str]:
        if self.version == "v3.0":
            path = self.root / "meta" / "tasks.parquet"
            if not path.is_file():
                return {}
            require_regular_file(path)
            frame = read_parquet(path)
            rows = task_rows_from_frame(frame)
            return {
                int(row["task_index"]): str(row.get("task", row.get("name", "")))
                for row in rows
            }
        path = self.root / "meta" / "tasks.jsonl"
        if not path.is_file():
            return {}
        require_regular_file(path)
        return {
            int(row["task_index"]): str(row.get("task", row.get("name", "")))
            for row in read_json_lines(path)
        }

    def _load_episode_metadata(self) -> dict[int, dict[str, Any]]:
        if self.version != "v3.0":
            path = self.root / "meta" / "episodes.jsonl"
            return {
                int(row["episode_index"]): row
                for row in (read_json_lines(path) if path.is_file() else [])
            }
        rows: list[dict[str, Any]] = []
        for path in safe_parquet_files(self.root / "meta" / "episodes"):
            rows.extend(read_parquet(path).to_dict("records"))
        return {int(row["episode_index"]): row for row in rows}

    def episode(self, episode_index: int) -> tuple[pd.DataFrame, dict[str, Any]]:
        metadata = self.episode_metadata.get(episode_index, {})
        if self.version == "v3.0":
            chunk = int(metadata.get("data/chunk_index", 0))
            file_index = int(metadata.get("data/file_index", 0))
            path = self.root / f"data/chunk-{chunk:03d}/file-{file_index:03d}.parquet"
            require_regular_file(path)
            if self._v3_data_cache is None or self._v3_data_cache[0] != path:
                self._v3_data_cache = (path, read_parquet(path))
            data = self._v3_data_cache[1]
            data = data[data["episode_index"] == episode_index].copy()
        else:
            chunk_size = int(self.info.get("chunks_size", 1000))
            template = self.info.get(
                "data_path",
                "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            )
            relative = template.format(
                episode_chunk=episode_index // chunk_size,
                episode_index=episode_index,
            )
            path = safe_child(self.root, relative)
            require_regular_file(path)
            data = read_parquet(path)
        if data.empty:
            raise CurationTransformError(f"Episode {episode_index} has no frames")
        return data.reset_index(drop=True), metadata

    def video_source(
        self, episode_index: int, video_key: str, metadata: dict[str, Any]
    ) -> tuple[Path, int]:
        if self.version == "v3.0":
            chunk = int(metadata.get(f"videos/{video_key}/chunk_index", 0))
            file_index = int(metadata.get(f"videos/{video_key}/file_index", 0))
            start = int(
                round(
                    float(metadata.get(f"videos/{video_key}/from_timestamp", 0))
                    * self.fps
                )
            )
            return (
                safe_child(
                    self.root,
                    f"videos/{video_key}/chunk-{chunk:03d}/file-{file_index:03d}.mp4",
                ),
                start,
            )
        chunk_size = int(self.info.get("chunks_size", 1000))
        template = self.info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
        relative = template.format(
            episode_chunk=episode_index // chunk_size,
            episode_index=episode_index,
            video_key=video_key,
        )
        return safe_child(self.root, relative), 0
