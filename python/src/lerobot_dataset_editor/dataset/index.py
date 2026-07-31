"""Incremental metadata-only SQLite index under the XDG cache."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
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
from .loader import load_document

FFPROBE_COMMAND = [
    "ffprobe",
    "-v", "error",
    "-print_format", "json",
    "-show_format",
    "-show_streams",
]
_VIDEO_SUFFIXES = frozenset({".mp4", ".mkv", ".mov", ".avi", ".webm"})
_INDEX_SUFFIXES = frozenset({".json", ".jsonl", ".parquet", *_VIDEO_SUFFIXES})
_SCHEMA_VERSION = 2

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS index_meta (
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL UNIQUE,
    source_fingerprint TEXT NOT NULL,
    detected_version TEXT NOT NULL,
    document_json TEXT NOT NULL,
    indexed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS parquet_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    relative_path TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    num_rows INTEGER NOT NULL,
    num_row_groups INTEGER NOT NULL,
    schema_columns TEXT NOT NULL,
    UNIQUE(dataset_id, relative_path)
);
CREATE TABLE IF NOT EXISTS video_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    relative_path TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    duration_sec REAL,
    codec TEXT,
    width INTEGER,
    height INTEGER,
    UNIQUE(dataset_id, relative_path)
);
"""


def default_index_path() -> Path:
    cache_home = os.environ.get("XDG_CACHE_HOME")
    root = Path(cache_home) if cache_home else Path.home() / ".cache"
    return root / "lerobot-dataset-editor" / "index.sqlite"


def _fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}:{stat.st_ino}"


def _index_files(source: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for relative_root in ("meta", "data", "videos"):
        root = source / relative_root
        if not root.is_dir():
            continue
        files.extend(
            candidate
            for candidate in root.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in _INDEX_SUFFIXES
        )
    provenance = source / "PROVENANCE.md"
    if provenance.is_file():
        files.append(provenance)
    return tuple(sorted(set(files), key=lambda path: path.relative_to(source).as_posix()))


def _source_fingerprint(source: Path) -> str:
    digest = hashlib.sha256()
    for path in _index_files(source):
        digest.update(path.relative_to(source).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_fingerprint(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _dataset_id(source_path: Path) -> str:
    return hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()[:32]


def _serialize_document(doc: DatasetDocument) -> str:
    value: dict[str, Any] = {
        "source_path": doc.source_path,
        "version": {
            "family": doc.version.family,
            "version": doc.version.version,
            "raw": doc.version.raw,
        },
        "features": [
            {
                "name": feature.name,
                "dtype": feature.dtype,
                "shape": list(feature.shape),
                "names": list(feature.names) if feature.names else None,
            }
            for feature in doc.features
        ],
        "episodes": [
            {
                "index": episode.index,
                "length": episode.length,
                "chunk_index": episode.chunk_index,
                "file_index": episode.file_index,
                "tasks": list(episode.tasks),
                "frame_start": episode.frame_start,
                "frame_end": episode.frame_end,
            }
            for episode in doc.episodes
        ],
        "total_frames": doc.total_frames,
        "fps": doc.fps,
        "media": [
            {
                "video_key": media.video_key,
                "path_template": media.path_template,
                "episodes_with_video": list(media.episodes_with_video),
            }
            for media in doc.media
        ],
        "tasks": list(doc.tasks),
        "has_annotations": doc.has_annotations,
        "annotation_styles": list(doc.annotation_styles) if doc.annotation_styles else None,
        "provenance": doc.provenance,
        "validation": {
            "valid": doc.validation.valid,
            "errors": list(doc.validation.errors),
            "warnings": list(doc.validation.warnings),
        },
        "indexed_at": doc.indexed_at,
    }
    return json.dumps(value, separators=(",", ":"))


def _deserialize_document(data: str) -> DatasetDocument:
    value = json.loads(data)
    return DatasetDocument(
        source_path=value["source_path"],
        version=DatasetVersion(**value["version"]),
        features=tuple(
            FeatureSpec(
                name=item["name"],
                dtype=item["dtype"],
                shape=tuple(item["shape"]),
                names=tuple(item["names"]) if item["names"] else None,
            )
            for item in value["features"]
        ),
        episodes=tuple(
            EpisodeRef(
                index=item["index"],
                length=item["length"],
                chunk_index=item["chunk_index"],
                file_index=item["file_index"],
                tasks=tuple(item["tasks"]),
                frame_start=item.get("frame_start", 0),
                frame_end=item.get("frame_end", item.get("length", 0)),
            )
            for item in value["episodes"]
        ),
        total_frames=value["total_frames"],
        fps=value["fps"],
        media=tuple(
            MediaRef(
                video_key=item["video_key"],
                path_template=item["path_template"],
                episodes_with_video=tuple(item["episodes_with_video"]),
            )
            for item in value["media"]
        ),
        tasks=tuple(value["tasks"]),
        has_annotations=value["has_annotations"],
        annotation_styles=tuple(value["annotation_styles"]) if value["annotation_styles"] else None,
        provenance=value["provenance"],
        validation=ValidationResult(
            valid=value["validation"]["valid"],
            errors=tuple(value["validation"]["errors"]),
            warnings=tuple(value["validation"]["warnings"]),
        ),
        indexed_at=value["indexed_at"],
    )


def _safe_probe_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for name in ("HOME", "LANG", "LC_ALL", "PATH"):
        if (value := os.environ.get(name)) is not None:
            env[name] = value
    return env


class DatasetIndex:
    """Cache version-neutral documents and file metadata without decoding data."""

    def __init__(
        self,
        db_path: Path | None = None,
        *,
        ffprobe_path: str | None = None,
    ) -> None:
        self._db_path = db_path or default_index_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self._db_path.parent, 0o700)
        self._ffprobe_path = ffprobe_path if ffprobe_path is not None else shutil.which("ffprobe")
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._migrate()

    def _migrate(self) -> None:
        self._conn.executescript(_CREATE_TABLES)
        dataset_columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(datasets)")
        }
        if "source_fingerprint" not in dataset_columns:
            # This database is a disposable derived cache. Rebuild the pre-release
            # info_fingerprint layout without touching any source dataset.
            self._conn.executescript(
                """
                DROP TABLE IF EXISTS video_files;
                DROP TABLE IF EXISTS parquet_files;
                DROP TABLE IF EXISTS datasets;
                DROP TABLE IF EXISTS index_meta;
                """
            )
            self._conn.executescript(_CREATE_TABLES)

        row = self._conn.execute("SELECT schema_version FROM index_meta").fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO index_meta (schema_version, created_at) VALUES (?, ?)",
                (_SCHEMA_VERSION, datetime.now(timezone.utc).isoformat()),
            )
        elif row[0] > _SCHEMA_VERSION:
            raise RuntimeError(
                f"Dataset index schema {row[0]} is newer than {_SCHEMA_VERSION}"
            )
        elif row[0] < _SCHEMA_VERSION:
            self._conn.execute("DELETE FROM video_files")
            self._conn.execute("DELETE FROM parquet_files")
            self._conn.execute("DELETE FROM datasets")
            self._conn.execute(
                "UPDATE index_meta SET schema_version = ?, created_at = ?",
                (_SCHEMA_VERSION, datetime.now(timezone.utc).isoformat()),
            )
        self._conn.commit()

    def get_or_build(self, dataset_path: str | Path) -> DatasetDocument:
        source = Path(dataset_path).resolve()
        info_path = source / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Dataset metadata not found: {info_path}")
        current_fingerprint = _source_fingerprint(source)
        dataset_id = _dataset_id(source)
        row = self._conn.execute(
            "SELECT source_fingerprint, document_json FROM datasets WHERE id = ?",
            (dataset_id,),
        ).fetchone()
        if row is not None and row[0] == current_fingerprint:
            return _deserialize_document(row[1])

        document = load_document(source)
        self._conn.execute(
            """INSERT OR REPLACE INTO datasets
               (id, source_path, source_fingerprint, detected_version, document_json, indexed_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                dataset_id,
                str(source),
                current_fingerprint,
                document.version.version,
                _serialize_document(document),
                document.indexed_at,
            ),
        )
        self._store_parquet_metadata(dataset_id, source)
        self._store_video_metadata(dataset_id, source)
        self._conn.commit()
        return document

    def _store_parquet_metadata(self, dataset_id: str, source: Path) -> None:
        self._conn.execute("DELETE FROM parquet_files WHERE dataset_id = ?", (dataset_id,))
        for path in (item for item in _index_files(source) if item.suffix.lower() == ".parquet"):
            try:
                metadata = pq.read_metadata(path)
                schema = pq.read_schema(path)
            except Exception:
                continue
            self._conn.execute(
                """INSERT INTO parquet_files
                   (dataset_id, relative_path, fingerprint, num_rows, num_row_groups, schema_columns)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    dataset_id,
                    path.relative_to(source).as_posix(),
                    _fingerprint(path),
                    metadata.num_rows,
                    metadata.num_row_groups,
                    json.dumps(schema.names),
                ),
            )

    def _store_video_metadata(self, dataset_id: str, source: Path) -> None:
        self._conn.execute("DELETE FROM video_files WHERE dataset_id = ?", (dataset_id,))
        for path in (item for item in _index_files(source) if item.suffix.lower() in _VIDEO_SUFFIXES):
            duration, codec, width, height = self._probe_video(path)
            self._conn.execute(
                """INSERT INTO video_files
                   (dataset_id, relative_path, fingerprint, duration_sec, codec, width, height)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    dataset_id,
                    path.relative_to(source).as_posix(),
                    _fingerprint(path),
                    duration,
                    codec,
                    width,
                    height,
                ),
            )

    def _probe_video(self, path: Path) -> tuple[float | None, str | None, int | None, int | None]:
        if not self._ffprobe_path:
            return None, None, None, None
        command = [self._ffprobe_path, *FFPROBE_COMMAND[1:], str(path)]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=10,
                env=_safe_probe_env(),
                check=False,
            )
            if result.returncode != 0:
                return None, None, None, None
            payload = json.loads(result.stdout)
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            return None, None, None, None
        streams = payload.get("streams") if isinstance(payload, dict) else None
        video_stream = next(
            (
                stream
                for stream in streams or []
                if isinstance(stream, dict) and stream.get("codec_type") == "video"
            ),
            {},
        )
        format_info = payload.get("format", {}) if isinstance(payload, dict) else {}
        raw_duration = format_info.get("duration") or video_stream.get("duration")
        try:
            duration = float(raw_duration) if raw_duration is not None else None
        except (TypeError, ValueError):
            duration = None
        return (
            duration,
            video_stream.get("codec_name"),
            video_stream.get("width"),
            video_stream.get("height"),
        )

    def close(self) -> None:
        self._conn.close()
