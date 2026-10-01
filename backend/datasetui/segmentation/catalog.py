"""Metadata-only dataset catalog and per-selection frame references."""

from __future__ import annotations

import hashlib
import json
import uuid

from datasetui.database import RecipeRevisionMismatchError, utc_now
from datasetui.segmentation.frames import (
    _SNAPSHOT_TABLE,
    _safe_regular_path,
    verified_file_sha256,
)
from datasetui.transforms import _DatasetSource, _read_regular_bytes, _safe_dataset_root


def _metadata(database, settings, dataset_id):
    record = database.get_dataset(dataset_id)
    if not record["available"] or record["readiness"] != "ready":
        raise RecipeRevisionMismatchError(dataset_id)
    root = _safe_dataset_root(
        settings.nas_root, record["storage_area"], record["relative_path"]
    )
    path = _safe_regular_path(root, root / "meta/info.json")
    raw = _read_regular_bytes(path, max_bytes=2 * 1024 * 1024)
    info = json.loads(raw)
    return record, root, info, hashlib.sha256(raw).hexdigest()


def dataset_catalog(database, settings, dataset_id):
    _, _, info, revision = _metadata(database, settings, dataset_id)
    keys = [
        key
        for key, value in info.get("features", {}).items()
        if isinstance(value, dict) and value.get("dtype") == "video"
    ]
    if any(key in {"", ".", ".."} or "/" in key or "\\" in key for key in keys):
        raise ValueError("Unsafe video feature name")
    return {
        "metadata_revision": revision,
        "total_episodes": int(info["total_episodes"]),
        "video_keys": keys,
        "fps": float(info["fps"]),
    }


def selected_scope(database, settings, dataset_id, episode_index, video_key):
    record, root, info, revision = _metadata(database, settings, dataset_id)
    catalog = dataset_catalog(database, settings, dataset_id)
    if (
        not 0 <= episode_index < catalog["total_episodes"]
        or video_key not in catalog["video_keys"]
    ):
        raise ValueError("Selection is outside dataset")
    dependencies = [{"path": "meta/info.json", "sha256": revision}]
    metadata = None
    if info["codebase_version"] == "v3.0":
        import pyarrow.parquet as pq

        directory = root / "meta/episodes"
        # Read only episode metadata columns, never action/state parquet.
        for candidate in sorted(directory.glob("chunk-*/*.parquet")):
            path = _safe_regular_path(root, candidate)
            columns = [
                "episode_index",
                "length",
                f"videos/{video_key}/chunk_index",
                f"videos/{video_key}/file_index",
                f"videos/{video_key}/from_timestamp",
            ]
            digest = verified_file_sha256(path)
            table = pq.read_table(
                path, columns=columns, filters=[("episode_index", "=", episode_index)]
            )
            if table.num_rows:
                if table.num_rows != 1:
                    raise ValueError("Duplicate episode metadata")
                metadata = table.to_pylist()[0]
                dependencies.append(
                    {"path": path.relative_to(root).as_posix(), "sha256": digest}
                )
                break
    else:
        path = _safe_regular_path(root, root / "meta/episodes.jsonl")
        digest = verified_file_sha256(path)
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["episode_index"] == episode_index:
                    metadata = row
                    break
        dependencies.append({"path": "meta/episodes.jsonl", "sha256": digest})
    if metadata is None or int(metadata["length"]) < 1:
        raise ValueError("Episode metadata is unavailable")
    # Reuse the path resolver only; never run its eager constructor or episode().
    resolver = object.__new__(_DatasetSource)
    resolver.root, resolver.info = root, info
    resolver.version, resolver.fps = info["codebase_version"], float(info["fps"])
    path, start = resolver.video_source(episode_index, video_key, metadata)
    path = _safe_regular_path(root, path)
    reference = {
        "registry_fingerprint": record["fingerprint"],
        "episode_index": episode_index,
        "video_key": video_key,
        "video_path": path.relative_to(root).as_posix(),
        "start": start,
        "count": int(metadata["length"]),
        "sha256": verified_file_sha256(path),
        "metadata_dependencies": dependencies,
    }
    for dependency in dependencies:
        if (
            verified_file_sha256(_safe_regular_path(root, root / dependency["path"]))
            != dependency["sha256"]
        ):
            raise RecipeRevisionMismatchError(dataset_id)
    token = str(uuid.uuid4())
    with database.connect() as connection:
        connection.execute(_SNAPSHOT_TABLE)
        connection.execute(
            "INSERT INTO segmentation_frame_snapshots VALUES (?, ?, ?, ?, ?)",
            (token, dataset_id, "", json.dumps([reference]), utc_now()),
        )
    return {
        "metadata_revision": revision,
        "frame_token": token,
        "episode_index": episode_index,
        "video_key": video_key,
        "length": reference["count"],
        "fps": resolver.fps,
    }
