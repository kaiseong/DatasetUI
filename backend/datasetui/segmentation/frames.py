"""Immutable frame snapshots: content-verified single-frame reads."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
import uuid
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from typing import Any, NamedTuple

from PIL import Image

from datasetui.config import Settings
from datasetui.database import Database, RecipeRevisionMismatchError, utc_now
from datasetui.dataset_io.files import safe_dataset_root
from datasetui.segmentation.media import decode_frame, MAX_IMAGE_PIXELS
from datasetui.segmentation.paths import safe_regular_path
from datasetui.segmentation.source import load_source


_DIGEST_CACHE_LIMIT = 32


_SNAPSHOT_TABLE = """
CREATE TABLE IF NOT EXISTS segmentation_frame_snapshots (
    token TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    refs_json TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


class _FileIdentity(NamedTuple):
    path: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


_digest_cache: OrderedDict[_FileIdentity, str] = OrderedDict()


_digest_cache_lock = threading.Lock()


def create_frame_snapshot(
    database: Database,
    settings: Settings,
    dataset_id: str,
    scope: dict[str, Any],
) -> str:
    """Persist immutable video references for a previously verified dataset scope."""
    fingerprint, scope_episodes, scope_video_keys = _validate_scope(scope)
    root, source = load_source(
        database,
        settings,
        dataset_id,
        fingerprint,
        verify_content=True,
    )
    if scope_video_keys != list(source.video_keys):
        raise RecipeRevisionMismatchError(dataset_id)

    total_episodes = int(source.info.get("total_episodes", 0))
    if len(scope_episodes) != total_episodes:
        raise RecipeRevisionMismatchError(dataset_id)

    registry_fingerprint = database.get_dataset(dataset_id)["fingerprint"]
    refs: list[dict[str, Any]] = []
    video_digests: dict[str, str] = {}
    for episode_index, descriptor in enumerate(scope_episodes):
        if descriptor["episode_index"] != episode_index:
            raise RecipeRevisionMismatchError(dataset_id)
        data, metadata = source.episode(episode_index)
        frame_count = len(data)
        if descriptor["length"] != frame_count:
            raise RecipeRevisionMismatchError(dataset_id)
        for video_key in scope_video_keys:
            path, start = source.video_source(episode_index, video_key, metadata)
            if isinstance(start, bool) or not isinstance(start, int) or start < 0:
                raise RecipeRevisionMismatchError(dataset_id)
            path = safe_snapshot_path(root, path)
            relative_path = path.relative_to(root).as_posix()
            digest = video_digests.get(relative_path)
            if digest is None:
                digest = verified_file_sha256(path)
                video_digests[relative_path] = digest
            refs.append(
                {
                    "registry_fingerprint": registry_fingerprint,
                    "episode_index": episode_index,
                    "video_key": video_key,
                    "video_path": relative_path,
                    "start": start,
                    "count": frame_count,
                    "sha256": digest,
                }
            )

    # Hashing the videos may take time. Recheck the complete source immediately
    # before publishing the snapshot so it cannot bind a mixed tree revision.
    load_source(
        database,
        settings,
        dataset_id,
        fingerprint,
        verify_content=True,
    )
    token = str(uuid.uuid4())
    with database.connect() as connection:
        connection.execute(_SNAPSHOT_TABLE)
        connection.execute(
            """
            INSERT INTO segmentation_frame_snapshots(
                token, dataset_id, fingerprint, refs_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                token,
                dataset_id,
                fingerprint,
                json.dumps(refs, sort_keys=True, separators=(",", ":")),
                utc_now(),
            ),
        )
    return token


def read_snapshot_frame(
    database: Database,
    settings: Settings,
    dataset_id: str,
    token: str,
    episode_index: int,
    video_key: str,
    frame_index: int,
    *,
    verify_only: bool = False,
) -> bytes:
    """Decode one frame from a scope-bound, content-verified video reference."""
    _validate_uuid_token(token)
    if (
        isinstance(episode_index, bool)
        or not isinstance(episode_index, int)
        or episode_index < 0
        or isinstance(frame_index, bool)
        or not isinstance(frame_index, int)
        or frame_index < 0
        or not isinstance(video_key, str)
        or not video_key
    ):
        raise ValueError("Invalid frame snapshot selection")

    with database.connect() as connection:
        connection.execute(_SNAPSHOT_TABLE)
        row = connection.execute(
            """
            SELECT fingerprint, refs_json
            FROM segmentation_frame_snapshots
            WHERE token = ? AND dataset_id = ?
            """,
            (token, dataset_id),
        ).fetchone()
    if row is None:
        raise ValueError("Frame snapshot is unavailable")

    record = database.get_dataset(dataset_id)
    reference = _select_reference(
        row["refs_json"], episode_index=episode_index, video_key=video_key
    )
    fingerprint = str(reference.get("registry_fingerprint", row["fingerprint"]))
    if (
        not record["available"]
        or record["readiness"] != "ready"
        or record["fingerprint"] != fingerprint
    ):
        raise RecipeRevisionMismatchError(dataset_id)

    if frame_index >= reference["count"]:
        raise ValueError("Frame is outside the snapshotted episode")

    try:
        root = safe_dataset_root(
            settings.nas_root, record["storage_area"], record["relative_path"]
        )
    except Exception as exc:
        raise ValueError("Dataset snapshot path is unavailable") from exc
    for dependency in reference.get("metadata_dependencies", []):
        dependency_path = safe_snapshot_path(
            root, root / _validate_relative_path(dependency["path"])
        )
        if verified_file_sha256(dependency_path) != dependency["sha256"]:
            raise RecipeRevisionMismatchError(dataset_id)
    relative_path = _validate_relative_path(reference["video_path"])
    path = safe_snapshot_path(root, root / relative_path)

    actual_digest, verified_identity = _verified_file(path)
    if actual_digest != reference["sha256"]:
        raise RecipeRevisionMismatchError(dataset_id)
    if verify_only:
        return b""
    frame = decode_frame(path, reference["start"] + frame_index)
    if _file_identity(path) != verified_identity:
        raise RecipeRevisionMismatchError(dataset_id)
    if frame.shape[0] * frame.shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError("Video frame exceeds the image pixel limit")

    output = BytesIO()
    Image.fromarray(frame, mode="RGB").save(output, format="PNG", compress_level=6)
    return output.getvalue()


def verified_file_sha256(path: Path) -> str:
    """Return a stable SHA-256, cached by immutable file identity attributes."""
    return _verified_file(path)[0]


def _verified_file(path: Path) -> tuple[str, _FileIdentity]:
    identity = _file_identity(path)
    with _digest_cache_lock:
        cached = _digest_cache.get(identity)
        if cached is not None:
            if _file_identity(path) != identity:
                raise ValueError("File changed during verification")
            _digest_cache.move_to_end(identity)
            return cached, identity

        digest = hashlib.sha256()
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
            try:
                opened = _identity_from_stat(path, os.fstat(descriptor))
                if opened != identity:
                    raise ValueError("File changed during verification")
                while chunk := os.read(descriptor, 8 * 1024 * 1024):
                    digest.update(chunk)
                after_read = _identity_from_stat(path, os.fstat(descriptor))
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise ValueError("File could not be verified") from exc
        if after_read != identity or _file_identity(path) != identity:
            raise ValueError("File changed during verification")

        value = digest.hexdigest()
        _digest_cache[identity] = value
        _digest_cache.move_to_end(identity)
        while len(_digest_cache) > _DIGEST_CACHE_LIMIT:
            _digest_cache.popitem(last=False)
        return value, identity


def _file_identity(path: Path) -> _FileIdentity:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError("File is unavailable") from exc
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("File path is unsafe")
    return _identity_from_stat(path, metadata)


def _identity_from_stat(path: Path, metadata: os.stat_result) -> _FileIdentity:
    return _FileIdentity(
        str(path.absolute()),
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _validate_scope(
    scope: dict[str, Any],
) -> tuple[str, list[dict[str, int]], list[str]]:
    if not isinstance(scope, dict):
        raise ValueError("Invalid dataset scope")
    fingerprint = scope.get("fingerprint")
    episodes = scope.get("episodes")
    video_keys = scope.get("video_keys")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
        or not isinstance(episodes, list)
        or not isinstance(video_keys, list)
        or not video_keys
        or any(not isinstance(key, str) or not key for key in video_keys)
    ):
        raise ValueError("Invalid dataset scope")
    normalized_episodes: list[dict[str, int]] = []
    for descriptor in episodes:
        if not isinstance(descriptor, dict):
            raise ValueError("Invalid dataset scope")
        episode_index = descriptor.get("episode_index")
        length = descriptor.get("length")
        if (
            isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
            or episode_index < 0
            or isinstance(length, bool)
            or not isinstance(length, int)
            or length < 1
        ):
            raise ValueError("Invalid dataset scope")
        normalized_episodes.append({"episode_index": episode_index, "length": length})
    return fingerprint, normalized_episodes, list(video_keys)


def _select_reference(
    refs_json: str, *, episode_index: int, video_key: str
) -> dict[str, Any]:
    try:
        references = json.loads(refs_json)
    except (TypeError, ValueError) as exc:
        raise ValueError("Frame snapshot is invalid") from exc
    if not isinstance(references, list):
        raise ValueError("Frame snapshot is invalid")
    matches = [
        reference
        for reference in references
        if isinstance(reference, dict)
        and reference.get("episode_index") == episode_index
        and reference.get("video_key") == video_key
    ]
    if len(matches) != 1:
        raise ValueError("Frame snapshot selection is unavailable")
    reference = matches[0]
    if (
        isinstance(reference.get("start"), bool)
        or not isinstance(reference.get("start"), int)
        or reference["start"] < 0
        or isinstance(reference.get("count"), bool)
        or not isinstance(reference.get("count"), int)
        or reference["count"] < 1
        or not isinstance(reference.get("video_path"), str)
        or not isinstance(reference.get("sha256"), str)
        or len(reference["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in reference["sha256"])
    ):
        raise ValueError("Frame snapshot is invalid")
    return reference


def _validate_uuid_token(token: str) -> None:
    try:
        parsed = uuid.UUID(token)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Invalid frame snapshot token") from exc
    if parsed.version != 4 or str(parsed) != token:
        raise ValueError("Invalid frame snapshot token")


def _validate_relative_path(value: str) -> Path:
    if not isinstance(value, str) or not value or "://" in value or "\\" in value:
        raise ValueError("Frame snapshot path is invalid")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("Frame snapshot path is invalid")
    return path


SNAPSHOT_FILE_MESSAGES = {
    "invalid": "Frame snapshot path is invalid",
    "unavailable": "Frame snapshot file is unavailable",
    "unsafe_path": "Frame snapshot path is unsafe",
    "unsafe_file": "Frame snapshot file is unsafe",
}


def safe_snapshot_path(root: Path, path: Path) -> Path:
    """Snapshot reads report problems as ValueError (HTTP 409, not job errors)."""
    return safe_regular_path(
        root, path, error=ValueError, messages=SNAPSHOT_FILE_MESSAGES
    )
