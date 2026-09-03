from __future__ import annotations

import fcntl
import errno
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from httpx import HTTPError
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import (
    HfHubHTTPError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from datasetui.config import Settings
from datasetui.database import Database, ImmutableRevisionConflictError
from datasetui.datasets import inspect_dataset
from datasetui.hf_errors import (
    HuggingFaceDatasetNotFoundError,
    HuggingFaceImportConflictError,
    HuggingFaceImportTooLargeError,
    HuggingFaceImportValidationError,
    HuggingFaceRevisionNotFoundError,
    HuggingFaceUnavailableError,
)


HF_NAMESPACE = "rainbowrobotics"
COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9_])?$")
logger = logging.getLogger("datasetui.huggingface")


def validate_dataset_name(value: str) -> str:
    name = value.strip()
    if not DATASET_NAME_PATTERN.fullmatch(name) or ".." in name:
        raise ValueError("invalid Hugging Face dataset name")
    return name


def validate_revision(value: str) -> str:
    revision = value.strip()
    if (
        not revision
        or len(revision) > 200
        or revision.startswith("/")
        or "\\" in revision
        or any(part in {"", ".", ".."} for part in revision.split("/"))
        or any(ord(character) < 32 or ord(character) == 127 for character in revision)
    ):
        raise ValueError("invalid Hugging Face revision")
    return revision


def validate_commit_sha(value: str) -> str:
    commit_sha = value.strip().lower()
    if not COMMIT_SHA_PATTERN.fullmatch(commit_sha):
        raise ValueError("invalid Hugging Face commit SHA")
    return commit_sha


class HuggingFaceGateway:
    def __init__(self, token: str | None = None):
        self.token = token
        self.api = HfApi(token=token)

    def list_datasets(self, *, query: str | None, limit: int) -> list[dict[str, Any]]:
        try:
            results = self.api.list_datasets(
                author=HF_NAMESPACE,
                search=query or None,
                sort="last_modified",
                limit=limit,
                token=self.token,
            )
            return [self._dataset_summary(item) for item in results]
        except (HfHubHTTPError, HTTPError, OfflineModeIsEnabled) as exc:
            raise HuggingFaceUnavailableError(
                "Hugging Face datasets are temporarily unavailable"
            ) from exc

    def list_revisions(self, dataset_name: str) -> list[dict[str, str]]:
        repo_id = _repo_id(dataset_name)
        try:
            refs = self.api.list_repo_refs(
                repo_id,
                repo_type="dataset",
                token=self.token,
            )
        except RepositoryNotFoundError as exc:
            raise HuggingFaceDatasetNotFoundError("Dataset not found") from exc
        except (HfHubHTTPError, HTTPError, OfflineModeIsEnabled) as exc:
            raise HuggingFaceUnavailableError(
                "Hugging Face revisions are temporarily unavailable"
            ) from exc

        revisions = [
            {
                "name": reference.name,
                "kind": kind,
                "commit_sha": validate_commit_sha(reference.target_commit),
            }
            for kind, collection in (("branch", refs.branches), ("tag", refs.tags))
            for reference in collection
        ]
        revisions.sort(
            key=lambda item: (
                item["name"] != "main",
                item["kind"] != "branch",
                item["name"].casefold(),
            )
        )
        return revisions[:200]

    def resolve_revision(self, dataset_name: str, revision: str) -> dict[str, Any]:
        repo_id = _repo_id(dataset_name)
        requested_revision = validate_revision(revision)
        try:
            info = self.api.dataset_info(
                repo_id,
                revision=requested_revision,
                files_metadata=True,
                token=self.token,
            )
        except RepositoryNotFoundError as exc:
            raise HuggingFaceDatasetNotFoundError("Dataset not found") from exc
        except RevisionNotFoundError as exc:
            raise HuggingFaceRevisionNotFoundError("Revision not found") from exc
        except (HfHubHTTPError, HTTPError, OfflineModeIsEnabled) as exc:
            raise HuggingFaceUnavailableError(
                "Hugging Face dataset details are temporarily unavailable"
            ) from exc

        siblings = list(info.siblings or [])
        known_sizes = [item.size for item in siblings if item.size is not None]
        return {
            "repo_id": repo_id,
            "requested_revision": requested_revision,
            "commit_sha": validate_commit_sha(info.sha),
            "file_count": len(siblings),
            "total_bytes": sum(known_sizes)
            if len(known_sizes) == len(siblings)
            else None,
        }

    def download_snapshot(
        self,
        *,
        repo_id: str,
        commit_sha: str,
        local_dir: Path,
    ) -> Path:
        try:
            downloaded = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=validate_commit_sha(commit_sha),
                local_dir=local_dir,
                token=self.token,
                max_workers=8,
            )
        except RepositoryNotFoundError as exc:
            raise HuggingFaceDatasetNotFoundError("Dataset not found") from exc
        except RevisionNotFoundError as exc:
            raise HuggingFaceRevisionNotFoundError("Revision not found") from exc
        except (HfHubHTTPError, HTTPError, OfflineModeIsEnabled) as exc:
            raise HuggingFaceUnavailableError(
                "Hugging Face download is temporarily unavailable"
            ) from exc
        return Path(downloaded)

    @staticmethod
    def _dataset_summary(item: Any) -> dict[str, Any]:
        repo_id = str(item.id)
        namespace, separator, name = repo_id.partition("/")
        if not separator or namespace.casefold() != HF_NAMESPACE:
            raise HuggingFaceUnavailableError(
                "Hugging Face returned an unexpected dataset namespace"
            )
        last_modified = item.last_modified
        if isinstance(last_modified, datetime):
            last_modified_text = (
                last_modified.astimezone(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
        else:
            last_modified_text = str(last_modified) if last_modified else None
        return {
            "repo_id": f"{HF_NAMESPACE}/{validate_dataset_name(name)}",
            "name": name,
            "private": bool(item.private),
            "gated": bool(item.gated),
            "downloads": int(item.downloads or 0),
            "likes": int(item.likes or 0),
            "last_modified": last_modified_text,
            "latest_commit_sha": validate_commit_sha(item.sha),
            "tags": [str(tag) for tag in (item.tags or [])[:30]],
        }


def import_huggingface_dataset(
    *,
    database: Database,
    settings: Settings,
    payload: dict[str, Any],
    job_id: str,
    worker_id: str,
    gateway: HuggingFaceGateway | None = None,
) -> dict[str, Any]:
    dataset_name = validate_dataset_name(payload["dataset_name"])
    requested_revision = validate_revision(payload["requested_revision"])
    commit_sha = validate_commit_sha(payload["commit_sha"])
    generation = payload.get("generation")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 1
    ):
        raise HuggingFaceImportValidationError("Invalid import generation")
    repo_id = _repo_id(dataset_name)
    gateway = gateway or HuggingFaceGateway(settings.hf_read_token)

    resolved = gateway.resolve_revision(dataset_name, commit_sha)
    if resolved["commit_sha"] != commit_sha:
        raise HuggingFaceImportValidationError(
            "Resolved commit does not match the requested import"
        )
    _enforce_size_limit(resolved.get("total_bytes"), settings.hf_import_max_bytes)

    lock_root = settings.jobs_root / "hf-import-locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_name = hashlib.sha256(repo_id.encode()).hexdigest() + ".lock"
    with _exclusive_lock(lock_root / lock_name):
        return _import_locked(
            database=database,
            settings=settings,
            gateway=gateway,
            repo_id=repo_id,
            dataset_name=dataset_name,
            requested_revision=requested_revision,
            commit_sha=commit_sha,
            generation=generation,
            job_id=job_id,
            worker_id=worker_id,
        )


def _import_locked(
    *,
    database: Database,
    settings: Settings,
    gateway: HuggingFaceGateway,
    repo_id: str,
    dataset_name: str,
    requested_revision: str,
    commit_sha: str,
    generation: int,
    job_id: str,
    worker_id: str,
) -> dict[str, Any]:
    raw_root = _require_real_directory(settings.nas_root / "raw", "raw storage")
    manifests_root = _require_real_directory(
        settings.nas_root / "manifests", "manifest storage"
    )
    revision_root = _safe_directory_chain(
        raw_root,
        ("hf", HF_NAMESPACE, dataset_name, "revisions"),
    )
    dataset_root = revision_root / commit_sha
    relative_path = (
        Path("hf") / HF_NAMESPACE / dataset_name / "revisions" / commit_sha
    ).as_posix()
    manifest_dir = _safe_directory_chain(
        manifests_root,
        ("hf", HF_NAMESPACE, dataset_name),
    )
    manifest_path = manifest_dir / f"{commit_sha}.json"

    if dataset_root.exists():
        if dataset_root.is_symlink() or not dataset_root.is_dir():
            raise HuggingFaceImportConflictError(
                "Immutable revision destination is not a directory"
            )
        candidate = inspect_dataset(
            area_root=raw_root,
            storage_area="raw",
            relative_path=relative_path,
        )
        if candidate.readiness != "ready":
            raise HuggingFaceImportConflictError(
                "Immutable revision destination is not a valid dataset"
            )
        manifest = _build_tree_manifest(dataset_root)
        database.assert_job_lease(job_id, worker_id=worker_id)
        return _finish_import_record(
            database=database,
            repo_id=repo_id,
            requested_revision=requested_revision,
            commit_sha=commit_sha,
            relative_path=relative_path,
            manifest=manifest,
            dataset_base=revision_root.parent,
            reused=True,
            generation=generation,
            job_id=job_id,
            worker_id=worker_id,
            manifest_path=manifest_path,
        )

    staging_parent = settings.staging_root / "hf-imports"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        tempfile.mkdtemp(
            prefix=f"{dataset_name}-{commit_sha[:12]}-", dir=staging_parent
        )
    )
    incoming_path = revision_root / f".incoming-{commit_sha}-{uuid.uuid4().hex}"
    try:
        gateway.download_snapshot(
            repo_id=repo_id,
            commit_sha=commit_sha,
            local_dir=staging_path,
        )
        metadata_cache = staging_path / ".cache" / "huggingface"
        if metadata_cache.exists():
            shutil.rmtree(metadata_cache)
            _remove_empty_parent(metadata_cache.parent, staging_path)
        candidate = inspect_dataset(
            area_root=staging_parent,
            storage_area="raw",
            relative_path=staging_path.name,
        )
        if candidate.readiness != "ready":
            raise HuggingFaceImportValidationError(
                "Downloaded dataset is not a complete supported LeRobot dataset"
            )
        source_manifest = _build_tree_manifest(staging_path)
        _enforce_size_limit(
            source_manifest["total_bytes"], settings.hf_import_max_bytes
        )

        shutil.copytree(staging_path, incoming_path, symlinks=True)
        copied_manifest = _build_tree_manifest(incoming_path)
        if copied_manifest != source_manifest:
            raise HuggingFaceImportValidationError(
                "Copied dataset does not match the downloaded snapshot"
            )
        database.assert_job_lease(job_id, worker_id=worker_id)
        try:
            incoming_path.rename(dataset_root)
        except OSError as exc:
            if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            existing_manifest = _build_tree_manifest(dataset_root)
            if existing_manifest != source_manifest:
                raise HuggingFaceImportConflictError(
                    "Immutable revision already exists with different content"
                )
        return _finish_import_record(
            database=database,
            repo_id=repo_id,
            requested_revision=requested_revision,
            commit_sha=commit_sha,
            relative_path=relative_path,
            manifest=source_manifest,
            dataset_base=revision_root.parent,
            reused=False,
            generation=generation,
            job_id=job_id,
            worker_id=worker_id,
            manifest_path=manifest_path,
        )
    finally:
        shutil.rmtree(staging_path, ignore_errors=True)
        shutil.rmtree(incoming_path, ignore_errors=True)


def _finish_import_record(
    *,
    database: Database,
    repo_id: str,
    requested_revision: str,
    commit_sha: str,
    relative_path: str,
    manifest: dict[str, Any],
    dataset_base: Path,
    reused: bool,
    generation: int,
    job_id: str,
    worker_id: str,
    manifest_path: Path,
) -> dict[str, Any]:
    pointer = {
        "repo_id": repo_id,
        "requested_revision": requested_revision,
        "commit_sha": commit_sha,
        "relative_path": relative_path,
        "manifest_sha256": manifest["tree_sha256"],
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "generation": generation,
    }
    try:
        database.record_hf_revision(
            repo_id=repo_id,
            requested_revision=requested_revision,
            commit_sha=commit_sha,
            relative_path=relative_path,
            manifest_sha256=manifest["tree_sha256"],
            file_count=manifest["file_count"],
            total_bytes=manifest["total_bytes"],
        )
    except ImmutableRevisionConflictError as exc:
        raise HuggingFaceImportConflictError(
            "Immutable revision already exists with different content"
        ) from exc
    _write_json_atomic(
        manifest_path,
        _manifest_document(repo_id, commit_sha, manifest),
    )
    database.assert_job_lease(job_id, worker_id=worker_id)
    is_current = database.claim_hf_current(
        repo_id=repo_id,
        commit_sha=commit_sha,
        generation=generation,
        job_id=job_id,
        worker_id=worker_id,
    )
    if not is_current:
        return {**pointer, "reused": reused, "current": False}

    pointer_document = {
        "schema_version": 1,
        "repo_id": repo_id,
        "requested_revision": requested_revision,
        "commit_sha": commit_sha,
        "revision_path": f"revisions/{commit_sha}",
        "manifest_sha256": manifest["tree_sha256"],
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "generation": generation,
    }
    _write_json_atomic(dataset_base / "current.json", pointer_document)
    if not database.confirm_hf_current(
        repo_id=repo_id,
        commit_sha=commit_sha,
        generation=generation,
        job_id=job_id,
        worker_id=worker_id,
    ):
        database.assert_job_lease(job_id, worker_id=worker_id)
        return {**pointer, "reused": reused, "current": False}
    return {**pointer, "reused": reused, "current": True}


def _build_tree_manifest(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in sorted(directory_names):
            path = current_path / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise HuggingFaceImportValidationError(
                    "Dataset contains an unsafe directory entry"
                )
        for name in sorted(file_names):
            path = current_path / name
            relative_path = path.relative_to(root).as_posix()
            file_digest = hashlib.sha256()
            size = 0
            try:
                file_descriptor = os.open(
                    path,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                )
            except OSError as exc:
                raise HuggingFaceImportValidationError(
                    "Dataset contains an unsafe file entry"
                ) from exc
            try:
                file_stat = os.fstat(file_descriptor)
                if not stat.S_ISREG(file_stat.st_mode):
                    raise HuggingFaceImportValidationError(
                        "Dataset contains an unsafe file entry"
                    )
                while chunk := os.read(file_descriptor, 1024 * 1024):
                    file_digest.update(chunk)
                    size += len(chunk)
            finally:
                os.close(file_descriptor)
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(size).encode())
            digest.update(b"\0")
            digest.update(file_digest.hexdigest().encode())
            digest.update(b"\n")
            file_count += 1
            total_bytes += size
    return {
        "tree_sha256": digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def _manifest_document(
    repo_id: str, commit_sha: str, manifest: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": {"type": "huggingface", "repo_id": repo_id, "commit_sha": commit_sha},
        "content": manifest,
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_directory_chain(root: Path, components: tuple[str, ...]) -> Path:
    current = _require_real_directory(root, "storage root")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(current, flags)
    try:
        for component in components:
            if component in {"", ".", ".."} or "/" in component or "\\" in component:
                raise HuggingFaceImportValidationError("Unsafe storage path component")
            try:
                os.mkdir(component, mode=0o775, dir_fd=directory_fd)
            except FileExistsError:
                pass
            next_fd = os.open(component, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
            current = current / component
    except OSError as exc:
        raise HuggingFaceImportValidationError("Unsafe storage directory") from exc
    finally:
        os.close(directory_fd)
    return current


def _require_real_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise HuggingFaceImportValidationError(f"{label} is unavailable")
    return path


def _remove_empty_parent(path: Path, stop: Path) -> None:
    current = path
    while current != stop:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _enforce_size_limit(total_bytes: int | None, maximum: int) -> None:
    if total_bytes is not None and total_bytes > maximum:
        raise HuggingFaceImportTooLargeError(
            "Dataset exceeds the configured import size limit"
        )


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    with path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _repo_id(dataset_name: str) -> str:
    return f"{HF_NAMESPACE}/{validate_dataset_name(dataset_name)}"
