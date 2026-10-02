"""Publishing staged outputs into derived/, and guarding against source changes."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import uuid
from pathlib import Path
from typing import Any

from datasetui.config import Settings
from datasetui.content_integrity import ContentIntegrityError, dataset_tree_identity
from datasetui.database import Database, RecipeRevisionMismatchError
from datasetui.dataset_io.files import (
    real_directory,
    write_json_atomic,
)
from datasetui.datasets import scan_storage_area
from datasetui.job_progress import ProgressCallback, report_progress
from datasetui.transform_errors import CurationTransformError


def publish_output(
    *,
    database: Database,
    settings: Settings,
    job_id: str,
    worker_id: str,
    staging_path: Path,
    output_name: str,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    derived = real_directory(settings.nas_root / "derived")
    final = derived / output_name
    report_progress(
        on_progress,
        stage="publish",
        completed=0,
        total=0,
        unit="bytes",
        current_item="출력 해시 계산",
        force=True,
    )
    source_manifest = tree_manifest(staging_path)
    if final.exists():
        if (
            final.is_symlink()
            or not final.is_dir()
            or tree_manifest(final) != source_manifest
        ):
            raise CurationTransformError("Derived output name already exists")
        database.begin_job_finalization(job_id, worker_id=worker_id)
        report_progress(
            on_progress,
            stage="publish",
            completed=1,
            total=1,
            unit="items",
            current_item="기존 출력 재사용",
            force=True,
        )
        return source_manifest
    incoming = derived / f".incoming-{job_id}-{uuid.uuid4().hex}"
    try:
        report_progress(
            on_progress,
            stage="publish",
            completed=0,
            total=0,
            unit="bytes",
            current_item="출력 복사",
            force=True,
        )
        shutil.copytree(staging_path, incoming)
        report_progress(
            on_progress,
            stage="publish",
            completed=0,
            total=0,
            unit="bytes",
            current_item="복사 결과 확인",
            force=True,
        )
        if tree_manifest(incoming) != source_manifest:
            raise CurationTransformError("Derived output copy verification failed")
        # This atomic transition is the cancellation/publication boundary: a
        # cancel request that wins first prevents the final rename, while jobs
        # already finalizing are no longer advertised as cancellable.
        database.begin_job_finalization(job_id, worker_id=worker_id)
        incoming.rename(final)
        report_progress(
            on_progress,
            stage="publish",
            completed=1,
            total=1,
            unit="items",
            current_item="출력 게시 완료",
            force=True,
        )
        return source_manifest
    finally:
        shutil.rmtree(incoming, ignore_errors=True)




def write_run_manifest(
    settings: Settings, job_id: str, result: dict[str, Any]
) -> None:
    manifests = real_directory(settings.nas_root / "manifests")
    root = manifests / "curation"
    root.mkdir(exist_ok=True)
    root = real_directory(root)
    write_json_atomic(root / f"{job_id}.json", {"schema_version": 1, **result})


def refresh_derived_registry(database: Database, settings: Settings) -> None:
    generation = database.begin_dataset_scan("derived")
    database.synchronize_datasets(
        storage_area="derived",
        records=scan_storage_area(
            settings.nas_root,
            "derived",
            max_depth=settings.dataset_scan_max_depth,
        ),
        scan_generation=generation,
    )


def tree_manifest(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    total = 0
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories.sort()
        for name in directories:
            if (current_path / name).is_symlink():
                raise CurationTransformError("Derived dataset contains a symlink")
        for name in sorted(files):
            path = current_path / name
            descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            file_digest = hashlib.sha256()
            size = 0
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise CurationTransformError(
                        "Derived dataset contains an unsafe file"
                    )
                while chunk := os.read(descriptor, 1024 * 1024):
                    file_digest.update(chunk)
                    size += len(chunk)
            finally:
                os.close(descriptor)
            relative = path.relative_to(root).as_posix()
            digest.update(f"{relative}\0{size}\0{file_digest.hexdigest()}\n".encode())
            count += 1
            total += size
    return {
        "tree_sha256": digest.hexdigest(),
        "file_count": count,
        "total_bytes": total,
    }


def source_tree_identity(root: Path) -> str:
    """Stat-only snapshot of a source tree, taken before the job reads it."""
    try:
        return dataset_tree_identity(root)
    except ContentIntegrityError as exc:
        raise CurationTransformError("Dataset source contains an unsafe entry") from exc


def assert_source_unchanged(root: Path, baseline: str, dataset_id: str) -> None:
    """Refuse to publish an output built from a source modified mid-job.

    The registry fingerprint covers only ``meta/info.json``; this closes the gap
    for data and video files without hashing their contents.
    """
    try:
        current = dataset_tree_identity(root)
    except ContentIntegrityError:
        raise RecipeRevisionMismatchError(dataset_id) from None
    if current != baseline:
        raise RecipeRevisionMismatchError(dataset_id)
