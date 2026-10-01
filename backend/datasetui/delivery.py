from __future__ import annotations

import os
import posixpath
import shutil
import stat
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from datasetui.config import Settings
from datasetui.database import Database, RecipeRevisionMismatchError
from datasetui.huggingface import HF_NAMESPACE
from datasetui.validation_integrity import (
    ContentIntegrityError,
    VALIDATOR_POLICY,
    validation_content_manifest,
)
from datasetui.job_progress import JobProgressReporter
from datasetui.transform_errors import CurationTransformError
from datasetui.transforms import (
    _real_directory,
    _safe_dataset_root,
    _write_json_atomic,
)


class ExportGateRequiredError(CurationTransformError):
    pass


class HuggingFaceExternalOperationAmbiguousError(CurationTransformError):
    def __init__(self, repo_id: str, phase: str):
        self.repo_id = repo_id
        self.phase = phase
        super().__init__(
            f"Hugging Face {phase} outcome is uncertain for repository {repo_id}"
        )


def _require_current_gate(database: Database, record: dict, manifest: dict) -> None:
    gate = database.export_gate_result(
        dataset_id=record["id"], dataset_fingerprint=record["fingerprint"]
    )
    if (
        not gate
        or gate.get("job_status") != "succeeded"
        or gate.get("passed") is not True
        or gate.get("validator_policy") != VALIDATOR_POLICY
        or gate.get("content_manifest") != manifest
    ):
        raise ExportGateRequiredError("A current content-bound export gate is required")


def _source(
    database: Database,
    settings: Settings,
    payload: dict[str, Any],
    *,
    exclude_job_id: str | None = None,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    record = database.get_dataset(payload["dataset_id"])
    if (
        not record["available"]
        or record["readiness"] != "ready"
        or record["fingerprint"] != payload["fingerprint"]
        or record["storage_area"] != payload["storage_area"]
        or record["relative_path"] != payload["relative_path"]
    ):
        raise RecipeRevisionMismatchError(payload["dataset_id"])
    if exclude_job_id is None and not database.has_successful_export_gate(
        dataset_id=record["id"], dataset_fingerprint=record["fingerprint"]
    ):
        raise ExportGateRequiredError(
            "A successful export gate is required for this exact dataset revision"
        )
    source = _safe_dataset_root(
        settings.nas_root, record["storage_area"], record["relative_path"]
    )
    try:
        manifest = validation_content_manifest(source)
    except ContentIntegrityError as exc:
        raise CurationTransformError("Dataset content could not be verified") from exc
    gate = database.export_gate_result(
        dataset_id=record["id"],
        dataset_fingerprint=record["fingerprint"],
        exclude_job_id=exclude_job_id,
    )
    if (
        not gate
        or gate.get("job_status") != "succeeded"
        or gate.get("passed") is not True
        or gate.get("validator_policy") != VALIDATOR_POLICY
        or gate.get("content_manifest") != manifest
    ):
        raise ExportGateRequiredError(
            "Export gate does not match current dataset content"
        )
    return record, source, manifest


def export_to_nas(
    *,
    database: Database,
    settings: Settings,
    payload: dict[str, Any],
    job_id: str,
    worker_id: str,
) -> dict[str, Any]:
    progress = JobProgressReporter(
        database, job_id=job_id, worker_id=worker_id, output_name=payload["output_name"]
    )
    _emit_progress(
        progress,
        stage="preparing",
        completed=0,
        total=0,
        unit="items",
        current_item="내보낼 데이터셋 확인",
    )
    record, source, source_manifest = _source(database, settings, payload)
    exports = _real_directory(settings.nas_root / "exports")
    final = exports / payload["output_name"]
    if final.exists():
        if (
            final.is_symlink()
            or not final.is_dir()
            or validation_content_manifest(final) != source_manifest
        ):
            raise CurationTransformError("NAS export name already exists")
        _emit_progress(
            progress,
            stage="complete",
            completed=1,
            total=1,
            unit="items",
            current_item="기존 NAS 내보내기 확인",
        )
        return _delivery_result(
            "nas",
            record,
            source_manifest,
            output_name=payload["output_name"],
            reused=True,
        )
    incoming = exports / f".incoming-{job_id}-{uuid.uuid4().hex}"
    _emit_progress(
        progress,
        stage="copy",
        completed=0,
        total=source_manifest["file_count"],
        unit="files",
        current_item="NAS로 파일 복사",
    )
    try:
        _copytree_with_progress(
            source,
            incoming,
            total_files=source_manifest["file_count"],
            on_progress=progress,
        )
        _emit_progress(
            progress,
            stage="verify",
            completed=0,
            total=0,
            unit="files",
            current_item="복사본 무결성 확인",
        )
        if validation_content_manifest(incoming) != source_manifest:
            raise CurationTransformError("NAS export copy verification failed")
        database.assert_job_lease(job_id, worker_id=worker_id)
        _require_current_gate(database, record, source_manifest)
        _emit_progress(
            progress,
            stage="publish",
            completed=0,
            total=1,
            unit="items",
            current_item="NAS 내보내기 공개",
        )
        database.begin_job_finalization(job_id, worker_id=worker_id)
        incoming.rename(final)
        _emit_progress(
            progress,
            stage="publish",
            completed=1,
            total=1,
            unit="items",
            current_item="NAS 내보내기 공개 완료",
        )
    finally:
        shutil.rmtree(incoming, ignore_errors=True)
    result = _delivery_result(
        "nas", record, source_manifest, output_name=payload["output_name"], reused=False
    )
    _write_delivery_manifest(settings, job_id, result)
    _emit_progress(
        progress,
        stage="complete",
        completed=1,
        total=1,
        unit="items",
        current_item="NAS 내보내기 완료",
    )
    return result


def upload_to_huggingface(
    *,
    database: Database,
    settings: Settings,
    payload: dict[str, Any],
    job_id: str,
    worker_id: str,
) -> dict[str, Any]:
    progress = JobProgressReporter(
        database, job_id=job_id, worker_id=worker_id, output_name=payload["repo_name"]
    )
    _emit_progress(
        progress,
        stage="preparing",
        completed=0,
        total=0,
        unit="items",
        current_item="업로드할 데이터셋 확인",
    )
    record, source, source_manifest = _source(database, settings, payload)
    if not settings.hf_write_token:
        raise CurationTransformError("Hugging Face write access is not configured")
    from huggingface_hub import HfApi

    repo_id = f"{HF_NAMESPACE}/{payload['repo_name']}"
    staging_parent = settings.staging_root / "hf-uploads"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{job_id}-", dir=staging_parent))
    external_phase: str | None = None
    try:
        upload_root = staging / payload["repo_name"]
        _copytree_with_progress(
            source,
            upload_root,
            total_files=source_manifest["file_count"],
            on_progress=progress,
        )
        _emit_progress(
            progress,
            stage="verify",
            completed=0,
            total=0,
            unit="files",
            current_item="업로드 준비본 무결성 확인",
        )
        if validation_content_manifest(upload_root) != source_manifest:
            raise CurationTransformError(
                "Hugging Face staging copy verification failed"
            )
        readme = upload_root / "README.md"
        existing = readme.read_text(encoding="utf-8") if readme.is_file() else ""
        readme.write_text(
            existing.rstrip()
            + "\n\n## DatasetUI lineage\n\n"
            + f"- Source fingerprint: `{record['fingerprint']}`\n"
            + "- Export gate: passed\n",
            encoding="utf-8",
        )
        upload_manifest = validation_content_manifest(upload_root)
        database.assert_job_lease(job_id, worker_id=worker_id)
        api = HfApi(token=settings.hf_write_token)
        _require_current_gate(database, record, source_manifest)
        _emit_progress(
            progress,
            stage="register",
            completed=0,
            total=0,
            unit="items",
            current_item=f"{repo_id} 저장소 생성",
        )
        # External publication cannot be rolled back safely after this point.
        database.begin_job_finalization(job_id, worker_id=worker_id)
        external_phase = "repository creation"
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=payload["visibility"] == "private",
            exist_ok=False,
            token=settings.hf_write_token,
        )
        external_phase = "upload"
        _emit_progress(
            progress,
            stage="upload",
            completed=0,
            total=0,
            unit="bytes",
            current_item=f"{repo_id} 업로드",
        )
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=upload_root,
            commit_message="Publish verified DatasetUI export",
            token=settings.hf_write_token,
        )
        external_phase = None
        database.assert_job_lease(job_id, worker_id=worker_id)
        _emit_progress(
            progress,
            stage="verify",
            completed=0,
            total=0,
            unit="items",
            current_item="Hugging Face 업로드 결과 확인",
        )
        result = {
            "kind": "huggingface",
            "source_dataset_id": record["id"],
            "source_fingerprint": record["fingerprint"],
            "repo_id": repo_id,
            "visibility": payload["visibility"],
            "commit_url": str(getattr(commit, "commit_url", "")),
            "source_content_manifest": source_manifest,
            "upload_content_manifest": upload_manifest,
        }
        _write_delivery_manifest(settings, job_id, result)
        _emit_progress(
            progress,
            stage="complete",
            completed=1,
            total=1,
            unit="items",
            current_item="Hugging Face 업로드 완료",
        )
        return result
    except Exception:
        if external_phase is not None:
            raise HuggingFaceExternalOperationAmbiguousError(
                repo_id, external_phase
            ) from None
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def copy_to_pc_with_key(
    *,
    database: Database,
    settings: Settings,
    payload: dict[str, Any],
    job_id: str,
    worker_id: str,
) -> dict[str, Any]:
    progress = JobProgressReporter(
        database, job_id=job_id, worker_id=worker_id, output_name=payload["destination"]
    )
    _emit_progress(
        progress,
        stage="preparing",
        completed=0,
        total=0,
        unit="items",
        current_item="전송할 데이터셋 확인",
    )
    record, source, source_manifest = _source(database, settings, payload)
    return _copy_verified_to_pc(
        expected_manifest=source_manifest,
        settings=settings,
        source=source,
        record=record,
        host=payload["host"],
        port=payload["port"],
        username=payload["username"],
        destination=payload["destination"],
        password=None,
        expected_fingerprint=source_manifest["tree_sha256"],
        lease_check=lambda: database.assert_job_lease(job_id, worker_id=worker_id),
        finalize=lambda: database.begin_job_finalization(job_id, worker_id=worker_id),
        on_progress=progress,
    )


def copy_to_pc_with_password(
    *,
    database: Database,
    settings: Settings,
    dataset_id: str,
    profile_id: str,
    host: str,
    port: int,
    username: str,
    password: str,
    destination: str,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    database.get_profile(profile_id)
    record = database.get_dataset(dataset_id)
    payload = {
        "dataset_id": record["id"],
        "fingerprint": record["fingerprint"],
        "storage_area": record["storage_area"],
        "relative_path": record["relative_path"],
    }
    record, source, source_manifest = _source(database, settings, payload)
    return _copy_verified_to_pc(
        expected_manifest=source_manifest,
        settings=settings,
        source=source,
        record=record,
        host=host,
        port=port,
        username=username,
        destination=destination,
        password=password,
        expected_fingerprint=source_manifest["tree_sha256"],
        lease_check=None,
        on_progress=on_progress,
    )


def _copy_verified_to_pc(
    *,
    expected_manifest: dict[str, Any],
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Upload a private, verified copy, never a concurrently changing NAS tree."""
    settings = kwargs["settings"]
    staging_parent = settings.staging_root / "pc-copies"
    staging_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="verified-", dir=staging_parent
    ) as directory:
        source = Path(directory) / "dataset"
        _copytree_with_progress(
            kwargs["source"],
            source,
            total_files=expected_manifest["file_count"],
            on_progress=on_progress,
            stage="preparing",
        )
        _emit_progress(
            on_progress,
            stage="verify",
            completed=0,
            total=0,
            unit="files",
            current_item="전송 준비본 무결성 확인",
        )
        if validation_content_manifest(source) != expected_manifest:
            raise ExportGateRequiredError(
                "PC staging copy differs from the validated dataset"
            )
        return _copy_to_pc(**{**kwargs, "source": source, "on_progress": on_progress})


def _copy_to_pc(
    *,
    settings: Settings,
    source: Path,
    record: dict[str, Any],
    host: str,
    port: int,
    username: str,
    destination: str,
    password: str | None,
    expected_fingerprint: str,
    lease_check: Callable[[], None] | None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    finalize: Callable[[], None] | None = None,
) -> dict[str, Any]:
    import paramiko

    known_hosts = settings.ssh_known_hosts_path
    if known_hosts.is_symlink() or not known_hosts.is_file():
        raise CurationTransformError("SSH known-hosts verification is not configured")
    client = paramiko.SSHClient()
    client.load_host_keys(str(known_hosts))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    connect: dict[str, Any] = {
        "hostname": host,
        "port": port,
        "username": username,
        "timeout": 15,
        "banner_timeout": 15,
        "auth_timeout": 15,
        "allow_agent": False,
        "look_for_keys": False,
    }
    if password is None:
        key = settings.ssh_private_key_path
        if key.is_symlink() or not key.is_file():
            raise CurationTransformError("SSH key authentication is not configured")
        connect["key_filename"] = str(key)
    else:
        connect["password"] = password
    _emit_progress(
        on_progress,
        stage="connect",
        completed=0,
        total=0,
        unit="items",
        current_item=f"{host}:{port} 연결",
    )
    client.connect(**connect)
    sftp = client.open_sftp()
    incoming = ""
    try:
        home = sftp.normalize(".")
        relative = destination[2:]
        final = posixpath.join(home, relative)
        parent = posixpath.dirname(final)
        _remote_mkdirs(sftp, parent)
        try:
            sftp.lstat(final)
        except FileNotFoundError:
            pass
        else:
            raise CurationTransformError("PC destination already exists")
        incoming = posixpath.join(home, f".datasetui-incoming-{uuid.uuid4().hex}")
        sftp.mkdir(incoming)
        files, total = _sftp_tree(
            sftp,
            source,
            incoming,
            lease_check=lease_check,
            on_progress=on_progress,
            total_bytes=validation_content_manifest(source)["total_bytes"],
        )
        _emit_progress(
            on_progress,
            stage="verify",
            completed=0,
            total=0,
            unit="files",
            current_item="전송 원본 무결성 재확인",
        )
        if validation_content_manifest(source)["tree_sha256"] != expected_fingerprint:
            raise RecipeRevisionMismatchError(record["id"])
        if lease_check is not None:
            lease_check()
        if finalize is not None:
            finalize()
        sftp.rename(incoming, final)
        incoming = ""
        _emit_progress(
            on_progress,
            stage="complete",
            completed=1,
            total=1,
            unit="items",
            current_item="사용자 PC 복사 완료",
        )
        return {"ok": True, "files": files, "bytes": total, "destination": destination}
    finally:
        if incoming:
            _remote_remove_tree(sftp, incoming)
        sftp.close()
        client.close()


def _sftp_tree(
    sftp: Any,
    source: Path,
    target: str,
    *,
    lease_check: Callable[[], None] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    total_bytes: int = 0,
) -> tuple[int, int]:
    files = 0
    total = 0
    for current, directories, names in os.walk(source, followlinks=False):
        if lease_check is not None:
            lease_check()
        current_path = Path(current)
        relative = current_path.relative_to(source).as_posix()
        remote = target if relative == "." else posixpath.join(target, relative)
        for directory in sorted(directories):
            local = current_path / directory
            if local.is_symlink():
                raise CurationTransformError("Dataset contains a symlink")
            sftp.mkdir(posixpath.join(remote, directory))
        for name in sorted(names):
            local = current_path / name
            metadata = local.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise CurationTransformError("Dataset contains an unsafe file")
            sftp.put(str(local), posixpath.join(remote, name))
            if lease_check is not None:
                lease_check()
            files += 1
            total += metadata.st_size
            if on_progress is not None:
                on_progress(
                    {
                        "stage": "copy",
                        "completed": total,
                        "total": total_bytes,
                        "unit": "bytes",
                        "current_item": name,
                    }
                )
    return files, total


def _copytree_with_progress(
    source: Path,
    destination: Path,
    *,
    total_files: int,
    on_progress: Callable[[dict[str, Any]], None] | None,
    stage: str = "copy",
) -> None:
    completed = 0
    _emit_progress(
        on_progress,
        stage=stage,
        completed=0,
        total=total_files,
        unit="files",
        current_item="파일 복사 시작",
    )

    def copy_file(source_file: str, destination_file: str) -> str:
        nonlocal completed
        copied = shutil.copy2(source_file, destination_file)
        completed += 1
        try:
            _emit_progress(
                on_progress,
                stage=stage,
                completed=completed,
                total=total_files,
                unit="files",
                current_item=Path(source_file).name,
            )
        except OSError as exc:
            raise _ProgressCallbackFailure(exc) from exc
        return copied

    try:
        shutil.copytree(source, destination, copy_function=copy_file)
    except _ProgressCallbackFailure as exc:
        raise exc.original from exc


class _ProgressCallbackFailure(Exception):
    def __init__(self, original: OSError):
        super().__init__(str(original))
        self.original = original


def _emit_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    *,
    stage: str,
    completed: int,
    total: int,
    unit: str,
    current_item: str,
) -> None:
    if callback is not None:
        callback(
            {
                "stage": stage,
                "completed": completed,
                "total": total,
                "unit": unit,
                "current_item": current_item,
                "_force": completed >= total if total > 0 else False,
            }
        )


def _remote_mkdirs(sftp: Any, path: str) -> None:
    current = "/"
    for part in path.strip("/").split("/"):
        current = posixpath.join(current, part)
        try:
            metadata = sftp.lstat(current)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise CurationTransformError("PC destination path is unsafe")
        except FileNotFoundError:
            sftp.mkdir(current)


def _remote_remove_tree(sftp: Any, path: str) -> None:
    try:
        for item in sftp.listdir_attr(path):
            child = posixpath.join(path, item.filename)
            if stat.S_ISDIR(item.st_mode) and not stat.S_ISLNK(item.st_mode):
                _remote_remove_tree(sftp, child)
            else:
                sftp.remove(child)
        sftp.rmdir(path)
    except Exception:
        pass


def _delivery_result(
    kind: str,
    record: dict[str, Any],
    manifest: dict[str, Any],
    *,
    output_name: str,
    reused: bool,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "source_dataset_id": record["id"],
        "source_fingerprint": record["fingerprint"],
        "output_name": output_name,
        "manifest_sha256": manifest["tree_sha256"],
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "reused": reused,
    }


def _write_delivery_manifest(
    settings: Settings, job_id: str, result: dict[str, Any]
) -> None:
    root = _real_directory(settings.nas_root / "manifests") / "delivery"
    root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(root / f"{job_id}.json", {"schema_version": 1, **result})
