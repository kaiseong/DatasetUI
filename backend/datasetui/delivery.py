from __future__ import annotations

import json
import os
import posixpath
import shutil
import stat
import tempfile
import uuid
from pathlib import Path
from typing import Any

from datasetui.config import Settings
from datasetui.database import Database, RecipeRevisionMismatchError
from datasetui.huggingface import HF_NAMESPACE
from datasetui.transform_errors import CurationTransformError
from datasetui.transforms import (
    _real_directory,
    _safe_dataset_root,
    _tree_manifest,
    _write_json_atomic,
)


class ExportGateRequiredError(CurationTransformError):
    pass


def _source(
    database: Database, settings: Settings, payload: dict[str, Any]
) -> tuple[dict[str, Any], Path]:
    record = database.get_dataset(payload["dataset_id"])
    if (
        not record["available"]
        or record["readiness"] != "ready"
        or record["fingerprint"] != payload["fingerprint"]
        or record["storage_area"] != payload["storage_area"]
        or record["relative_path"] != payload["relative_path"]
    ):
        raise RecipeRevisionMismatchError(payload["dataset_id"])
    if not database.has_successful_export_gate(
        dataset_id=record["id"], dataset_fingerprint=record["fingerprint"]
    ):
        raise ExportGateRequiredError(
            "A successful export gate is required for this exact dataset revision"
        )
    return record, _safe_dataset_root(
        settings.nas_root, record["storage_area"], record["relative_path"]
    )


def export_to_nas(
    *, database: Database, settings: Settings, payload: dict[str, Any], job_id: str, worker_id: str
) -> dict[str, Any]:
    record, source = _source(database, settings, payload)
    exports = _real_directory(settings.nas_root / "exports")
    final = exports / payload["output_name"]
    source_manifest = _tree_manifest(source)
    if final.exists():
        if final.is_symlink() or not final.is_dir() or _tree_manifest(final) != source_manifest:
            raise CurationTransformError("NAS export name already exists")
        return _delivery_result("nas", record, source_manifest, output_name=payload["output_name"], reused=True)
    incoming = exports / f".incoming-{job_id}-{uuid.uuid4().hex}"
    try:
        shutil.copytree(source, incoming)
        if _tree_manifest(incoming) != source_manifest:
            raise CurationTransformError("NAS export copy verification failed")
        database.assert_job_lease(job_id, worker_id=worker_id)
        incoming.rename(final)
    finally:
        shutil.rmtree(incoming, ignore_errors=True)
    result = _delivery_result("nas", record, source_manifest, output_name=payload["output_name"], reused=False)
    _write_delivery_manifest(settings, job_id, result)
    return result


def upload_to_huggingface(
    *, database: Database, settings: Settings, payload: dict[str, Any], job_id: str, worker_id: str
) -> dict[str, Any]:
    record, source = _source(database, settings, payload)
    if not settings.hf_write_token:
        raise CurationTransformError("Hugging Face write access is not configured")
    from huggingface_hub import HfApi

    repo_id = f"{HF_NAMESPACE}/{payload['repo_name']}"
    staging_parent = settings.staging_root / "hf-uploads"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{job_id}-", dir=staging_parent))
    created = False
    try:
        upload_root = staging / payload["repo_name"]
        shutil.copytree(source, upload_root)
        readme = upload_root / "README.md"
        existing = readme.read_text(encoding="utf-8") if readme.is_file() else ""
        readme.write_text(
            existing.rstrip()
            + "\n\n## DatasetUI lineage\n\n"
            + f"- Source fingerprint: `{record['fingerprint']}`\n"
            + "- Export gate: passed\n",
            encoding="utf-8",
        )
        _tree_manifest(upload_root)
        database.assert_job_lease(job_id, worker_id=worker_id)
        api = HfApi(token=settings.hf_write_token)
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=payload["visibility"] == "private",
            exist_ok=False,
            token=settings.hf_write_token,
        )
        created = True
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=upload_root,
            commit_message="Publish verified DatasetUI export",
            token=settings.hf_write_token,
        )
        result = {
            "kind": "huggingface",
            "source_dataset_id": record["id"],
            "source_fingerprint": record["fingerprint"],
            "repo_id": repo_id,
            "visibility": payload["visibility"],
            "commit_url": str(getattr(commit, "commit_url", "")),
        }
        _write_delivery_manifest(settings, job_id, result)
        return result
    except Exception:
        if created:
            try:
                HfApi(token=settings.hf_write_token).delete_repo(
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=settings.hf_write_token,
                )
            except Exception:
                pass
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def copy_to_pc_with_key(
    *, database: Database, settings: Settings, payload: dict[str, Any]
) -> dict[str, Any]:
    record, source = _source(database, settings, payload)
    return _copy_to_pc(
        settings=settings,
        source=source,
        record=record,
        host=payload["host"],
        port=payload["port"],
        username=payload["username"],
        destination=payload["destination"],
        password=None,
    )


def copy_to_pc_with_password(
    *, database: Database, settings: Settings, dataset_id: str, profile_id: str,
    host: str, port: int, username: str, password: str, destination: str
) -> dict[str, Any]:
    database.get_profile(profile_id)
    record = database.get_dataset(dataset_id)
    payload = {
        "dataset_id": record["id"],
        "fingerprint": record["fingerprint"],
        "storage_area": record["storage_area"],
        "relative_path": record["relative_path"],
    }
    record, source = _source(database, settings, payload)
    return _copy_to_pc(
        settings=settings,
        source=source,
        record=record,
        host=host,
        port=port,
        username=username,
        destination=destination,
        password=password,
    )


def _copy_to_pc(
    *, settings: Settings, source: Path, record: dict[str, Any], host: str,
    port: int, username: str, destination: str, password: str | None
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
        files, total = _sftp_tree(sftp, source, incoming)
        sftp.rename(incoming, final)
        incoming = ""
        return {"ok": True, "files": files, "bytes": total, "destination": destination}
    finally:
        if incoming:
            _remote_remove_tree(sftp, incoming)
        sftp.close()
        client.close()


def _sftp_tree(sftp: Any, source: Path, target: str) -> tuple[int, int]:
    files = 0
    total = 0
    for current, directories, names in os.walk(source, followlinks=False):
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
            files += 1
            total += metadata.st_size
    return files, total


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
    kind: str, record: dict[str, Any], manifest: dict[str, Any], *, output_name: str, reused: bool
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


def _write_delivery_manifest(settings: Settings, job_id: str, result: dict[str, Any]) -> None:
    root = _real_directory(settings.nas_root / "manifests") / "delivery"
    root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(root / f"{job_id}.json", {"schema_version": 1, **result})
