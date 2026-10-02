from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import datasetui.delivery.transfer as delivery_module
from test_transforms import _settings, _write_v21

from datasetui.database import Database, JobLeaseLostError
from datasetui.datasets import inspect_dataset
from datasetui.delivery.transfer import (
    ExportGateRequiredError,
    HuggingFaceExternalOperationAmbiguousError,
    copy_to_pc_with_key,
    copy_to_pc_with_password,
    export_to_nas,
    upload_to_huggingface,
)
from datasetui.jobs import validate_job_payload
from datasetui.validation.integrity import validation_content_manifest, VALIDATOR_POLICY


def _registered(tmp_path: Path) -> tuple:
    settings = _settings(tmp_path)
    source = settings.nas_root / "raw/lab/pick"
    _write_v21(source)
    database = Database(settings.database_path)
    database.initialize()
    candidate = inspect_dataset(
        area_root=settings.nas_root / "raw",
        storage_area="raw",
        relative_path="lab/pick",
    )
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw", records=[candidate.as_record()], scan_generation=generation
    )
    dataset = database.list_datasets()[0]
    profile = database.create_profile("Exporter")
    return settings, database, dataset, profile, source


def _pass_gate(database: Database, dataset: dict, profile: dict) -> None:
    payload = {
        "dataset_id": dataset["id"],
        "fingerprint": dataset["fingerprint"],
        "storage_area": dataset["storage_area"],
        "relative_path": dataset["relative_path"],
        "mode": "export_gate",
    }
    job, _ = database.create_job(
        kind="datasets.validate",
        queue_name="cpu",
        profile_id=profile["id"],
        payload=payload,
        idempotency_key="gate",
    )
    database.record_validation_run(
        job_id=job["id"],
        dataset_id=dataset["id"],
        dataset_fingerprint=dataset["fingerprint"],
        mode="export_gate",
    )
    database.claim_job(job["id"], worker_id="validator", lease_seconds=120)
    database.succeed_job(
        job["id"],
        {
            "mode": "export_gate",
            "passed": True,
            "validator_policy": VALIDATOR_POLICY,
            "content_manifest": validation_content_manifest(
                database.path.parent
                / "nas"
                / dataset["storage_area"]
                / dataset["relative_path"]
            ),
        },
        worker_id="validator",
    )


def test_nas_delivery_requires_gate_and_publishes_verified_copy(tmp_path: Path) -> None:
    settings, database, dataset, profile, source = _registered(tmp_path)
    payload = {
        "dataset_id": dataset["id"],
        "fingerprint": dataset["fingerprint"],
        "storage_area": dataset["storage_area"],
        "relative_path": dataset["relative_path"],
        "output_name": "pick-export",
    }
    job, _ = database.create_job(
        kind="datasets.export_nas",
        queue_name="io",
        profile_id=profile["id"],
        payload=payload,
        idempotency_key="export",
    )
    database.claim_job(job["id"], worker_id="exporter", lease_seconds=120)
    with pytest.raises(ExportGateRequiredError):
        export_to_nas(
            database=database,
            settings=settings,
            payload=payload,
            job_id=job["id"],
            worker_id="exporter",
        )

    _pass_gate(database, dataset, profile)
    result = export_to_nas(
        database=database,
        settings=settings,
        payload=payload,
        job_id=job["id"],
        worker_id="exporter",
    )
    output = settings.nas_root / "exports/pick-export"
    assert result["manifest_sha256"]
    assert output.is_dir()
    assert (output / "meta/info.json").read_bytes() == (
        source / "meta/info.json"
    ).read_bytes()
    assert (settings.nas_root / "manifests/delivery" / f"{job['id']}.json").is_file()


def test_export_gate_is_bound_to_the_exact_fingerprint(tmp_path: Path) -> None:
    _, database, dataset, profile, _ = _registered(tmp_path)
    _pass_gate(database, dataset, profile)
    assert database.has_successful_export_gate(
        dataset_id=dataset["id"], dataset_fingerprint=dataset["fingerprint"]
    )
    assert not database.has_successful_export_gate(
        dataset_id=dataset["id"], dataset_fingerprint="f" * 64
    )


def test_one_time_pc_password_is_not_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, database, dataset, profile, _ = _registered(tmp_path)
    _pass_gate(database, dataset, profile)
    captured: dict = {}

    def fake_copy(**kwargs):
        captured.update(kwargs)
        return {
            "ok": True,
            "files": 1,
            "bytes": 10,
            "destination": kwargs["destination"],
        }

    monkeypatch.setattr(delivery_module, "_copy_to_pc", fake_copy)
    result = copy_to_pc_with_password(
        database=database,
        settings=settings,
        dataset_id=dataset["id"],
        profile_id=profile["id"],
        host="192.168.0.51",
        port=22,
        username="researcher",
        password="one-time-secret",
        destination="~/pick",
    )
    assert result["ok"] is True
    assert captured["password"] == "one-time-secret"
    assert "one-time-secret" not in str(database.list_jobs(limit=100))


def test_delivery_job_payload_rejects_any_secret_field() -> None:
    payload = {
        "dataset_id": "11111111-1111-4111-8111-111111111111",
        "fingerprint": "a" * 64,
        "storage_area": "derived",
        "relative_path": "safe/output",
        "host": "192.168.0.51",
        "port": 22,
        "username": "researcher",
        "destination": "~/pick",
        "password": "must-not-persist",
    }
    with pytest.raises(ValueError, match="delivery payload"):
        validate_job_payload("datasets.copy_pc_key", payload)


def test_pc_key_delivery_lost_lease_cannot_publish_remote_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, database, dataset, profile, _ = _registered(tmp_path)
    _pass_gate(database, dataset, profile)
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("test host key", encoding="utf-8")
    private_key = tmp_path / "id_ed25519"
    private_key.write_text("test private key", encoding="utf-8")
    settings = replace(
        settings,
        ssh_known_hosts_path=known_hosts,
        ssh_private_key_path=private_key,
    )
    payload = {
        "dataset_id": dataset["id"],
        "fingerprint": dataset["fingerprint"],
        "storage_area": dataset["storage_area"],
        "relative_path": dataset["relative_path"],
        "host": "192.168.0.51",
        "port": 22,
        "username": "researcher",
        "destination": "~/pick",
    }
    job, _ = database.create_job(
        kind="datasets.copy_pc_key",
        queue_name="io",
        profile_id=profile["id"],
        payload=payload,
        idempotency_key="pc-key-lease",
    )
    database.claim_job(job["id"], worker_id="copy-worker", lease_seconds=120)

    class FakeSftp:
        renamed = False

        def normalize(self, value):
            return "/home/researcher"

        def lstat(self, path):
            raise FileNotFoundError(path)

        def mkdir(self, path):
            return None

        def rename(self, source, target):
            self.renamed = True

        def listdir_attr(self, path):
            return []

        def rmdir(self, path):
            return None

        def close(self):
            return None

    fake_sftp = FakeSftp()

    class FakeClient:
        def load_host_keys(self, path):
            return None

        def set_missing_host_key_policy(self, policy):
            return None

        def connect(self, **kwargs):
            return None

        def open_sftp(self):
            return fake_sftp

        def close(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "paramiko",
        SimpleNamespace(SSHClient=FakeClient, RejectPolicy=object),
    )
    monkeypatch.setattr(delivery_module, "_sftp_tree", lambda *args, **kwargs: (1, 10))
    monkeypatch.setattr(
        database,
        "assert_job_lease",
        lambda *args, **kwargs: (_ for _ in ()).throw(JobLeaseLostError(job["id"])),
    )

    with pytest.raises(JobLeaseLostError):
        copy_to_pc_with_key(
            database=database,
            settings=settings,
            payload=payload,
            job_id=job["id"],
            worker_id="copy-worker",
        )
    assert fake_sftp.renamed is False


def test_hf_post_upload_lease_loss_does_not_delete_external_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, database, dataset, profile, _ = _registered(tmp_path)
    _pass_gate(database, dataset, profile)
    settings = replace(settings, hf_write_token="test-token")
    payload = {
        "dataset_id": dataset["id"],
        "fingerprint": dataset["fingerprint"],
        "storage_area": dataset["storage_area"],
        "relative_path": dataset["relative_path"],
        "repo_name": "pick-export",
        "visibility": "private",
    }
    job, _ = database.create_job(
        kind="datasets.upload_hf",
        queue_name="io",
        profile_id=profile["id"],
        payload=payload,
        idempotency_key="hf-lease",
    )
    database.claim_job(job["id"], worker_id="hf-worker", lease_seconds=120)
    deleted: list[str] = []

    class FakeApi:
        def __init__(self, token=None):
            return None

        def create_repo(self, **kwargs):
            return None

        def upload_folder(self, **kwargs):
            return SimpleNamespace(commit_url="https://example.invalid/commit")

        def delete_repo(self, **kwargs):
            deleted.append(kwargs["repo_id"])

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    lease_checks = 0

    def lease_check(*args, **kwargs):
        nonlocal lease_checks
        lease_checks += 1
        if lease_checks == 2:
            raise JobLeaseLostError(job["id"])

    monkeypatch.setattr(database, "assert_job_lease", lease_check)

    with pytest.raises(JobLeaseLostError):
        upload_to_huggingface(
            database=database,
            settings=settings,
            payload=payload,
            job_id=job["id"],
            worker_id="hf-worker",
        )
    assert deleted == []


def test_hf_uncertain_upload_is_reported_without_attempting_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, database, dataset, profile, _ = _registered(tmp_path)
    _pass_gate(database, dataset, profile)
    settings = replace(settings, hf_write_token="test-token")
    payload = {
        "dataset_id": dataset["id"],
        "fingerprint": dataset["fingerprint"],
        "storage_area": dataset["storage_area"],
        "relative_path": dataset["relative_path"],
        "repo_name": "pick-uncertain",
        "visibility": "private",
    }
    job, _ = database.create_job(
        kind="datasets.upload_hf",
        queue_name="io",
        profile_id=profile["id"],
        payload=payload,
        idempotency_key="hf-uncertain",
    )
    database.claim_job(job["id"], worker_id="hf-worker", lease_seconds=120)
    deleted: list[str] = []

    class FakeApi:
        def __init__(self, token=None):
            return None

        def create_repo(self, **kwargs):
            return None

        def upload_folder(self, **kwargs):
            raise TimeoutError("unknown upload outcome")

        def delete_repo(self, **kwargs):
            deleted.append(kwargs["repo_id"])

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)

    with pytest.raises(HuggingFaceExternalOperationAmbiguousError) as captured:
        upload_to_huggingface(
            database=database,
            settings=settings,
            payload=payload,
            job_id=job["id"],
            worker_id="hf-worker",
        )
    assert captured.value.phase == "upload"
    assert deleted == []


def test_pc_key_delivery_checks_lease_immediately_before_remote_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, database, dataset, profile, _ = _registered(tmp_path)
    _pass_gate(database, dataset, profile)
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("test host key", encoding="utf-8")
    private_key = tmp_path / "id_ed25519"
    private_key.write_text("test private key", encoding="utf-8")
    settings = replace(
        settings,
        ssh_known_hosts_path=known_hosts,
        ssh_private_key_path=private_key,
    )
    payload = {
        "dataset_id": dataset["id"],
        "fingerprint": dataset["fingerprint"],
        "storage_area": dataset["storage_area"],
        "relative_path": dataset["relative_path"],
        "host": "192.168.0.51",
        "port": 22,
        "username": "researcher",
        "destination": "~/pick",
    }
    job, _ = database.create_job(
        kind="datasets.copy_pc_key",
        queue_name="io",
        profile_id=profile["id"],
        payload=payload,
        idempotency_key="pc-key-lease",
    )
    database.claim_job(job["id"], worker_id="copy-worker", lease_seconds=120)

    class FakeSftp:
        renamed = False

        def normalize(self, value):
            return "/home/researcher"

        def lstat(self, path):
            raise FileNotFoundError(path)

        def mkdir(self, path):
            return None

        def rename(self, source, target):
            self.renamed = True

        def listdir_attr(self, path):
            return []

        def rmdir(self, path):
            return None

        def close(self):
            return None

    fake_sftp = FakeSftp()

    class FakeClient:
        def load_host_keys(self, path):
            return None

        def set_missing_host_key_policy(self, policy):
            return None

        def connect(self, **kwargs):
            return None

        def open_sftp(self):
            return fake_sftp

        def close(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "paramiko",
        SimpleNamespace(SSHClient=FakeClient, RejectPolicy=object),
    )
    monkeypatch.setattr(delivery_module, "_sftp_tree", lambda *args, **kwargs: (1, 10))
    monkeypatch.setattr(
        database,
        "assert_job_lease",
        lambda *args, **kwargs: (_ for _ in ()).throw(JobLeaseLostError(job["id"])),
    )

    with pytest.raises(JobLeaseLostError):
        copy_to_pc_with_key(
            database=database,
            settings=settings,
            payload=payload,
            job_id=job["id"],
            worker_id="copy-worker",
        )
    assert fake_sftp.renamed is False
