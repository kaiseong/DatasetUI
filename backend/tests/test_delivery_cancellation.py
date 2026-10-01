from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from datasetui.database import JobCancellationConflictError
from datasetui.job_cancellation import JobCancellationRequested
from datasetui.delivery import export_to_nas, upload_to_huggingface
from test_delivery import _registered, _pass_gate


@pytest.mark.parametrize("kind", ["datasets.export_nas", "datasets.upload_hf"])
def test_cancel_wins_before_publication(tmp_path: Path, monkeypatch, kind: str):
    settings, database, dataset, profile, _ = _registered(tmp_path)
    _pass_gate(database, dataset, profile)
    settings = replace(settings, hf_write_token="test-token")
    payload = {key: dataset[key] for key in ("fingerprint", "storage_area", "relative_path")}
    payload.update(dataset_id=dataset["id"], output_name="cancel-boundary", repo_name="cancel-boundary", visibility="private")
    job, _ = database.create_job(kind=kind, queue_name="io", profile_id=profile["id"], payload=payload, idempotency_key="cancel-boundary")
    database.claim_job(job["id"], worker_id="worker", lease_seconds=120)
    original = database.begin_job_finalization
    calls = []

    def cancel_then_finalize(*args, **kwargs):
        database.request_job_cancellation(job["id"], profile_id=profile["id"])
        original(*args, **kwargs)

    class FakeApi:
        def __init__(self, **kwargs):
            pass

        def create_repo(self, **kwargs):
            calls.append("create")

        def upload_folder(self, **kwargs):
            calls.append("upload")
            return SimpleNamespace(commit_url="https://example.invalid/commit")

    monkeypatch.setattr(database, "begin_job_finalization", cancel_then_finalize)
    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    run = export_to_nas if kind == "datasets.export_nas" else upload_to_huggingface
    with pytest.raises(JobCancellationRequested):
        run(database=database, settings=settings, payload=payload, job_id=job["id"], worker_id="worker")
    assert calls == []
    assert not (settings.nas_root / "exports/cancel-boundary").exists()
    assert not list((settings.nas_root / "exports").glob(".incoming-*"))
    assert database.get_job(job["id"])["cancellation_requested"]


def test_hf_publication_wins_rejects_late_cancellation(tmp_path: Path, monkeypatch):
    settings, database, dataset, profile, _ = _registered(tmp_path)
    _pass_gate(database, dataset, profile)
    settings = replace(settings, hf_write_token="test-token")
    payload = {key: dataset[key] for key in ("fingerprint", "storage_area", "relative_path")}
    payload.update(dataset_id=dataset["id"], repo_name="guard-boundary", visibility="private")
    job, _ = database.create_job(kind="datasets.upload_hf", queue_name="io", profile_id=profile["id"], payload=payload, idempotency_key="guard-boundary")
    database.claim_job(job["id"], worker_id="worker", lease_seconds=120)

    class FakeApi:
        def __init__(self, **kwargs):
            pass

        def create_repo(self, **kwargs):
            with pytest.raises(JobCancellationConflictError):
                database.request_job_cancellation(job["id"], profile_id=profile["id"])

        def upload_folder(self, **kwargs):
            return SimpleNamespace(commit_url="https://example.invalid/commit")

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    result = upload_to_huggingface(database=database, settings=settings, payload=payload, job_id=job["id"], worker_id="worker")
    assert result["repo_id"].endswith("/guard-boundary")
    assert database.get_job(job["id"])["cancellation_guarded_at"]

