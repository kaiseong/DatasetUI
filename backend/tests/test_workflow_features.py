"""Regression coverage for automatic delivery and shared-library operations."""

from pathlib import Path

from test_delivery import _registered, _pass_gate
from dataclasses import replace
from types import SimpleNamespace
import json
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from datasetui.api import create_router
from datasetui.delivery_workflow import (
    create_delivery_workflow,
    reconcile_deliveries,
    recover_workflows,
    run_delivery_preflight,
)
from datasetui.library_operations import LibraryOperationError, run_library_operation
from datasetui.queueing import RecordingDispatcher
from test_dataset_trash import _environment, _trash, _restore


def test_delivery_waits_for_export_gate(tmp_path: Path):
    from datasetui.delivery_workflow import (
        create_delivery_workflow,
        reconcile_deliveries,
    )
    from datasetui.queueing import RecordingDispatcher

    settings, database, dataset, profile, _ = _registered(tmp_path)
    dispatcher = RecordingDispatcher()
    job, created = create_delivery_workflow(
        database,
        dispatcher,
        settings,
        kind="datasets.export_nas",
        profile_id=profile["id"],
        payload={
            "dataset_id": dataset["id"],
            "fingerprint": dataset["fingerprint"],
            "storage_area": dataset["storage_area"],
            "relative_path": dataset["relative_path"],
            "output_name": "exported",
        },
        idempotency_key="delivery",
    )
    assert created and job["status"] == "queued"
    assert job["validation_job_id"]
    assert dispatcher.enqueued == [(job["validation_job_id"], "cpu")]
    assert database.claim_job(job["id"], worker_id="early") is None
    validation = database.get_job(job["validation_job_id"])
    database.claim_job(validation["id"], worker_id="validator")
    database.succeed_job(validation["id"], {"passed": False}, worker_id="validator")
    reconcile_deliveries(database, dispatcher, settings)
    assert database.get_job(job["id"])["status"] == "failed"
    assert dispatcher.enqueued == [(validation["id"], "cpu")]


def test_valid_gate_is_checked_by_preflight_before_delivery(tmp_path: Path):
    from datasetui.delivery_workflow import (
        create_delivery_workflow,
        reconcile_deliveries,
        run_delivery_preflight,
    )
    from datasetui.queueing import RecordingDispatcher

    settings, database, dataset, profile, _ = _registered(tmp_path)
    _pass_gate(database, dataset, profile)
    dispatcher = RecordingDispatcher()
    payload = {
        "dataset_id": dataset["id"],
        "fingerprint": dataset["fingerprint"],
        "storage_area": dataset["storage_area"],
        "relative_path": dataset["relative_path"],
        "output_name": "exported",
    }
    job, _ = create_delivery_workflow(
        database,
        dispatcher,
        settings,
        kind="datasets.export_nas",
        profile_id=profile["id"],
        payload=payload,
        idempotency_key="delivery",
    )
    preflight = database.get_job(job["validation_job_id"])
    database.claim_job(preflight["id"], worker_id="validator")
    result = run_delivery_preflight(
        database, settings, preflight["payload"], preflight["id"], "validator"
    )
    assert result["passed"] is True and result["reused"] is True
    database.succeed_job(preflight["id"], result, worker_id="validator")
    reconcile_deliveries(database, dispatcher, settings)
    assert dispatcher.enqueued[-1] == (job["id"], "io")
    assert database.claim_job(job["id"], worker_id="delivery") is not None


def _client(database, dispatcher, settings):
    app = FastAPI()
    app.include_router(create_router(database, dispatcher, settings=settings))
    return TestClient(app)


def _payload(dataset, **extra):
    return {
        "dataset_id": dataset["id"],
        "fingerprint": dataset["fingerprint"],
        "storage_area": dataset["storage_area"],
        "relative_path": dataset["relative_path"],
        **extra,
    }


@pytest.mark.parametrize(
    "method,extra",
    [
        ("nas", {"output_name": "copy"}),
        ("huggingface", {"repo_name": "copy", "visibility": "private"}),
        (
            "pc/key",
            {
                "host": "192.168.0.51",
                "port": 22,
                "username": "robot",
                "destination": "~/copy",
            },
        ),
        (
            "pc/password",
            {
                "host": "192.168.0.51",
                "port": 22,
                "username": "robot",
                "destination": "~/copy",
                "password": "never-persist-me",
            },
        ),
    ],
)
def test_every_delivery_accepts_without_existing_validation(
    tmp_path, monkeypatch, method, extra
):
    settings, database, dataset, profile, _ = _registered(tmp_path)
    settings = replace(settings, credential_redis_url="redis://memory-only")
    secrets = {}

    class Vault:
        def put(self, key, value):
            secrets.setdefault(key, value)

        def exists(self, key):
            return key in secrets

        def delete(self, key):
            secrets.pop(key, None)

    monkeypatch.setattr(
        "datasetui.credential_store.credential_store", lambda _: Vault()
    )
    dispatcher = RecordingDispatcher()
    client = _client(database, dispatcher, settings)
    body = {"profile_id": profile["id"], "idempotency_key": "intent", **extra}
    url = f"/api/v1/datasets/{dataset['id']}/deliveries/{method}"
    response = client.post(url, json=body)
    assert response.status_code == 202, response.text
    parent = response.json()
    assert parent["wait_reason"] == "validation"
    assert len(dispatcher.enqueued) == 1 and dispatcher.enqueued[0][1] == "cpu"
    repeated = client.post(url, json=body)
    assert repeated.status_code == 200 and repeated.json()["id"] == parent["id"]
    assert len(database.list_jobs()) == 2
    serialized = json.dumps(database.list_jobs()) + json.dumps(
        database.list_job_events(parent["id"])
    )
    assert "never-persist-me" not in serialized and "password" not in parent["payload"]
    other = database.create_profile("Other")
    denied = client.post(
        f"/api/v1/jobs/{parent['id']}/cancel", json={"profile_id": other["id"]}
    )
    assert denied.status_code == 403
    cancelled = client.post(
        f"/api/v1/jobs/{parent['id']}/cancel", json={"profile_id": profile["id"]}
    )
    assert cancelled.json()["status"] == "cancelled"
    assert database.get_job(parent["validation_job_id"])["status"] == "cancelled"
    assert not secrets


def test_shared_validation_is_retained_until_last_delivery_cancelled(tmp_path):
    settings, database, dataset, profile, _ = _registered(tmp_path)
    dispatcher = RecordingDispatcher()
    jobs = [
        create_delivery_workflow(
            database,
            dispatcher,
            settings,
            kind="datasets.export_nas",
            profile_id=profile["id"],
            payload=_payload(dataset, output_name=f"copy-{i}"),
            idempotency_key=f"intent-{i}",
        )[0]
        for i in range(2)
    ]
    assert jobs[0]["validation_job_id"] == jobs[1]["validation_job_id"]
    database.request_job_cancellation(jobs[0]["id"], profile_id=profile["id"])
    reconcile_deliveries(database, dispatcher, settings)
    assert database.get_job(jobs[1]["validation_job_id"])["status"] == "queued"
    database.request_job_cancellation(jobs[1]["id"], profile_id=profile["id"])
    reconcile_deliveries(database, dispatcher, settings)
    assert database.get_job(jobs[1]["validation_job_id"])["status"] == "cancelled"


def test_changed_content_does_not_reuse_cached_gate(tmp_path, monkeypatch):
    settings, database, dataset, profile, source = _registered(tmp_path)
    _pass_gate(database, dataset, profile)
    dispatcher = RecordingDispatcher()
    job, _ = create_delivery_workflow(
        database,
        dispatcher,
        settings,
        kind="datasets.export_nas",
        profile_id=profile["id"],
        payload=_payload(dataset, output_name="copy"),
        idempotency_key="intent",
    )
    (source / "changed.bin").write_bytes(b"changed after validation")
    check = database.claim_job(job["validation_job_id"], worker_id="worker")
    calls = []

    def validate(**kwargs):
        calls.append(kwargs)
        return {"passed": False, "mode": "export_gate"}

    monkeypatch.setattr("datasetui.validation.validate_registered_dataset", validate)
    result = run_delivery_preflight(
        database, settings, check["payload"], check["id"], "worker"
    )
    assert calls and not result["passed"]
    database.succeed_job(check["id"], result, worker_id="worker")
    recover_workflows(database, dispatcher, settings)
    assert database.get_job(job["id"])["status"] == "failed"
    assert not any(queue == "io" for _, queue in dispatcher.enqueued)


def test_queue_recovery_does_not_need_browser(tmp_path):
    settings, database, dataset, profile, _ = _registered(tmp_path)
    dispatcher = RecordingDispatcher(available=False)
    with pytest.raises(ConnectionError):
        create_delivery_workflow(
            database,
            dispatcher,
            settings,
            kind="datasets.export_nas",
            profile_id=profile["id"],
            payload=_payload(dataset, output_name="copy"),
            idempotency_key="intent",
        )
    dispatcher.available = True
    recover_workflows(database, dispatcher, settings)
    check = next(
        job
        for job in database.list_jobs()
        if job["kind"] == "datasets.delivery_preflight"
    )
    assert dispatcher.enqueued == [(check["id"], "cpu")]
    database.claim_job(check["id"], worker_id="worker")
    database.succeed_job(check["id"], {"passed": True}, worker_id="worker")
    recover_workflows(database, dispatcher, settings)
    parent = next(
        job for job in database.list_jobs() if job["kind"] == "datasets.export_nas"
    )
    assert dispatcher.enqueued[-1] == (parent["id"], "io")


def _empty_body(profile, dataset, key="purge"):
    return {
        "profile_id": profile["id"],
        "idempotency_key": key,
        "confirmed": True,
        "items": [
            {
                "dataset_id": dataset["id"],
                "expected_fingerprint": dataset["fingerprint"],
            }
        ],
    }


def test_trash_confirmation_cancellation_and_restore(tmp_path):
    settings, database, profile, dataset, client = _environment(tmp_path)
    assert _trash(client, profile, dataset).status_code == 200
    body = _empty_body(profile, dataset)
    assert (
        client.post(
            "/api/v1/dataset-trash/empty", json={**body, "confirmed": False}
        ).status_code
        == 422
    )
    response = client.post("/api/v1/dataset-trash/empty", json=body)
    assert response.status_code == 202, response.text
    job = response.json()
    assert database.get_dataset_trash(dataset["id"])["state"] == "purge_queued"
    assert _restore(client, profile, dataset).status_code == 409
    assert (
        client.post("/api/v1/dataset-trash/empty", json=body).json()["id"] == job["id"]
    )
    assert (
        client.post(
            f"/api/v1/jobs/{job['id']}/cancel", json={"profile_id": profile["id"]}
        ).json()["status"]
        == "cancelled"
    )
    assert _restore(client, profile, dataset).status_code == 200
    assert (
        settings.nas_root / "raw/team/sample/data/part.bin"
    ).read_bytes() == b"dataset-payload"


def test_trash_purge_unlinks_symlinks_without_touching_outside(tmp_path):
    settings, database, profile, dataset, client = _environment(tmp_path)
    assert _trash(client, profile, dataset).status_code == 200
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("untouched")
    root = settings.nas_root / "raw/.datasetui-trash" / dataset["id"]
    (root / "link").symlink_to(outside, target_is_directory=True)
    job = client.post(
        "/api/v1/dataset-trash/empty", json=_empty_body(profile, dataset)
    ).json()
    database.claim_job(job["id"], worker_id="worker")
    result = run_library_operation(
        job["kind"], database, settings, job["payload"], job["id"], "worker"
    )
    assert result["deleted"] == [dataset["id"]] and not result["failed"]
    assert not root.exists() and (outside / "keep").read_text() == "untouched"
    database.succeed_job(job["id"], result, worker_id="worker")
    assert not database.list_dataset_trash(profile_id=profile["id"])
    assert _restore(client, profile, dataset).status_code == 409


def test_trash_replaced_root_is_not_deleted_and_requires_recovery(tmp_path):
    settings, database, profile, dataset, client = _environment(tmp_path)
    assert _trash(client, profile, dataset).status_code == 200
    root = settings.nas_root / "raw/.datasetui-trash" / dataset["id"]
    original = root.with_name("preserved-original")
    root.rename(original)
    root.mkdir()
    (root / "keep").write_text("replacement")
    job = client.post(
        "/api/v1/dataset-trash/empty", json=_empty_body(profile, dataset)
    ).json()
    database.claim_job(job["id"], worker_id="worker")
    result = run_library_operation(
        job["kind"], database, settings, job["payload"], job["id"], "worker"
    )
    assert not result["deleted"] and result["partial_failure"]
    assert (root / "keep").read_text() == "replacement" and original.exists()
    assert database.get_dataset_trash(dataset["id"])["state"] == "recovery_required"


def test_inflight_permanent_delete_is_not_replayed_after_crash(tmp_path):
    settings, database, profile, dataset, client = _environment(tmp_path)
    assert _trash(client, profile, dataset).status_code == 200
    job = client.post(
        "/api/v1/dataset-trash/empty", json=_empty_body(profile, dataset)
    ).json()
    database.claim_job(job["id"], worker_id="crashed", lease_seconds=120)
    database.begin_job_finalization(job["id"], worker_id="crashed")
    with database.connect() as connection:
        connection.execute(
            "UPDATE dataset_purge_reservations SET state='purging' WHERE job_id=?",
            (job["id"],),
        )
        connection.execute(
            "UPDATE jobs SET lease_expires_at='2000-01-01T00:00:00+00:00' WHERE id=?",
            (job["id"],),
        )
    dispatcher = RecordingDispatcher()
    recover_workflows(database, dispatcher, settings)
    assert database.get_job(job["id"])["status"] == "interrupted"
    assert database.get_dataset_trash(dataset["id"])["state"] == "recovery_required"
    assert not dispatcher.enqueued


def _hf_body(profile):
    return {
        "profile_id": profile["id"],
        "confirmed": True,
        "expected_repo_id": "rainbowrobotics/sample",
        "expected_commit_sha": "a" * 40,
        "idempotency_key": "delete",
    }


@pytest.mark.parametrize("changed", [False, True])
def test_hf_delete_is_confirmed_namespace_bound_and_revision_checked(
    tmp_path, monkeypatch, changed
):
    settings, database, dataset, profile, source = _registered(tmp_path)
    settings = replace(settings, hf_write_token="server-write-token")
    client = _client(database, RecordingDispatcher(), settings)
    body = _hf_body(profile)
    assert (
        client.post(
            "/api/v1/hf/datasets/sample/delete", json={**body, "confirmed": False}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/hf/datasets/sample/delete",
            json={**body, "expected_repo_id": "other/sample"},
        ).status_code
        == 409
    )
    deleted = []

    class Api:
        def __init__(self, **kwargs):
            assert kwargs["token"] == settings.hf_write_token

        def dataset_info(self, repo):
            return SimpleNamespace(sha=("b" if changed else "a") * 40)

        def delete_repo(self, **kwargs):
            deleted.append(kwargs)

    monkeypatch.setattr("huggingface_hub.HfApi", Api)
    response = client.post("/api/v1/hf/datasets/sample/delete", json=body)
    assert response.status_code == 202
    job = response.json()
    database.claim_job(job["id"], worker_id="worker")
    if changed:
        with pytest.raises(LibraryOperationError, match="변경"):
            run_library_operation(
                job["kind"], database, settings, job["payload"], job["id"], "worker"
            )
        assert not deleted
    else:
        result = run_library_operation(
            job["kind"], database, settings, job["payload"], job["id"], "worker"
        )
        assert result["deleted"] and deleted == [
            {"repo_id": "rainbowrobotics/sample", "repo_type": "dataset"}
        ]
        assert (
            client.post(
                f"/api/v1/jobs/{job['id']}/cancel", json={"profile_id": profile["id"]}
            ).status_code
            == 409
        )
    assert source.exists() and database.get_dataset(dataset["id"])["available"]
    assert "server-write-token" not in json.dumps(database.list_jobs())


def test_hf_delete_excludes_concurrent_import_and_upload(tmp_path):
    settings, database, dataset, profile, _ = _registered(tmp_path)
    settings = replace(settings, hf_write_token="write")
    client = _client(database, RecordingDispatcher(), settings)
    assert (
        client.post(
            "/api/v1/hf/datasets/sample/delete", json=_hf_body(profile)
        ).status_code
        == 202
    )
    from datasetui.database import IdempotencyConflictError

    with pytest.raises(IdempotencyConflictError, match="대기 또는 실행"):
        database.create_hf_import_job(
            profile_id=profile["id"],
            repo_id="rainbowrobotics/sample",
            dataset_name="sample",
            requested_revision="main",
            commit_sha="a" * 40,
            expected_file_count=2,
            expected_total_bytes=20,
            idempotency_key="import",
        )
    response = client.post(
        f"/api/v1/datasets/{dataset['id']}/deliveries/huggingface",
        json={
            "profile_id": profile["id"],
            "repo_name": "sample",
            "visibility": "private",
            "idempotency_key": "upload",
        },
    )
    assert response.status_code == 409


def test_server_sort_and_search_before_limit_support_more_than_500(tmp_path):
    settings, database, dataset, profile, _ = _registered(tmp_path)
    with database.connect() as connection:
        template = dict(
            connection.execute(
                "SELECT * FROM datasets WHERE id=?", (dataset["id"],)
            ).fetchone()
        )
        columns = list(template)
        for index in range(510):
            record = {
                **template,
                "id": str(uuid.uuid4()),
                "name": f"item-{index:04d}",
                "relative_path": f"items/{index:04d}",
                "first_seen_at": f"2099-09-30T00:{index//60:02d}:{index%60:02d}+00:00",
            }
            connection.execute(
                f"INSERT INTO datasets ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                tuple(record[column] for column in columns),
            )
    client = _client(database, RecordingDispatcher(), settings)
    newest = client.get(
        "/api/v1/datasets", params={"sort": "newest", "limit": 1}
    ).json()
    assert newest[0]["name"] == "item-0509"
    found = client.get("/api/v1/datasets", params={"q": "item-0509", "limit": 1}).json()
    assert found[0]["name"] == "item-0509"
    first = client.get(
        "/api/v1/datasets", params={"sort": "name_asc", "limit": 500}
    ).json()
    rest = client.get(
        "/api/v1/datasets", params={"sort": "name_asc", "limit": 500, "offset": 500}
    ).json()
    assert (
        len(first) == 500
        and len(rest) == 11
        and len({row["id"] for row in first + rest}) == 511
    )
    assert not client.get("/api/v1/datasets", params={"q": "%", "limit": 1}).json()


def test_active_team_jobs_are_not_hidden_by_recent_terminal_jobs(tmp_path):
    settings, database, dataset, profile, _ = _registered(tmp_path)
    dispatcher = RecordingDispatcher()
    client = _client(database, dispatcher, settings)
    other = database.create_profile("Other")
    old, _ = database.create_job(
        kind="datasets.scan",
        queue_name="cpu",
        profile_id=other["id"],
        payload={"storage_areas": ["raw"]},
        idempotency_key="old",
    )
    database.mark_enqueued(old["id"], database.rq_job_id_for(old["id"]))
    dispatcher.enqueue(
        job_id=old["id"], queue_name="cpu", rq_job_id=database.rq_job_id_for(old["id"])
    )
    for index in range(205):
        job, _ = database.create_job(
            kind="phase2.smoke",
            queue_name="cpu",
            profile_id=profile["id"],
            payload={},
            idempotency_key=f"done-{index}",
        )
        database.cancel_queued_job(job["id"], profile_id=profile["id"])
    assert old["id"] not in {
        job["id"] for job in client.get("/api/v1/jobs?limit=200").json()
    }
    active = client.get("/api/v1/jobs?active_only=true&limit=200").json()
    assert (
        active[0]["id"] == old["id"]
        and active[0]["profile_name"] == "Other"
        and active[0]["queue_position"] == 1
    )


def test_expired_password_fails_waiting_job_without_persisted_secret(
    tmp_path, monkeypatch
):
    settings, database, dataset, profile, _ = _registered(tmp_path)
    settings = replace(settings, credential_redis_url="redis://memory")
    secrets = {}

    class Vault:
        def put(self, key, value):
            secrets[key] = value

        def exists(self, key):
            return key in secrets

        def delete(self, key):
            secrets.pop(key, None)

    monkeypatch.setattr(
        "datasetui.credential_store.credential_store", lambda _: Vault()
    )
    job, _ = create_delivery_workflow(
        database,
        RecordingDispatcher(),
        settings,
        kind="datasets.copy_pc_password",
        profile_id=profile["id"],
        payload=_payload(
            dataset,
            host="192.168.0.51",
            port=22,
            username="robot",
            destination="~/copy",
        ),
        idempotency_key="password",
        password="secret",
    )
    secrets.clear()  # Memory-store restart or TTL expiry.
    reconcile_deliveries(database, RecordingDispatcher(), settings)
    assert database.get_job(job["id"])["error_code"] == "credential_unavailable"
    assert database.get_job(job["validation_job_id"])["status"] == "cancelled"
    assert "secret" not in json.dumps(database.list_jobs())


def test_trash_purge_snapshot_excludes_datasets_trashed_later(tmp_path):
    import shutil
    from datasetui.datasets import inspect_dataset

    settings, database, profile, dataset, client = _environment(tmp_path)
    assert _trash(client, profile, dataset).status_code == 200
    job = client.post(
        "/api/v1/dataset-trash/empty", json=_empty_body(profile, dataset)
    ).json()
    later_root = settings.nas_root / "raw/later"
    shutil.copytree(
        settings.nas_root / "raw/.datasetui-trash" / dataset["id"], later_root
    )
    candidate = inspect_dataset(
        area_root=settings.nas_root / "raw", storage_area="raw", relative_path="later"
    )
    generation = database.begin_dataset_scan("raw")
    database.synchronize_datasets(
        storage_area="raw", records=[candidate.as_record()], scan_generation=generation
    )
    later = database.list_datasets()[0]
    assert _trash(client, profile, later).status_code == 200
    database.claim_job(job["id"], worker_id="worker")
    result = run_library_operation(
        job["kind"], database, settings, job["payload"], job["id"], "worker"
    )
    assert result["deleted"] == [dataset["id"]]
    assert (settings.nas_root / "raw/.datasetui-trash" / later["id"]).exists()
    assert database.get_dataset_trash(later["id"])["state"] == "trashed"


@pytest.mark.parametrize("status", [403, 404, 500])
def test_hf_access_errors_never_claim_remote_delete_succeeded(
    tmp_path, monkeypatch, status
):
    import httpx
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    settings, database, _, profile, _ = _registered(tmp_path)
    settings = replace(settings, hf_write_token="write")
    client = _client(database, RecordingDispatcher(), settings)
    job = client.post(
        "/api/v1/hf/datasets/sample/delete", json=_hf_body(profile)
    ).json()
    calls = []

    class Api:
        def __init__(self, **kwargs):
            pass

        def dataset_info(self, repo):
            response = httpx.Response(
                status, request=httpx.Request("GET", "https://huggingface.co/test")
            )
            error = RepositoryNotFoundError if status == 404 else HfHubHTTPError
            raise error("secret server detail", response=response)

        def delete_repo(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr("huggingface_hub.HfApi", Api)
    database.claim_job(job["id"], worker_id="worker")
    with pytest.raises(LibraryOperationError) as failure:
        run_library_operation(
            job["kind"], database, settings, job["payload"], job["id"], "worker"
        )
    assert failure.value.code == "hf_delete_denied" and "secret" not in str(
        failure.value
    )
    assert not calls


def test_credential_store_refuses_persistence_and_consumes_once(tmp_path, monkeypatch):
    from datasetui.credential_store import CredentialStore, CredentialUnavailableError

    settings, _, _, _, _ = _registered(tmp_path)
    settings = replace(
        settings, credential_redis_url="redis://temporary", credential_ttl_seconds=60
    )

    class Redis:
        config = {b"save": b"", b"appendonly": b"no"}
        values = {}

        def config_get(self, *args):
            return self.config

        def set(self, key, value, **kwargs):
            assert kwargs == {"ex": 60, "nx": True}
            self.values.setdefault(key, value.encode())

        def getdel(self, key):
            return self.values.pop(key, None)

        def delete(self, key):
            self.values.pop(key, None)

        def exists(self, key):
            return key in self.values

    redis = Redis()
    monkeypatch.setattr("redis.Redis.from_url", lambda *args, **kwargs: redis)
    store = CredentialStore(settings)
    store.put("job", "first")
    store.put("job", "cannot-replace-first")
    assert store.exists("job") and store.take("job") == "first"
    with pytest.raises(CredentialUnavailableError):
        store.take("job")
    redis.config = {b"save": b"3600 1", b"appendonly": b"no"}
    with pytest.raises(CredentialUnavailableError):
        store.put("other", "secret")
    with pytest.raises(CredentialUnavailableError):
        CredentialStore(replace(settings, credential_redis_url=settings.redis_url))


def test_password_worker_exception_does_not_enter_log_or_rq_traceback(
    tmp_path, monkeypatch, caplog
):
    from datasetui.tasks import run_job
    from datasetui.config import Settings
    import traceback

    settings, database, _, profile, _ = _registered(tmp_path)
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: settings))
    job, _ = database.create_job(
        kind="datasets.copy_pc_password",
        queue_name="io",
        profile_id=profile["id"],
        payload={},
        idempotency_key="secret-failure",
    )

    def failing(*args, **kwargs):
        raise RuntimeError("one-time-secret /private/address")

    monkeypatch.setattr("datasetui.tasks.run_registered_job", failing)
    with pytest.raises(RuntimeError) as failure:
        run_job(job["id"])
    assert "one-time-secret" not in "".join(traceback.format_exception(failure.value))
    assert "one-time-secret" not in caplog.text
    assert "one-time-secret" not in json.dumps(database.list_jobs())


def test_password_worker_uses_verified_staging_and_publication_guard(
    tmp_path, monkeypatch
):
    from datasetui.config import Settings
    from datasetui.jobs import run_registered_job

    settings, database, dataset, profile, source = _registered(tmp_path)
    settings = replace(settings, credential_redis_url="redis://memory")
    _pass_gate(database, dataset, profile)
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: settings))
    secrets = {}

    class Vault:
        def put(self, key, value):
            secrets.setdefault(key, value)

        def exists(self, key):
            return key in secrets

        def take(self, key):
            return secrets.pop(key)

        def delete(self, key):
            secrets.pop(key, None)

    monkeypatch.setattr(
        "datasetui.credential_store.credential_store", lambda _: Vault()
    )
    dispatcher = RecordingDispatcher()
    job, _ = create_delivery_workflow(
        database,
        dispatcher,
        settings,
        kind="datasets.copy_pc_password",
        profile_id=profile["id"],
        payload=_payload(
            dataset,
            host="192.168.0.51",
            port=22,
            username="robot",
            destination="~/copy",
        ),
        idempotency_key="password",
        password="one-time-secret",
    )
    check = database.claim_job(job["validation_job_id"], worker_id="validator")
    result = run_delivery_preflight(
        database, settings, check["payload"], check["id"], "validator"
    )
    database.succeed_job(check["id"], result, worker_id="validator")
    reconcile_deliveries(database, dispatcher, settings)
    database.claim_job(job["id"], worker_id="sender")

    def transfer(**kwargs):
        assert kwargs["password"] == "one-time-secret"
        assert kwargs["source"] != source and kwargs["source"].is_dir()
        kwargs["lease_check"]()
        kwargs["finalize"]()
        return {"ok": True, "files": 1, "bytes": 10}

    monkeypatch.setattr("datasetui.delivery._copy_to_pc", transfer)
    delivered = run_registered_job(
        job["kind"], job["payload"], job_id=job["id"], worker_id="sender"
    )
    assert delivered["ok"] and not secrets
    assert database.get_job(job["id"])["cancellation_guarded_at"]
    assert "one-time-secret" not in json.dumps(database.list_jobs())


def test_purge_stops_if_open_root_is_moved_out_of_trash(tmp_path):
    from datasetui.dataset_trash import purge_dataset_from_trash, DatasetTrashPathError

    settings, database, profile, dataset, client = _environment(tmp_path)
    assert _trash(client, profile, dataset).status_code == 200
    record = database.get_dataset_trash(dataset["id"])
    root = settings.nas_root / "raw/.datasetui-trash" / dataset["id"]
    moved = settings.nas_root / "raw/recovered-outside-trash"
    calls = []

    def rename_after_open():
        if not calls:
            root.rename(moved)
            root.mkdir()
            (root / "replacement").write_text("untouched")
            calls.append(True)

    with pytest.raises(DatasetTrashPathError):
        purge_dataset_from_trash(
            settings.nas_root, record, lease_check=rename_after_open
        )
    assert (moved / "data/part.bin").read_bytes() == b"dataset-payload"
    assert (root / "replacement").read_text() == "untouched"


def test_cancelled_delivery_stays_successfully_cancelled_if_orphan_check_finishes(
    tmp_path, monkeypatch
):
    from datasetui.database import Database

    settings, database, dataset, profile, _ = _registered(tmp_path)
    dispatcher = RecordingDispatcher()
    parent, _ = create_delivery_workflow(
        database,
        dispatcher,
        settings,
        kind="datasets.export_nas",
        profile_id=profile["id"],
        payload=_payload(dataset, output_name="copy"),
        idempotency_key="intent",
    )
    database.request_job_cancellation(parent["id"], profile_id=profile["id"])
    original = Database.get_job
    switched = []

    def finish_after_snapshot(self, job_id):
        if (
            self.path == database.path
            and job_id == parent["validation_job_id"]
            and not switched
        ):
            switched.append(True)
            self.claim_job(job_id, worker_id="finishing")
            self.succeed_job(job_id, {"passed": True}, worker_id="finishing")
        return original(self, job_id)

    monkeypatch.setattr(Database, "get_job", finish_after_snapshot)
    reconcile_deliveries(database, dispatcher, settings)
    assert database.get_job(parent["id"])["status"] == "cancelled"


def test_credential_store_cannot_use_another_database_on_job_redis(tmp_path):
    from datasetui.credential_store import CredentialStore, CredentialUnavailableError

    settings, _, _, _, _ = _registered(tmp_path)
    with pytest.raises(CredentialUnavailableError):
        CredentialStore(
            replace(
                settings,
                redis_url="redis://same-host:6379/0",
                credential_redis_url="redis://same-host/1",
            )
        )
