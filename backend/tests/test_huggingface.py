from __future__ import annotations

import json
import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx
from fastapi import FastAPI

from datasetui.api import _hf_dataset_status, create_router
from datasetui.config import Settings
from datasetui.database import Database
from datasetui.hf_errors import (
    HuggingFaceImportConflictError,
    HuggingFaceImportValidationError,
)
from datasetui.huggingface import HuggingFaceGateway, import_huggingface_dataset
from datasetui.jobs import _validate_hf_import_payload
from datasetui.queueing import RecordingDispatcher


SHA = "a" * 40
NEW_SHA = "b" * 40


class LocalClient:
    def __init__(self, app: FastAPI):
        self.app = app

    def get(self, path: str) -> httpx.Response:
        return self.request("GET", path)

    def post(self, path: str, *, json: dict) -> httpx.Response:
        return self.request("POST", path, json=json)

    def request(self, method: str, path: str, **kwargs) -> httpx.Response:
        async def send() -> httpx.Response:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self.app),
                base_url="http://testserver",
            ) as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(send())


class FakeGateway:
    def __init__(self, *, commit_sha: str = SHA):
        self.commit_sha = commit_sha
        self.queries: list[tuple[str | None, int]] = []

    def list_datasets(self, *, query: str | None, limit: int):
        self.queries.append((query, limit))
        return [
            {
                "repo_id": "rainbowrobotics/pick-cup",
                "name": "pick-cup",
                "private": True,
                "gated": False,
                "downloads": 12,
                "likes": 2,
                "last_modified": "2026-09-03T00:00:00Z",
                "latest_commit_sha": self.commit_sha,
                "tags": ["lerobot"],
            }
        ]

    def list_revisions(self, dataset_name: str):
        assert dataset_name == "pick-cup"
        return [{"name": "main", "kind": "branch", "commit_sha": self.commit_sha}]

    def resolve_revision(self, dataset_name: str, revision: str):
        assert dataset_name == "pick-cup"
        return {
            "repo_id": "rainbowrobotics/pick-cup",
            "requested_revision": revision,
            "commit_sha": self.commit_sha,
            "file_count": 2,
            "total_bytes": 512,
        }

    def download_snapshot(self, *, repo_id: str, commit_sha: str, local_dir: Path):
        assert repo_id == "rainbowrobotics/pick-cup"
        assert commit_sha == self.commit_sha
        (local_dir / "meta").mkdir(parents=True)
        (local_dir / "data").mkdir()
        (local_dir / "meta" / "info.json").write_text(
            json.dumps(
                {
                    "codebase_version": "v3.0",
                    "features": {},
                    "total_episodes": 1,
                    "total_frames": 3,
                }
            ),
            encoding="utf-8",
        )
        (local_dir / "data" / "part-000.parquet").write_bytes(b"dataset")
        return local_dir


class InvalidDatasetGateway(FakeGateway):
    def download_snapshot(self, *, repo_id: str, commit_sha: str, local_dir: Path):
        (local_dir / "meta").mkdir(parents=True)
        (local_dir / "meta" / "info.json").write_text(
            '{"codebase_version":"v3.0","features":{}}',
            encoding="utf-8",
        )
        return local_dir


def _settings(tmp_path: Path) -> Settings:
    nas_root = tmp_path / "nas"
    for directory in ("raw", "derived", "exports", "manifests"):
        (nas_root / directory).mkdir(parents=True)
    for directory in ("cache", "staging", "jobs"):
        (tmp_path / directory).mkdir()
    return Settings(
        database_path=tmp_path / "datasetui.sqlite3",
        redis_url="redis://unused",
        allowed_origins=("https://192.168.0.3",),
        nas_root=nas_root,
        cache_root=tmp_path / "cache",
        staging_root=tmp_path / "staging",
        jobs_root=tmp_path / "jobs",
    )


def _client(tmp_path: Path):
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    dispatcher = RecordingDispatcher()
    gateway = FakeGateway()
    app = FastAPI()
    app.include_router(
        create_router(
            database,
            dispatcher,
            settings=settings,
            hf_gateway=gateway,
        )
    )
    return LocalClient(app), database, dispatcher, gateway, settings


def test_hf_gateway_uses_current_list_datasets_contract() -> None:
    captured: dict = {}

    class Api:
        def list_datasets(self, **kwargs):
            captured.update(kwargs)
            return [
                SimpleNamespace(
                    id="rainbowrobotics/pick-cup",
                    private=False,
                    gated=False,
                    downloads=3,
                    likes=1,
                    last_modified=None,
                    sha=SHA,
                    tags=["lerobot"],
                ),
                SimpleNamespace(
                    id="rainbowrobotics/safe_trailing_underscore_",
                    private=False,
                    gated=False,
                    downloads=0,
                    likes=0,
                    last_modified=None,
                    sha=NEW_SHA,
                    tags=[],
                ),
            ]

    gateway = HuggingFaceGateway()
    gateway.api = Api()

    datasets = gateway.list_datasets(query="pick", limit=5)
    assert [item["name"] for item in datasets] == [
        "pick-cup",
        "safe_trailing_underscore_",
    ]
    assert captured == {
        "author": "rainbowrobotics",
        "search": "pick",
        "sort": "last_modified",
        "limit": 5,
        "token": None,
    }


def test_hf_discovery_and_revision_endpoints_are_server_scoped(tmp_path: Path) -> None:
    client, _, _, gateway, _ = _client(tmp_path)

    datasets = client.get("/api/v1/hf/datasets?q=pick").json()
    assert gateway.queries == [("pick", 100)]
    assert datasets[0]["repo_id"] == "rainbowrobotics/pick-cup"
    assert datasets[0]["status"] == "not_downloaded"
    assert "token" not in client.get("/api/v1/hf/datasets").text.lower()

    revisions = client.get("/api/v1/hf/datasets/pick-cup/revisions")
    assert revisions.status_code == 200
    assert revisions.json() == [{"name": "main", "kind": "branch", "commit_sha": SHA}]


def test_hf_import_endpoint_pins_revision_and_dispatches_opaque_job(
    tmp_path: Path,
) -> None:
    client, database, dispatcher, _, _ = _client(tmp_path)
    profile_id = client.post("/api/v1/profiles", json={"name": "Researcher"}).json()[
        "id"
    ]
    response = client.post(
        "/api/v1/hf/imports",
        json={
            "profile_id": profile_id,
            "dataset_name": "pick-cup",
            "requested_revision": "main",
            "commit_sha": SHA,
            "idempotency_key": "hf-pick-cup-main",
        },
    )
    assert response.status_code == 202
    job = response.json()
    assert job["kind"] == "hf.import"
    assert job["payload"]["repo_id"] == "rainbowrobotics/pick-cup"
    assert set(job["payload"]).isdisjoint({"token", "password", "api_key"})
    assert dispatcher.enqueued == [(job["id"], "io")]
    assert database.list_hf_sources()[0]["desired_commit_sha"] == SHA

    generic = client.post(
        "/api/v1/jobs",
        json={
            "kind": "hf.import",
            "profile_id": profile_id,
            "payload": {},
            "idempotency_key": "bypass",
        },
    )
    assert generic.status_code == 422


def test_moved_revision_is_rejected_before_job_creation(tmp_path: Path) -> None:
    client, database, dispatcher, _, _ = _client(tmp_path)
    profile_id = client.post("/api/v1/profiles", json={"name": "Researcher"}).json()[
        "id"
    ]
    response = client.post(
        "/api/v1/hf/imports",
        json={
            "profile_id": profile_id,
            "dataset_name": "pick-cup",
            "requested_revision": "main",
            "commit_sha": NEW_SHA,
            "idempotency_key": "moved-ref",
        },
    )
    assert response.status_code == 409
    assert database.list_jobs() == []
    assert dispatcher.enqueued == []


def test_import_publishes_immutable_revision_and_confirmed_pointer(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    profile = database.create_profile("Researcher")
    job, _ = database.create_hf_import_job(
        profile_id=profile["id"],
        repo_id="rainbowrobotics/pick-cup",
        dataset_name="pick-cup",
        requested_revision="main",
        commit_sha=SHA,
        expected_file_count=2,
        expected_total_bytes=512,
        idempotency_key="publish",
    )
    database.claim_job(job["id"], worker_id="worker-1", lease_seconds=120)

    result = import_huggingface_dataset(
        database=database,
        settings=settings,
        payload=database.get_job(job["id"])["payload"],
        job_id=job["id"],
        worker_id="worker-1",
        gateway=FakeGateway(),
    )

    revision = settings.nas_root / "raw/hf/rainbowrobotics/pick-cup/revisions" / SHA
    assert result["current"] is True
    assert (revision / "meta/info.json").is_file()
    pointer = json.loads((revision.parent.parent / "current.json").read_text())
    assert pointer["commit_sha"] == SHA
    assert pointer["generation"] == 1
    source = database.list_hf_sources()[0]
    assert source["pointer_confirmed"] is True
    remote = {"latest_commit_sha": SHA}
    assert (
        _hf_dataset_status(
            remote=remote,
            source=source,
            latest_job=None,
            raw_root=settings.nas_root / "raw",
        )
        == "ready"
    )
    (revision.parent.parent / "current.json").write_text("{}", encoding="utf-8")
    assert (
        _hf_dataset_status(
            remote=remote,
            source=source,
            latest_job=None,
            raw_root=settings.nas_root / "raw",
        )
        == "incomplete"
    )


def test_older_import_cannot_replace_a_newer_requested_generation(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    profile = database.create_profile("Researcher")
    older, _ = database.create_hf_import_job(
        profile_id=profile["id"],
        repo_id="rainbowrobotics/pick-cup",
        dataset_name="pick-cup",
        requested_revision="old",
        commit_sha=SHA,
        expected_file_count=2,
        expected_total_bytes=512,
        idempotency_key="older",
    )
    database.claim_job(older["id"], worker_id="worker-old", lease_seconds=120)
    database.create_hf_import_job(
        profile_id=profile["id"],
        repo_id="rainbowrobotics/pick-cup",
        dataset_name="pick-cup",
        requested_revision="new",
        commit_sha=NEW_SHA,
        expected_file_count=2,
        expected_total_bytes=512,
        idempotency_key="newer",
    )

    result = import_huggingface_dataset(
        database=database,
        settings=settings,
        payload=database.get_job(older["id"])["payload"],
        job_id=older["id"],
        worker_id="worker-old",
        gateway=FakeGateway(),
    )
    assert result["current"] is False
    assert not (
        settings.nas_root / "raw/hf/rainbowrobotics/pick-cup/current.json"
    ).exists()
    source = database.list_hf_sources()[0]
    assert source["desired_commit_sha"] == NEW_SHA
    assert source["current_commit_sha"] is None


def test_invalid_download_never_reaches_the_immutable_revision(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    profile = database.create_profile("Researcher")
    job, _ = database.create_hf_import_job(
        profile_id=profile["id"],
        repo_id="rainbowrobotics/pick-cup",
        dataset_name="pick-cup",
        requested_revision="main",
        commit_sha=SHA,
        expected_file_count=1,
        expected_total_bytes=100,
        idempotency_key="invalid-download",
    )
    database.claim_job(job["id"], worker_id="worker-1", lease_seconds=120)

    try:
        import_huggingface_dataset(
            database=database,
            settings=settings,
            payload=database.get_job(job["id"])["payload"],
            job_id=job["id"],
            worker_id="worker-1",
            gateway=InvalidDatasetGateway(),
        )
    except HuggingFaceImportValidationError:
        pass
    else:
        raise AssertionError("an incomplete dataset must not be published")
    assert not (
        settings.nas_root / "raw/hf/rainbowrobotics/pick-cup/revisions" / SHA
    ).exists()
    assert database.list_hf_sources()[0]["current_commit_sha"] is None


def test_existing_revision_content_cannot_be_silently_replaced(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    profile = database.create_profile("Researcher")

    def create_and_claim(key: str, worker_id: str):
        job, _ = database.create_hf_import_job(
            profile_id=profile["id"],
            repo_id="rainbowrobotics/pick-cup",
            dataset_name="pick-cup",
            requested_revision="main",
            commit_sha=SHA,
            expected_file_count=2,
            expected_total_bytes=512,
            idempotency_key=key,
        )
        database.claim_job(job["id"], worker_id=worker_id, lease_seconds=120)
        return job

    first = create_and_claim("first-copy", "worker-1")
    import_huggingface_dataset(
        database=database,
        settings=settings,
        payload=database.get_job(first["id"])["payload"],
        job_id=first["id"],
        worker_id="worker-1",
        gateway=FakeGateway(),
    )
    revision = settings.nas_root / "raw/hf/rainbowrobotics/pick-cup/revisions" / SHA
    (revision / "data/part-000.parquet").write_bytes(b"tampered")
    second = create_and_claim("second-copy", "worker-2")

    try:
        import_huggingface_dataset(
            database=database,
            settings=settings,
            payload=database.get_job(second["id"])["payload"],
            job_id=second["id"],
            worker_id="worker-2",
            gateway=FakeGateway(),
        )
    except HuggingFaceImportConflictError:
        pass
    else:
        raise AssertionError("tampered immutable content must raise a conflict")
    assert (revision / "data/part-000.parquet").read_bytes() == b"tampered"


def test_import_rejects_a_symlinked_managed_storage_component(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (settings.nas_root / "raw/hf").symlink_to(outside, target_is_directory=True)
    database = Database(settings.database_path)
    database.initialize()
    profile = database.create_profile("Researcher")
    job, _ = database.create_hf_import_job(
        profile_id=profile["id"],
        repo_id="rainbowrobotics/pick-cup",
        dataset_name="pick-cup",
        requested_revision="main",
        commit_sha=SHA,
        expected_file_count=2,
        expected_total_bytes=512,
        idempotency_key="symlink-storage",
    )
    database.claim_job(job["id"], worker_id="worker-1", lease_seconds=120)

    try:
        import_huggingface_dataset(
            database=database,
            settings=settings,
            payload=database.get_job(job["id"])["payload"],
            job_id=job["id"],
            worker_id="worker-1",
            gateway=FakeGateway(),
        )
    except HuggingFaceImportValidationError:
        pass
    else:
        raise AssertionError("symlinked storage paths must be rejected")
    assert list(outside.iterdir()) == []


def test_import_request_forbids_credentials_and_path_overrides(tmp_path: Path) -> None:
    client, _, _, _, _ = _client(tmp_path)
    profile_id = client.post("/api/v1/profiles", json={"name": "Researcher"}).json()[
        "id"
    ]
    base = {
        "profile_id": profile_id,
        "dataset_name": "pick-cup",
        "requested_revision": "main",
        "commit_sha": SHA,
        "idempotency_key": "strict",
    }
    for field in ("token", "apiKey", "password_file", "path"):
        response = client.post("/api/v1/hf/imports", json={**base, field: "secret"})
        assert response.status_code == 422


def test_hf_request_accepts_shared_trailing_underscore_name_validator(
    tmp_path: Path,
) -> None:
    client, _, dispatcher, gateway, _ = _client(tmp_path)
    gateway.commit_sha = SHA
    gateway.resolve_revision = lambda dataset_name, revision: {
        "repo_id": f"rainbowrobotics/{dataset_name}",
        "requested_revision": revision,
        "commit_sha": SHA,
        "file_count": 2,
        "total_bytes": 512,
    }
    profile_id = client.post("/api/v1/profiles", json={"name": "Researcher"}).json()[
        "id"
    ]
    response = client.post(
        "/api/v1/hf/imports",
        json={
            "profile_id": profile_id,
            "dataset_name": "safe_trailing_underscore_",
            "requested_revision": "main",
            "commit_sha": SHA,
            "idempotency_key": "hf-trailing-underscore",
        },
    )
    assert response.status_code == 202
    assert response.json()["payload"]["dataset_name"] == "safe_trailing_underscore_"
    assert len(dispatcher.enqueued) == 1

    internal = _validate_hf_import_payload(response.json()["payload"])
    assert internal["repo_id"] == "rainbowrobotics/safe_trailing_underscore_"
