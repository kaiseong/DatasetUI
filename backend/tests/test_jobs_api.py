from __future__ import annotations

from fastapi.testclient import TestClient

from datasetui.database import Database
from datasetui.queueing import RecordingDispatcher


def _profile_id(client: TestClient) -> str:
    response = client.post("/api/v1/profiles", json={"name": "Researcher"})
    return response.json()["id"]


def test_job_is_dispatched_by_opaque_id_only(
    client: TestClient, dispatcher: RecordingDispatcher
) -> None:
    profile_id = _profile_id(client)
    response = client.post(
        "/api/v1/jobs",
        json={
            "kind": "phase2.smoke",
            "profile_id": profile_id,
            "payload": {},
            "idempotency_key": "opaque-id",
        },
    )
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "queued"
    assert dispatcher.enqueued == [(job["id"], "cpu")]
    assert job["rq_job_id"] == f"datasetui-{job['id']}"

    events = client.get(f"/api/v1/jobs/{job['id']}/events").json()
    assert [event["event_type"] for event in events] == ["queued", "dispatched"]


def test_job_idempotency_does_not_dispatch_twice(
    client: TestClient, dispatcher: RecordingDispatcher
) -> None:
    profile_id = _profile_id(client)
    payload = {
        "kind": "phase2.smoke",
        "profile_id": profile_id,
        "payload": {},
        "idempotency_key": "same-operation",
    }
    first = client.post("/api/v1/jobs", json=payload)
    second = client.post("/api/v1/jobs", json=payload)
    assert first.status_code == 202
    assert second.status_code == 200
    assert first.json()["id"] == second.json()["id"]
    assert len(dispatcher.enqueued) == 1


def test_idempotent_retry_recovers_a_job_not_written_to_redis(
    client: TestClient, database: Database, dispatcher: RecordingDispatcher
) -> None:
    profile_id = _profile_id(client)
    job, _ = database.create_job(
        kind="phase2.smoke",
        queue_name="cpu",
        profile_id=profile_id,
        payload={},
        idempotency_key="recover-dispatch",
    )
    database.mark_enqueued(job["id"], f"datasetui-{job['id']}")

    response = client.post(
        "/api/v1/jobs",
        json={
            "kind": "phase2.smoke",
            "profile_id": profile_id,
            "payload": {},
            "idempotency_key": "recover-dispatch",
        },
    )
    assert response.status_code == 200
    assert dispatcher.enqueued == [(job["id"], "cpu")]


def test_idempotency_key_cannot_be_reused_for_another_request(
    client: TestClient, database: Database
) -> None:
    profile_id = _profile_id(client)
    database.create_job(
        kind="future.kind",
        queue_name="io",
        profile_id=profile_id,
        payload={},
        idempotency_key="bound-request",
    )

    conflict = client.post(
        "/api/v1/jobs",
        json={
            "kind": "phase2.smoke",
            "profile_id": profile_id,
            "payload": {},
            "idempotency_key": "bound-request",
        },
    )
    assert conflict.status_code == 409


def test_job_uses_a_strict_kind_specific_payload(client: TestClient) -> None:
    profile_id = _profile_id(client)
    unexpected = client.post(
        "/api/v1/jobs",
        json={
            "kind": "phase2.smoke",
            "profile_id": profile_id,
            "payload": {"apiKey": "not-accepted"},
            "idempotency_key": "strict-payload",
        },
    )
    assert unexpected.status_code == 422

    unknown = client.post(
        "/api/v1/jobs",
        json={
            "kind": "shell.command",
            "profile_id": profile_id,
            "payload": {},
            "idempotency_key": "unknown-kind",
        },
    )
    assert unknown.status_code == 422

    privileged = client.post(
        "/api/v1/jobs",
        json={
            "kind": "curation.materialize",
            "profile_id": profile_id,
            "payload": {
                "snapshot_id": "00000000-0000-4000-8000-000000000000",
                "output_name": "bypass",
            },
            "idempotency_key": "curation-bypass",
        },
    )
    assert privileged.status_code == 422


def test_dataset_scan_job_payload_is_allowlisted(
    client: TestClient, dispatcher: RecordingDispatcher
) -> None:
    profile_id = _profile_id(client)
    accepted = client.post(
        "/api/v1/jobs",
        json={
            "kind": "datasets.scan",
            "profile_id": profile_id,
            "payload": {"storage_areas": ["raw"]},
            "idempotency_key": "scan-raw",
        },
    )
    assert accepted.status_code == 202
    assert dispatcher.enqueued == [(accepted.json()["id"], "io")]

    rejected = client.post(
        "/api/v1/jobs",
        json={
            "kind": "datasets.scan",
            "profile_id": profile_id,
            "payload": {"storage_areas": ["raw"], "path": "/etc"},
            "idempotency_key": "scan-arbitrary-path",
        },
    )
    assert rejected.status_code == 422


def test_queue_failure_is_persisted(
    database: Database, dispatcher: RecordingDispatcher
) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from datasetui.api import create_router

    dispatcher.available = False
    app = FastAPI()
    app.include_router(create_router(database, dispatcher))
    client = TestClient(app)
    profile_id = _profile_id(client)

    payload = {
        "kind": "phase2.smoke",
        "profile_id": profile_id,
        "payload": {},
        "idempotency_key": "queue-failure",
    }
    response = client.post("/api/v1/jobs", json=payload)
    assert response.status_code == 503
    job_id = response.json()["detail"]["job_id"]
    assert database.get_job(job_id)["status"] == "queued"
    assert database.get_job(job_id)["error_code"] == "queue_unavailable"

    dispatcher.available = True
    retried = client.post("/api/v1/jobs", json=payload)
    assert retried.status_code == 200
    assert retried.json()["id"] == job_id
    assert retried.json()["error_code"] is None
    assert dispatcher.enqueued == [(job_id, "cpu")]
    assert [event["event_type"] for event in database.list_job_events(job_id)] == [
        "queued",
        "dispatched",
        "dispatch_uncertain",
        "dispatch_recovered",
    ]


def test_system_health_reports_both_dependencies(
    client: TestClient, dispatcher: RecordingDispatcher
) -> None:
    healthy = client.get("/api/v1/system/health")
    assert healthy.status_code == 200
    assert healthy.json() == {
        "ok": True,
        "service": "datasetui-workbench",
        "database": "ok",
        "queue": "ok",
        "schema_versions": [1, 2, 3, 4, 5, 6],
    }

    dispatcher.available = False
    degraded = client.get("/api/v1/system/health")
    assert degraded.status_code == 503
    assert degraded.json()["queue"] == "error"
