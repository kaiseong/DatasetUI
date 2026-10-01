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


def test_queued_job_can_be_cancelled_by_its_profile(
    client: TestClient, dispatcher: RecordingDispatcher
) -> None:
    profile_id = _profile_id(client)
    created = client.post(
        "/api/v1/jobs",
        json={
            "kind": "phase2.smoke",
            "profile_id": profile_id,
            "payload": {},
            "idempotency_key": "cancel-via-api",
        },
    ).json()

    cancelled = client.post(
        f"/api/v1/jobs/{created['id']}/cancel",
        json={"profile_id": profile_id},
    )
    repeated = client.post(
        f"/api/v1/jobs/{created['id']}/cancel",
        json={"profile_id": profile_id},
    )

    assert cancelled.status_code == repeated.status_code == 200
    assert cancelled.json()["status"] == repeated.json()["status"] == "cancelled"
    assert dispatcher.removed == [
        (created["rq_job_id"], "cpu"),
        (created["rq_job_id"], "cpu"),
    ]
    events = client.get(f"/api/v1/jobs/{created['id']}/events").json()
    assert [event["event_type"] for event in events] == [
        "queued",
        "dispatched",
        "cancelled",
    ]


def test_job_cancellation_enforces_owner_and_requests_running_cancellation(
    client: TestClient, database: Database, dispatcher: RecordingDispatcher
) -> None:
    owner_id = _profile_id(client)
    other_id = client.post("/api/v1/profiles", json={"name": "Other"}).json()["id"]
    job, _ = database.create_job(
        kind="phase2.smoke",
        queue_name="cpu",
        profile_id=owner_id,
        payload={},
        idempotency_key="cancel-guard",
    )
    database.mark_enqueued(job["id"], f"datasetui-{job['id']}")

    wrong_owner = client.post(
        f"/api/v1/jobs/{job['id']}/cancel", json={"profile_id": other_id}
    )
    assert wrong_owner.status_code == 403
    assert database.get_job(job["id"])["status"] == "queued"

    database.claim_job(job["id"], worker_id="active")
    running = client.post(
        f"/api/v1/jobs/{job['id']}/cancel", json={"profile_id": owner_id}
    )
    assert running.status_code == 200
    assert running.json()["status"] == "running"
    assert running.json()["cancellation_requested"] is True
    assert running.json()["cancellation_requested_at"] is not None
    assert database.get_job(job["id"])["status"] == "running"
    assert dispatcher.removed == []


def test_queue_cleanup_failure_does_not_undo_database_cancellation(
    client: TestClient, database: Database, dispatcher: RecordingDispatcher
) -> None:
    profile_id = _profile_id(client)
    job, _ = database.create_job(
        kind="phase2.smoke",
        queue_name="cpu",
        profile_id=profile_id,
        payload={},
        idempotency_key="cancel-queue-unavailable",
    )
    database.mark_enqueued(job["id"], f"datasetui-{job['id']}")
    dispatcher.available = False

    response = client.post(
        f"/api/v1/jobs/{job['id']}/cancel", json={"profile_id": profile_id}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert database.get_job(job["id"])["status"] == "cancelled"


def test_finalizing_job_rejects_late_cancellation(
    client: TestClient, database: Database
) -> None:
    profile_id = _profile_id(client)
    job, _ = database.create_job(
        kind="phase2.smoke",
        queue_name="cpu",
        profile_id=profile_id,
        payload={},
        idempotency_key="cancel-too-late",
    )
    database.claim_job(job["id"], worker_id="publisher", lease_seconds=120)
    database.begin_job_finalization(job["id"], worker_id="publisher")

    response = client.post(
        f"/api/v1/jobs/{job['id']}/cancel", json={"profile_id": profile_id}
    )

    assert response.status_code == 409
    assert "결과 게시" in response.json()["detail"]
    stored = client.get(f"/api/v1/jobs/{job['id']}").json()
    assert stored["cancellation_guarded_at"] is not None
    assert stored["cancellation_requested"] is False


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
        "schema_versions": list(range(1, 17)),
    }

    dispatcher.available = False
    degraded = client.get("/api/v1/system/health")
    assert degraded.status_code == 503
    assert degraded.json()["queue"] == "error"
