from datetime import datetime, timedelta, timezone
import pytest
from datasetui.database import JobLeaseLostError
from datasetui.tasks import _public_failure
from rq.timeouts import JobTimeoutException
from test_curation_api import _register


def test_expired_validation_fails_once_and_is_not_requeued(client, database):
    dataset = _register(database)
    profile = database.create_profile("Validation recovery")
    job, _ = database.create_job(
        kind="datasets.validate",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={},
        idempotency_key="expired-validation",
    )
    database.record_validation_run(
        job_id=job["id"],
        dataset_id=dataset["id"],
        dataset_fingerprint=dataset["fingerprint"],
        mode="full",
    )
    database.claim_job(job["id"], worker_id="lost-worker", lease_seconds=120)
    expired = (
        (datetime.now(timezone.utc) - timedelta(seconds=5))
        .isoformat()
        .replace("+00:00", "Z")
    )
    with database.connect() as connection:
        connection.execute(
            "UPDATE jobs SET lease_expires_at = ? WHERE id = ?", (expired, job["id"])
        )
    response = client.get(f"/api/v1/datasets/{dataset['id']}/validations")
    assert response.status_code == 200
    assert response.json()[0]["status"] == "failed"
    assert response.json()[0]["error_code"] == "validation_interrupted"
    assert database.requeue_expired_jobs() == []
    assert database.requeue_expired_jobs() == []
    assert [e["event_type"] for e in database.list_job_events(job["id"])].count(
        "failed"
    ) == 1
    with pytest.raises(JobLeaseLostError):
        database.succeed_job(job["id"], {}, worker_id="lost-worker")


def test_live_validation_is_not_interrupted(database):
    p = database.create_profile("Active")
    j, _ = database.create_job(
        kind="datasets.validate",
        queue_name="cpu",
        profile_id=p["id"],
        payload={},
        idempotency_key="live",
    )
    database.claim_job(j["id"], worker_id="alive", lease_seconds=120)
    assert database.requeue_expired_jobs() == []
    assert database.get_job(j["id"])["status"] == "running"


def test_timeout_has_a_distinct_public_error():
    assert _public_failure(JobTimeoutException("timeout"))[0] == "job_timeout"


def test_validation_dispatch_uses_separate_timeout(client, database, monkeypatch):
    from datasetui.queueing import RecordingDispatcher

    seen = []
    original = RecordingDispatcher.enqueue

    def capture(self, **kwargs):
        seen.append(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(RecordingDispatcher, "enqueue", capture)
    dataset = _register(database)
    profile = database.create_profile("Timeout policy")
    response = client.post(
        f"/api/v1/datasets/{dataset['id']}/validations",
        json={
            "profile_id": profile["id"],
            "mode": "full",
            "idempotency_key": "timeout-policy",
        },
    )
    assert response.status_code == 202
    assert seen[-1]["job_timeout"] == 3600
