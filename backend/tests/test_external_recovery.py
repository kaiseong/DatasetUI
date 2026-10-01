import pytest


@pytest.mark.parametrize("kind", ["datasets.upload_hf", "datasets.copy_pc_key"])
def test_expired_external_job_is_not_automatically_retried(database, kind):
    profile = database.create_profile("Researcher")
    job, _ = database.create_job(
        kind=kind,
        queue_name="io",
        profile_id=profile["id"],
        payload={},
        idempotency_key=f"external-{kind}",
    )
    database.claim_job(job["id"], worker_id="lost", lease_seconds=120)
    with database.connect() as connection:
        connection.execute(
            "UPDATE jobs SET lease_expires_at = '2000-01-01T00:00:00Z' WHERE id = ?",
            (job["id"],),
        )
    assert database.requeue_expired_jobs() == []
    recovered = database.get_job(job["id"])
    assert recovered["status"] == "interrupted"
    assert recovered["error_code"] == "external_outcome_uncertain"
    assert database.claim_job(job["id"], worker_id="retry") is None
