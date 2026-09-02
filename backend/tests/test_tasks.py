from __future__ import annotations

from pathlib import Path

from datasetui.database import Database
from datasetui.tasks import run_job


def test_worker_claims_and_completes_job(tmp_path: Path, monkeypatch) -> None:
    database_path = tmp_path / "datasetui.sqlite3"
    monkeypatch.setenv("DATASETUI_DB_PATH", str(database_path))
    database = Database(database_path)
    database.initialize()
    profile = database.create_profile("Researcher")
    job, _ = database.create_job(
        kind="phase2.smoke",
        queue_name="cpu",
        profile_id=profile["id"],
        payload={},
        idempotency_key="worker-completion",
    )

    result = run_job(job["id"])
    assert result == {"job_id": job["id"], "status": "succeeded", "claimed": True}
    stored = database.get_job(job["id"])
    assert stored["result"] == {"ok": True}
    assert [event["event_type"] for event in database.list_job_events(job["id"])] == [
        "queued",
        "running",
        "succeeded",
    ]
