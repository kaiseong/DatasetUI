"""Content-bound CPU checks followed by existing IO delivery jobs."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from datasetui.config import Settings
from datasetui.database import Database, RecipeRevisionMismatchError
from datasetui.queueing import QueueDispatcher

LOG = logging.getLogger(__name__)


DELIVERY_KINDS = {
    "datasets.export_nas",
    "datasets.upload_hf",
    "datasets.copy_pc_key",
    "datasets.copy_pc_password",
}


def dispatch_registered_job(
    database: Database, dispatcher: QueueDispatcher, settings: Settings, job: dict
) -> dict:
    if job["status"] != "queued" or job.get("wait_reason"):
        return job
    timeout = (
        settings.validation_timeout_seconds
        if job["kind"] in {"datasets.validate", "datasets.delivery_preflight"}
        else None
    )
    if job["kind"] == "datasets.merge":
        timeout = settings.merge_timeout_seconds
    if job["kind"] == "curation.materialize":
        snapshot = database.get_curation_snapshot(job["payload"]["snapshot_id"])
        values = []
        if snapshot["relative_action"].get("enabled"):
            values.append(settings.relative_timeout_seconds)
        if snapshot["trim_config"].get("enabled"):
            values.append(settings.trim_timeout_seconds)
        timeout = max(values) if values else None
    rq_id = job["rq_job_id"] or database.rq_job_id_for(job["id"])
    if job["rq_job_id"] is None:
        database.mark_enqueued(job["id"], rq_id)
    try:
        dispatcher.enqueue(
            job_id=job["id"],
            queue_name=job["queue_name"],
            rq_job_id=rq_id,
            **({"job_timeout": timeout} if timeout else {}),
        )
    except Exception:
        database.record_dispatch_error(job["id"], "Unable to dispatch job")
        raise
    return database.clear_dispatch_error(job["id"])


def create_delivery_workflow(
    database: Database,
    dispatcher: QueueDispatcher,
    settings: Settings,
    *,
    kind: str,
    profile_id: str,
    payload: dict,
    idempotency_key: str,
    password: str | None = None,
):
    from datasetui.credential_store import credential_store

    job, created = database.create_delivery_with_validation(
        kind=kind,
        profile_id=profile_id,
        payload=payload,
        idempotency_key=idempotency_key,
    )
    if job["status"] != "queued":
        return job, created
    if kind == "datasets.copy_pc_password":
        try:
            if password is None:
                raise ValueError("password is required for queued PC transfer")
            credential_store(settings).put(job["id"], password)
            with database.connect() as connection:
                connection.execute(
                    "UPDATE jobs SET validation_ready=0 WHERE id=? AND status='queued' AND validation_ready=-1",
                    (job["id"],),
                )
        except Exception:
            database.fail_waiting_job(
                job["id"],
                "credential_unavailable",
                "임시 비밀번호를 보관하지 못했습니다. 저장소 설정을 확인한 뒤 다시 요청하세요.",
            )
            raise
    check = database.get_job(job["validation_job_id"])
    if check["status"] == "queued":
        dispatch_registered_job(database, dispatcher, settings, check)
    reconcile_deliveries(database, dispatcher, settings)
    return database.get_job(job["id"]), created


def reconcile_deliveries(
    database: Database, dispatcher: QueueDispatcher, settings: Settings
) -> None:
    # Read all dependencies, not the limited recent-job page. API restart and
    # worker completion use the same idempotent dispatch path.
    with database.connect() as connection:
        password_jobs = connection.execute(
            "SELECT id,validation_ready,created_at FROM jobs WHERE kind='datasets.copy_pc_password' AND status='queued'"
        ).fetchall()
    for row in password_jobs:
        # Allow the API to complete the two-phase memory reservation.
        age = (
            datetime.now(timezone.utc) - datetime.fromisoformat(row["created_at"])
        ).total_seconds()
        if row["validation_ready"] == -1 and age < 15:
            continue
        try:
            from datasetui.credential_store import credential_store

            present = credential_store(settings).exists(row["id"])
        except Exception:
            present = False
        if not present:
            database.fail_waiting_job(
                row["id"],
                "credential_unavailable",
                "임시 비밀번호가 만료되거나 소실되었습니다. 다시 입력해 새 작업을 요청하세요.",
            )
        elif row["validation_ready"] == -1:
            with database.connect() as connection:
                connection.execute(
                    "UPDATE jobs SET validation_ready=0 WHERE id=? AND status='queued' AND validation_ready=-1",
                    (row["id"],),
                )
    with database.connect() as connection:
        waiting = connection.execute(
            "SELECT id,validation_job_id,validation_ready FROM jobs WHERE status='queued' AND validation_job_id IS NOT NULL AND validation_ready >= 0"
        ).fetchall()
    for row in waiting:
        try:
            check = database.get_job(row["validation_job_id"])
            if check["status"] == "succeeded":
                if check["result"] and check["result"].get("passed") is True:
                    database.release_delivery_validation(row["id"])
                    dispatch_registered_job(
                        database, dispatcher, settings, database.get_job(row["id"])
                    )
                else:
                    database.fail_waiting_job(
                        row["id"],
                        "delivery_validation_failed",
                        "선행 전체 검사를 통과하지 못해 전달하지 않았습니다.",
                    )
            elif check["status"] in {"failed", "cancelled", "interrupted"}:
                database.fail_waiting_job(
                    row["id"],
                    "delivery_validation_failed",
                    "선행 검사가 실패하거나 취소되어 전달하지 않았습니다.",
                )
            elif check["status"] == "queued":
                dispatch_registered_job(database, dispatcher, settings, check)
        except Exception:
            LOG.warning("delivery reconciliation delayed for %s", row["id"])
    with database.connect() as connection:
        terminal = connection.execute(
            "SELECT id FROM jobs WHERE kind='datasets.copy_pc_password' AND status NOT IN ('queued','running')"
        ).fetchall()
        orphan_checks = connection.execute("""SELECT id FROM jobs c WHERE c.kind='datasets.delivery_preflight' AND c.status IN ('queued','running') AND c.cancellation_requested_at IS NULL
            AND NOT EXISTS (SELECT 1 FROM jobs d WHERE d.validation_job_id=c.id AND d.status IN ('queued','running'))""").fetchall()
        crashed_purges = connection.execute(
            "SELECT dataset_id FROM dataset_purge_reservations WHERE state='purging' AND job_id IN (SELECT id FROM jobs WHERE status NOT IN ('queued','running'))"
        ).fetchall()
        connection.execute(
            "UPDATE dataset_purge_reservations SET state='failed' WHERE state='purging' AND job_id IN (SELECT id FROM jobs WHERE status NOT IN ('queued','running'))"
        )
        connection.execute(
            "DELETE FROM dataset_purge_reservations WHERE state='reserved' AND job_id IN (SELECT id FROM jobs WHERE status NOT IN ('queued','running'))"
        )
    for row in crashed_purges:
        database.mark_dataset_trash_recovery_required(row["dataset_id"])
    for row in orphan_checks:
        try:
            check = database.get_job(row["id"])
            cancelled = database.request_job_cancellation(
                check["id"], profile_id=check["profile_id"]
            )
        except Exception:
            # It may have finished after the orphan snapshot. Do not turn a
            # successfully cancelled parent into an API failure.
            LOG.warning("orphan check cancellation delayed for %s", row["id"])
            continue
        if cancelled["status"] == "cancelled" and cancelled["rq_job_id"]:
            try:
                dispatcher.remove_pending(
                    rq_job_id=cancelled["rq_job_id"], queue_name=cancelled["queue_name"]
                )
            except Exception:
                LOG.warning("cancelled check removal delayed for %s", check["id"])
    if terminal and settings.credential_redis_url:
        from datasetui.credential_store import credential_store

        for row in terminal:
            try:
                credential_store(settings).delete(row["id"])
            except Exception:
                LOG.warning("credential cleanup delayed for job %s", row["id"])


def recover_workflows(
    database: Database, dispatcher: QueueDispatcher, settings: Settings
) -> None:
    """Server-owned recovery must not depend on an open browser tab."""
    for job_id in database.requeue_expired_jobs():
        try:
            dispatch_registered_job(
                database, dispatcher, settings, database.get_job(job_id)
            )
        except Exception:
            LOG.warning("expired job redispatch delayed for %s", job_id)
    # Retry accepted requests whose initial queue dispatch was unavailable.
    with database.connect() as connection:
        pending = connection.execute(
            "SELECT id FROM jobs WHERE status='queued' AND validation_ready=1 AND error_code='queue_unavailable'"
        ).fetchall()
    for row in pending:
        try:
            dispatch_registered_job(
                database, dispatcher, settings, database.get_job(row["id"])
            )
        except Exception:
            LOG.warning("job dispatch delayed for %s", row["id"])
    reconcile_deliveries(database, dispatcher, settings)


def run_delivery_preflight(
    database: Database, settings: Settings, payload: dict, job_id: str, worker_id: str
) -> dict:
    from datasetui.delivery.transfer import ExportGateRequiredError, gated_source
    from datasetui.job_progress import JobProgressReporter
    from datasetui.validation.run import validate_registered_dataset

    progress = JobProgressReporter(database, job_id=job_id, worker_id=worker_id)
    progress(
        {
            "stage": "export_gate",
            "completed": 0,
            "total": 0,
            "unit": "files",
            "current_item": "현재 데이터 내용과 검사 기록 확인",
            "_force": True,
        }
    )
    try:
        _record, _root, manifest = gated_source(
            database, settings, payload, exclude_job_id=job_id
        )
    except (ExportGateRequiredError, RecipeRevisionMismatchError):

        def report(value):
            database.assert_job_lease(job_id, worker_id=worker_id)
            database.update_validation_progress(
                job_id, worker_id=worker_id, progress=value
            )
            progress({**value, "stage": "validate", "unit": "episodes"})

        return validate_registered_dataset(
            database=database, settings=settings, payload=payload, on_progress=report
        )
    database.assert_job_lease(job_id, worker_id=worker_id)
    result = database.export_gate_result(
        dataset_id=payload["dataset_id"],
        dataset_fingerprint=payload["fingerprint"],
        exclude_job_id=job_id,
    )
    return {**result, "content_manifest": manifest, "reused": True}
