from __future__ import annotations

import logging
import os
import socket
import threading
import uuid
from contextlib import contextmanager
from collections.abc import Iterator

from datasetui.config import Settings
from datasetui.database import Database, JobLeaseLostError, RecipeRevisionMismatchError
from datasetui.datasets import DatasetRootUnavailableError
from datasetui.hf_errors import (
    HuggingFaceDatasetNotFoundError,
    HuggingFaceImportConflictError,
    HuggingFaceImportTooLargeError,
    HuggingFaceImportValidationError,
    HuggingFaceRevisionNotFoundError,
    HuggingFaceUnavailableError,
)
from datasetui.jobs import run_registered_job
from datasetui.job_cancellation import (
    JobCancellationRequested,
    cancellation_monitor,
)
from datasetui.job_progress import JobProgressReporter
from datasetui.transform_errors import CurationTransformError


logger = logging.getLogger("datasetui.worker")


def run_job(job_id: str) -> dict[str, object]:
    """RQ entrypoint. Redis carries only the opaque job ID."""

    settings = Settings.from_env()
    database = Database(settings.database_path)
    database.initialize()
    worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"
    job = database.claim_job(
        job_id,
        worker_id=worker_id,
        lease_seconds=settings.job_lease_seconds,
    )
    if job is None:
        current = database.get_job(job_id)
        return {"job_id": job_id, "status": current["status"], "claimed": False}

    try:
        with _heartbeat_lease(
            database,
            job_id=job_id,
            worker_id=worker_id,
            lease_seconds=settings.job_lease_seconds,
            heartbeat_seconds=settings.job_heartbeat_seconds,
        ), cancellation_monitor(
            database,
            job_id=job_id,
            worker_id=worker_id,
        ):
            progress = JobProgressReporter(
                database,
                job_id=job_id,
                worker_id=worker_id,
                output_name=_job_output_name(job),
            )
            progress(
                {
                    "stage": "preparing",
                    "completed": 0,
                    "total": 0,
                    "unit": "items",
                    "current_item": "작업자가 처리를 시작했습니다",
                    "_force": True,
                }
            )
            if job["kind"] in {
                "hf.import",
                "datasets.scan",
                "datasets.validate",
                "curation.materialize",
                "datasets.merge",
                "datasets.convert_v21",
                "datasets.export_nas",
                "datasets.upload_hf",
                "datasets.copy_pc_key",
                "datasets.copy_pc_password",
                "datasets.delivery_preflight",
                "hf.delete",
                "datasets.empty_trash",
                "segmentation.preview",
                "segmentation.sample",
                "segmentation.batch_prepare",
                "segmentation.export",
                "segmentation.batch_export",
            }:
                result = run_registered_job(
                    job["kind"],
                    job["payload"],
                    job_id=job_id,
                    worker_id=worker_id,
                )
            else:
                result = run_registered_job(job["kind"], job["payload"])
            database.assert_job_lease(job_id, worker_id=worker_id)
            progress(
                {
                    "stage": "complete",
                    "completed": 1,
                    "total": 1,
                    "unit": "items",
                    "current_item": "작업 완료",
                    "_force": True,
                }
            )
            finished = database.succeed_job(job_id, result, worker_id=worker_id)
    except JobCancellationRequested:
        logger.info("job %s cancellation acknowledged", job_id)
        finished = _complete_cancellation(database, job_id, worker_id=worker_id)
        return {"job_id": job_id, "status": finished["status"], "claimed": True}
    except Exception as exc:
        if job["kind"] == "datasets.copy_pc_password":
            # SSH/network exception text can include credentials. Neither our
            # logs nor RQ's persisted failure traceback may receive it.
            logger.warning("password transfer job %s failed", job_id)
        else:
            logger.exception("job %s failed", job_id)
        error_code, public_message = _public_failure(exc)
        try:
            database.fail_job(
                job_id,
                error_code,
                public_message,
                worker_id=worker_id,
            )
        except JobCancellationRequested:
            finished = _complete_cancellation(database, job_id, worker_id=worker_id)
            return {
                "job_id": job_id,
                "status": finished["status"],
                "claimed": True,
            }
        except JobLeaseLostError:
            logger.warning("job %s lost its lease before failure was recorded", job_id)
        if job["kind"] == "datasets.copy_pc_password":
            raise RuntimeError(public_message) from None
        raise
    finally:
        from datasetui.delivery_workflow import reconcile_deliveries
        from datasetui.queueing import RQDispatcher

        try:
            dispatcher = RQDispatcher(
                settings.redis_url,
                settings.job_timeout_seconds,
                settings.io_job_timeout_seconds,
            )
            reconcile_deliveries(database, dispatcher, settings)
        except Exception:
            logger.warning("workflow follow-up delayed for job %s", job_id)

    return {"job_id": job_id, "status": finished["status"], "claimed": True}


def _complete_cancellation(
    database: Database, job_id: str, *, worker_id: str
) -> dict[str, object]:
    try:
        return database.complete_job_cancellation(job_id, worker_id=worker_id)
    except JobLeaseLostError:
        logger.warning("job %s lost its lease before cancellation was recorded", job_id)
        return database.get_job(job_id)


def _job_output_name(job: dict[str, object]) -> str:
    payload = job.get("payload")
    if not isinstance(payload, dict):
        return ""
    for key in ("output_name", "repo_name", "dataset_name"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return ""


@contextmanager
def _heartbeat_lease(
    database: Database,
    *,
    job_id: str,
    worker_id: str,
    lease_seconds: int,
    heartbeat_seconds: int,
) -> Iterator[None]:
    if lease_seconds < 3:
        raise ValueError("job lease must be at least three seconds")
    interval = max(1, min(heartbeat_seconds, lease_seconds // 3))
    stopped = threading.Event()
    lost = threading.Event()

    def heartbeat() -> None:
        while not stopped.wait(interval):
            try:
                if not database.heartbeat_job(
                    job_id,
                    worker_id=worker_id,
                    lease_seconds=lease_seconds,
                ):
                    lost.set()
                    return
            except Exception:
                logger.exception("job %s heartbeat failed", job_id)
                lost.set()
                return

    thread = threading.Thread(
        target=heartbeat,
        name=f"datasetui-heartbeat-{job_id}",
        daemon=True,
    )
    thread.start()
    try:
        yield
        if lost.is_set():
            raise JobLeaseLostError(job_id)
    finally:
        stopped.set()
        thread.join(timeout=max(1, interval + 1))


def _public_failure(exc: Exception) -> tuple[str, str]:
    from datasetui.library_operations import LibraryOperationError
    from datasetui.credential_store import CredentialUnavailableError

    if isinstance(exc, LibraryOperationError):
        return exc.code, str(exc)
    if isinstance(exc, CredentialUnavailableError):
        return (
            "credential_unavailable",
            "비밀번호가 만료되거나 임시 저장소에 연결할 수 없습니다. 다시 입력해 새 작업을 요청하세요.",
        )
    from datasetui.delivery import ExportGateRequiredError, HuggingFaceExternalOperationAmbiguousError

    if isinstance(exc, ExportGateRequiredError):
        return (
            "export_gate_required",
            "Run the current content-bound export gate before delivery",
        )
    if isinstance(exc, HuggingFaceExternalOperationAmbiguousError):
        return (
            "external_outcome_uncertain",
            "External delivery outcome needs manual verification",
        )
    from rq.timeouts import JobTimeoutException

    if isinstance(exc, JobTimeoutException):
        return "job_timeout", "The job exceeded its execution time limit"

    from datasetui.segmentation.engine import Sam3UnavailableError, Sam3InferenceError, Sam3PromptMatchError

    if isinstance(exc, Sam3PromptMatchError):
        return "segmentation_guidance", str(exc)
    from datasetui.segmentation.errors import SegmentationGuidanceError

    if isinstance(exc, SegmentationGuidanceError):
        return "segmentation_guidance", str(exc)
    if isinstance(exc, Sam3UnavailableError):
        return (
            "segmentation_unavailable",
            "Check the SAM 3.1 checkpoint, SHA256 and CUDA worker configuration",
        )
    if isinstance(exc, Sam3InferenceError):
        return (
            "segmentation_failed",
            "SAM 3.1 did not produce a complete valid mask sequence",
        )
    if isinstance(exc, DatasetRootUnavailableError):
        return "storage_unavailable", "Dataset storage is unavailable"
    if isinstance(exc, HuggingFaceImportTooLargeError):
        return "hf_import_too_large", "Dataset exceeds the import size limit"
    if isinstance(
        exc,
        (HuggingFaceDatasetNotFoundError, HuggingFaceRevisionNotFoundError),
    ):
        return "hf_source_not_found", "The selected Hugging Face revision was not found"
    if isinstance(exc, HuggingFaceUnavailableError):
        return "hf_unavailable", "Hugging Face is temporarily unavailable"
    if isinstance(exc, HuggingFaceImportConflictError):
        return "hf_revision_conflict", "The immutable dataset revision conflicts"
    if isinstance(exc, HuggingFaceImportValidationError):
        return "hf_validation_failed", "The downloaded dataset failed validation"
    if isinstance(exc, JobLeaseLostError):
        return "worker_lost", "The worker stopped responding"
    if isinstance(exc, RecipeRevisionMismatchError):
        return "source_revision_changed", "The source dataset revision changed"
    if isinstance(exc, CurationTransformError):
        return "curation_failed", str(exc)
    return "job_failed", "The job could not be completed"
