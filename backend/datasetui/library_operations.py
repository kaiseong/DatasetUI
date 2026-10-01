"""Explicitly confirmed remote and NAS permanent deletion jobs."""

from uuid import UUID

from datasetui.database import Database, utc_now
from datasetui.job_progress import JobProgressReporter
from datasetui.huggingface import (
    HF_NAMESPACE,
    validate_dataset_name,
    validate_commit_sha,
)


class LibraryOperationError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def validate_library_operation(kind: str, payload: dict) -> dict:
    if kind == "hf.delete":
        if set(payload) != {"repo_id", "dataset_name", "expected_commit_sha"}:
            raise ValueError("invalid HF deletion payload")
        name = validate_dataset_name(payload["dataset_name"])
        if payload["repo_id"] != f"{HF_NAMESPACE}/{name}":
            raise ValueError("invalid HF deletion namespace")
        return {
            "repo_id": payload["repo_id"],
            "dataset_name": name,
            "expected_commit_sha": validate_commit_sha(payload["expected_commit_sha"]),
        }
    if (
        set(payload) != {"items"}
        or not isinstance(payload["items"], list)
        or not 1 <= len(payload["items"]) <= 500
    ):
        raise ValueError("invalid trash purge payload")
    seen = set()
    for item in payload["items"]:
        if not isinstance(item, dict) or set(item) != {
            "dataset_id",
            "expected_fingerprint",
        }:
            raise ValueError("invalid trash purge item")
        UUID(item["dataset_id"])
        fp = item["expected_fingerprint"]
        if (
            not isinstance(fp, str)
            or len(fp) != 64
            or any(char not in "0123456789abcdef" for char in fp)
        ):
            raise ValueError("invalid trash fingerprint")
        if item["dataset_id"] in seen:
            raise ValueError("duplicate purge item")
        seen.add(item["dataset_id"])
    return payload


def reserve_trash_items(database: Database, job: dict) -> None:
    with database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        status = connection.execute(
            "SELECT status,cancellation_requested_at FROM jobs WHERE id=?", (job["id"],)
        ).fetchone()
        if (
            status is None
            or status["status"] != "queued"
            or status["cancellation_requested_at"]
        ):
            return
        for item in job["payload"]["items"]:
            row = connection.execute(
                """SELECT t.state,d.fingerprint FROM dataset_trash t
                JOIN datasets d ON d.id=t.dataset_id WHERE t.dataset_id=?""",
                (item["dataset_id"],),
            ).fetchone()
            if (
                row is None
                or row["state"] != "trashed"
                or row["fingerprint"] != item["expected_fingerprint"]
            ):
                continue
            if database._active_jobs_reference_dataset(connection, item["dataset_id"]):
                continue
            connection.execute(
                """INSERT INTO dataset_purge_reservations(dataset_id,job_id,state,updated_at)
                VALUES (?,?,'reserved',?) ON CONFLICT(dataset_id) DO NOTHING""",
                (item["dataset_id"], job["id"], utc_now()),
            )


def run_library_operation(
    kind: str, database: Database, settings, payload: dict, job_id: str, worker_id: str
) -> dict:
    payload = validate_library_operation(kind, payload)
    progress = JobProgressReporter(database, job_id=job_id, worker_id=worker_id)
    if kind == "hf.delete":
        from huggingface_hub import HfApi
        from huggingface_hub.errors import RepositoryNotFoundError, HfHubHTTPError

        if not settings.hf_write_token:
            raise LibraryOperationError(
                "hf_delete_denied", "Hugging Face 삭제 권한이 설정되지 않았습니다."
            )
        api = HfApi(token=settings.hf_write_token)
        progress(
            {
                "stage": "verify",
                "completed": 0,
                "total": 0,
                "unit": "items",
                "current_item": payload["repo_id"],
                "_force": True,
            }
        )
        try:
            info = api.dataset_info(payload["repo_id"])
            if info.sha != payload["expected_commit_sha"]:
                raise LibraryOperationError(
                    "confirmation_mismatch",
                    "HF 데이터셋이 변경되었습니다. 목록을 확인하고 다시 삭제하세요.",
                )
        except RepositoryNotFoundError:
            # A private repository can also return 404 for an unauthorized token.
            # Do not falsely report that remote deletion succeeded.
            raise LibraryOperationError(
                "hf_delete_denied",
                "현재 토큰으로 HF 저장소를 확인할 수 없습니다. 원격 상태와 권한을 확인하세요.",
            ) from None
        except HfHubHTTPError:
            raise LibraryOperationError(
                "hf_delete_denied", "HF 저장소 접근 또는 삭제 권한을 확인하세요."
            ) from None
        database.begin_job_finalization(job_id, worker_id=worker_id)
        progress(
            {
                "stage": "delete",
                "completed": 0,
                "total": 1,
                "unit": "items",
                "current_item": "Hugging Face 저장소 영구 삭제",
                "_force": True,
            }
        )
        try:
            api.delete_repo(repo_id=payload["repo_id"], repo_type="dataset")
        except HfHubHTTPError as exc:
            if getattr(getattr(exc, "response", None), "status_code", None) in {
                401,
                403,
            }:
                raise LibraryOperationError(
                    "hf_delete_denied", "HF 저장소 삭제 권한이 없습니다."
                ) from None
            raise LibraryOperationError(
                "hf_delete_uncertain",
                "원격 삭제 결과를 확인하지 못했습니다. HF에서 확인 후 다시 요청하세요.",
            ) from None
        except Exception:
            raise LibraryOperationError(
                "hf_delete_uncertain",
                "원격 삭제 결과가 불명확합니다. 자동 재시도하지 않습니다.",
            ) from None
        return {"repo_id": payload["repo_id"], "deleted": True}

    from datasetui.dataset_trash import purge_dataset_from_trash

    with database.connect() as connection:
        reserved = {
            row["dataset_id"]
            for row in connection.execute(
                "SELECT dataset_id FROM dataset_purge_reservations WHERE job_id=? AND state='reserved'",
                (job_id,),
            )
        }
    deleted, failed, skipped = [], [], []
    for index, item in enumerate(payload["items"]):
        dataset_id = item["dataset_id"]
        database.assert_job_lease(job_id, worker_id=worker_id)
        progress(
            {
                "stage": "delete",
                "completed": index,
                "total": len(payload["items"]),
                "unit": "datasets",
                "current_item": dataset_id,
                "_force": True,
            }
        )
        if dataset_id not in reserved:
            skipped.append(
                {
                    "dataset_id": dataset_id,
                    "reason": "상태 변경·진행 작업·삭제 예약으로 제외",
                }
            )
            continue
        # No auto replay or cancellation may claim to undo permanent deletion.
        database.begin_job_finalization(job_id, worker_id=worker_id)
        with database.connect() as connection:
            changed = connection.execute(
                "UPDATE dataset_purge_reservations SET state='purging',updated_at=? WHERE dataset_id=? AND job_id=? AND state='reserved'",
                (utc_now(), dataset_id, job_id),
            ).rowcount
        if not changed:
            skipped.append({"dataset_id": dataset_id, "reason": "삭제 예약 상태 변경"})
            continue
        try:
            record = database.get_dataset_trash(dataset_id)
            if record["dataset"]["fingerprint"] != item["expected_fingerprint"]:
                raise ValueError("dataset changed")
            purge_dataset_from_trash(
                settings.nas_root,
                record,
                lease_check=lambda: database.assert_job_lease(
                    job_id, worker_id=worker_id
                ),
            )
            with database.connect() as connection:
                connection.execute(
                    "UPDATE dataset_purge_reservations SET state='purged',updated_at=? WHERE dataset_id=? AND job_id=?",
                    (utc_now(), dataset_id, job_id),
                )
            deleted.append(dataset_id)
        except Exception:
            with database.connect() as connection:
                connection.execute(
                    "UPDATE dataset_purge_reservations SET state='failed',updated_at=? WHERE dataset_id=? AND job_id=?",
                    (utc_now(), dataset_id, job_id),
                )
            database.mark_dataset_trash_recovery_required(dataset_id)
            failed.append(
                {
                    "dataset_id": dataset_id,
                    "reason": "영구 삭제가 중단되었습니다. 남은 파일은 관리자 확인이 필요합니다.",
                }
            )
    progress(
        {
            "stage": "delete",
            "completed": len(payload["items"]),
            "total": len(payload["items"]),
            "unit": "datasets",
            "_force": True,
        }
    )
    return {
        "deleted": deleted,
        "failed": failed,
        "skipped": skipped,
        "partial_failure": bool(failed),
    }
