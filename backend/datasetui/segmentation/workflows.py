"""Camera templates and episode x camera batches with durable review."""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
from uuid import uuid4

from datasetui.config import Settings
from datasetui.database import Database, IdempotencyConflictError, utc_now
from datasetui.delivery_workflow import dispatch_registered_job
from datasetui.queueing import QueueDispatcher
from datasetui.segmentation.contract import SegmentationSpec
from datasetui.segmentation.workflow_contract import (
    BatchCreate,
    BatchExportPayload,
    TemplateSave,
)

LOG = logging.getLogger(__name__)


MAX_BATCH_SPEC_BYTES = 8 * 1024 * 1024


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def initialize_segmentation_workflows(database: Database) -> None:
    with database.connect() as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS segmentation_templates (
                id TEXT PRIMARY KEY, profile_id TEXT NOT NULL REFERENCES profiles(id),
                name TEXT NOT NULL, snapshot_json TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL, archived_at TEXT
            );
            CREATE TABLE IF NOT EXISTS segmentation_batches (
                id TEXT PRIMARY KEY, profile_id TEXT NOT NULL REFERENCES profiles(id),
                template_id TEXT NOT NULL REFERENCES segmentation_templates(id),
                dataset_id TEXT NOT NULL REFERENCES datasets(id), fingerprint TEXT NOT NULL,
                source_registry_fingerprint TEXT,
                request_json TEXT NOT NULL, template_snapshot_json TEXT NOT NULL,
                idempotency_key TEXT NOT NULL, created_at TEXT NOT NULL,
                UNIQUE(profile_id, idempotency_key)
            );
            CREATE TABLE IF NOT EXISTS segmentation_batch_items (
                id TEXT PRIMARY KEY, batch_id TEXT NOT NULL REFERENCES segmentation_batches(id),
                ordinal INTEGER NOT NULL, episode_index INTEGER NOT NULL,
                video_key TEXT NOT NULL, spec_json TEXT NOT NULL,
                preview_id TEXT REFERENCES jobs(id), dispatch_error TEXT,
                approved_recipe_hash TEXT, approved_artifact_fingerprint TEXT,
                approved_token_hash TEXT, approved_at TEXT,
                UNIQUE(batch_id, episode_index, video_key), UNIQUE(batch_id, ordinal)
            );
        """)
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(segmentation_batches)")
        }
        if "source_registry_fingerprint" not in columns:
            connection.execute(
                "ALTER TABLE segmentation_batches ADD COLUMN source_registry_fingerprint TEXT"
            )


def _template(row) -> dict:
    return {
        **json.loads(row["snapshot_json"]),
        "id": row["id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "archived_at": row["archived_at"],
    }


def get_template(database: Database, template_id: str, profile_id: str) -> dict:
    database.get_profile(profile_id)
    with database.connect() as connection:
        row = connection.execute(
            "SELECT * FROM segmentation_templates WHERE id=? AND profile_id=? AND archived_at IS NULL",
            (template_id, profile_id),
        ).fetchone()
    if row is None:
        raise FileNotFoundError("템플릿을 찾을 수 없습니다.")
    return _template(row)


def list_templates(database: Database, profile_id: str) -> list[dict]:
    database.get_profile(profile_id)
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT * FROM segmentation_templates WHERE profile_id=? AND archived_at IS NULL ORDER BY updated_at DESC,id",
            (profile_id,),
        ).fetchall()
    return [_template(row) for row in rows]


def save_template(
    database: Database,
    settings: Settings,
    payload: TemplateSave,
    template_id: str | None = None,
) -> dict:
    from datasetui.segmentation.source import load_source

    profile_id = str(payload.profile_id)
    database.get_profile(profile_id)
    if template_id:
        get_template(database, template_id, profile_id)
    if payload.fingerprint:
        _, source = load_source(database, settings, str(payload.dataset_id), payload.fingerprint)
        keys = source.video_keys
    else:
        from datasetui.segmentation.catalog import dataset_catalog
        catalog = dataset_catalog(database, settings, str(payload.dataset_id))
        if catalog["metadata_revision"] != payload.metadata_revision:
            raise ValueError("데이터셋 메타정보가 변경되었습니다. 다시 선택하세요.")
        keys = catalog["video_keys"]
    if any(camera.video_key not in keys for camera in payload.cameras):
        raise ValueError("원본에 없는 카메라를 템플릿으로 저장할 수 없습니다.")
    now = utc_now()
    with database.connect() as connection:
        if template_id:
            count = connection.execute(
                "UPDATE segmentation_templates SET name=?,snapshot_json=?,updated_at=? WHERE id=? AND profile_id=? AND archived_at IS NULL",
                (
                    payload.name,
                    _json(payload.model_dump(mode="json")),
                    now,
                    template_id,
                    profile_id,
                ),
            ).rowcount
            if count != 1:
                raise FileNotFoundError("템플릿을 찾을 수 없습니다.")
        else:
            template_id = str(uuid4())
            connection.execute(
                "INSERT INTO segmentation_templates(id,profile_id,name,snapshot_json,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                (
                    template_id,
                    profile_id,
                    payload.name,
                    _json(payload.model_dump(mode="json")),
                    now,
                    now,
                ),
            )
    return get_template(database, template_id, profile_id)


def archive_template(database: Database, template_id: str, profile_id: str) -> None:
    get_template(database, template_id, profile_id)
    with database.connect() as connection:
        connection.execute(
            "UPDATE segmentation_templates SET archived_at=?,updated_at=? WHERE id=? AND profile_id=?",
            (utc_now(), utc_now(), template_id, profile_id),
        )


def _batch_row(database: Database, batch_id: str, profile_id: str):
    database.get_profile(profile_id)
    with database.connect() as connection:
        row = connection.execute(
            "SELECT * FROM segmentation_batches WHERE id=? AND profile_id=?",
            (batch_id, profile_id),
        ).fetchone()
    if row is None:
        raise FileNotFoundError("일괄 작업을 찾을 수 없습니다.")
    return row


def _item_rows(database: Database, batch_id: str):
    with database.connect() as connection:
        return connection.execute(
            "SELECT * FROM segmentation_batch_items WHERE batch_id=? ORDER BY ordinal",
            (batch_id,),
        ).fetchall()


def get_batch(database: Database, batch_id: str, profile_id: str) -> dict:
    batch = _batch_row(database, batch_id, profile_id)
    items = []
    for row in _item_rows(database, batch_id):
        job = database.get_job(row["preview_id"]) if row["preview_id"] else None
        result = job["result"] if job else None
        approved = bool(
            row["approved_at"]
            and job
            and job["status"] == "succeeded"
            and result
            and result.get("recipe_hash") == row["approved_recipe_hash"]
            and result.get("artifact_fingerprint")
            == row["approved_artifact_fingerprint"]
            and not result.get("selection_required")
            and not result.get("review_blocked")
            and row["approved_token_hash"]
        )
        items.append(
            {
                "id": row["id"],
                "episode_index": row["episode_index"],
                "video_key": row["video_key"],
                "preview_id": row["preview_id"],
                "job_status": job["status"] if job else "pending",
                "review_status": "approved" if approved else "pending",
                "dispatch_error": row["dispatch_error"]
                or (job.get("error_message") if job else None),
                "spec": {
                    key: value
                    for key, value in (
                        job["payload"]["spec"] if job else json.loads(row["spec_json"])
                    ).items()
                    if key != "background_base64"
                },
                "result": result,
                "started_at": job.get("started_at") if job else None,
                "finished_at": job.get("finished_at") if job else None,
                "progress": job.get("progress") if job else None,
                "approved_at": row["approved_at"] if approved else None,
            }
        )
    record = database.get_dataset(batch["dataset_id"])
    source_stale = (
        not record["available"]
        or record["readiness"] != "ready"
        or record["fingerprint"]
        != (batch["source_registry_fingerprint"] or batch["fingerprint"])
    )
    request = json.loads(batch["request_json"])
    template_snapshot = json.loads(batch["template_snapshot_json"])
    return {
        **request,
        "id": batch["id"],
        "warnings": reuse_warnings(template_snapshot, request),
        "template_snapshot": template_snapshot,
        "created_at": batch["created_at"],
        "source_stale": source_stale,
        "items": items,
        "ready_to_export": bool(items)
        and not source_stale
        and all(item["review_status"] == "approved" for item in items),
    }


def list_batches(database: Database, profile_id: str) -> list[dict]:
    database.get_profile(profile_id)
    with database.connect() as connection:
        ids = connection.execute(
            "SELECT id FROM segmentation_batches WHERE profile_id=? ORDER BY created_at DESC,id LIMIT 100",
            (profile_id,),
        ).fetchall()
    batches = []
    for row in ids:
        batch = get_batch(database, row["id"], profile_id)
        batch.pop("template_snapshot", None)
        for item in batch["items"]:
            item.pop("spec", None)
            item.pop("result", None)
        batches.append(batch)
    return batches


def _redetects_by_text(camera: dict, same_setup: bool) -> bool:
    return camera["camera_mode"] == "wrist" or (
        camera.get("reuse_policy") == "text_by_default" and not same_setup
    )


def reuse_warnings(template: dict, request: dict) -> list[str]:
    """Objects that cannot be carried to other episodes are reported, not hidden."""
    warnings = []
    for camera in template.get("cameras", []):
        if camera["video_key"] not in request.get("video_keys", []):
            continue
        if not _redetects_by_text(camera, request.get("same_camera_setup_confirmed", False)):
            continue
        items = [*camera.get("prompts", []), *camera.get("corrections", [])]
        objects = {item.get("object_id") for item in items}
        textual = {
            item.get("object_id")
            for item in camera.get("prompts", [])
            if item.get("text", "").strip()
        }
        dropped = sorted(
            (value for value in objects - textual if value is not None),
        )
        if dropped or (None in objects and None not in textual):
            names = ", ".join(f"객체 {value}" for value in dropped) or "번호 없는 객체"
            warnings.append(
                f"{camera['video_key']}: {names}는 텍스트 Instruction이 없어 "
                "대표 에피소드 외에는 적용되지 않습니다. 텍스트를 추가하거나 "
                "같은 설치·화각을 확인하세요."
            )
    return warnings


def _instantiate_camera(camera: dict, payload: BatchCreate, episode_index: int) -> dict:
    prompts = camera["prompts"]
    corrections = camera.get("corrections", [])
    manual_regions = camera.get("manual_regions", [])
    source_episode = camera.get("source_episode_index")
    if source_episode is not None and episode_index == source_episode:
        # The template was drawn on this exact video: use it unchanged.
        pass
    elif _redetects_by_text(camera, payload.same_camera_setup_confirmed):
        semantic_prompts = {}
        for prompt in prompts:
            if not prompt.get("text", "").strip():
                continue
            identity = (
                ("object", prompt["object_id"])
                if prompt.get("object_id") is not None
                else ("text", prompt["target"], prompt["text"])
            )
            semantic_prompts.setdefault(
                identity, {**prompt, "frame_index": 0, "points": [], "box": None, "selected_candidates": [], "member_candidate_id": None}
            )
        prompts = list(semantic_prompts.values())
        corrections, manual_regions = [], []
    elif not payload.same_camera_setup_confirmed:
        raise ValueError("고정 카메라의 설치 위치와 화각이 같음을 확인하세요.")
    if episode_index != camera.get("source_episode_index"):
        prompts = [{**p, "selected_candidates": [], "member_candidate_id": None} for p in prompts]
        corrections = [{**c, "member_candidate_id": None} for c in corrections]
    return SegmentationSpec.model_validate(
        {
            "dataset_id": str(payload.dataset_id),
            "fingerprint": payload.fingerprint,
            "episode_index": episode_index,
            "video_key": camera["video_key"],
            "mode": camera["mode"],
            "camera_mode": camera["camera_mode"],
            "render_mode": "black",
            "prompts": prompts,
            "corrections": corrections,
            "manual_regions": manual_regions,
        }
    ).model_dump(mode="json")


def create_batch(
    database: Database,
    dispatcher: QueueDispatcher,
    settings: Settings,
    payload: BatchCreate,
) -> dict:
    from datasetui.segmentation.source import load_source

    profile_id = str(payload.profile_id)
    database.get_profile(profile_id)
    request_json = _json(payload.model_dump(mode="json"))
    with database.connect() as connection:
        existing = connection.execute(
            "SELECT id,request_json FROM segmentation_batches WHERE profile_id=? AND idempotency_key=?",
            (profile_id, payload.idempotency_key),
        ).fetchone()
    if existing:
        if existing["request_json"] != request_json:
            raise IdempotencyConflictError(payload.idempotency_key)
        dispatch_batch(database, dispatcher, settings, existing["id"], profile_id)
        return get_batch(database, existing["id"], profile_id)
    template = get_template(database, str(payload.template_id), profile_id)
    cameras = {camera["video_key"]: camera for camera in template["cameras"]}
    if any(key not in cameras for key in payload.video_keys):
        raise ValueError("선택한 모든 카메라에 템플릿이 필요합니다.")
    _, source = load_source(
        database, settings, str(payload.dataset_id), payload.fingerprint
    )
    if any(key not in source.video_keys for key in payload.video_keys):
        raise ValueError("원본에 없는 카메라를 선택했습니다.")
    specs = []
    spec_bytes = 0
    for episode_index in payload.episode_indices:
        if episode_index >= int(source.info["total_episodes"]):
            raise ValueError("원본에 없는 에피소드를 선택했습니다.")
        frame, _ = source.episode(episode_index)
        if not 0 < len(frame) <= settings.segmentation_max_frames:
            raise ValueError("에피소드가 분할 작업 프레임 제한을 벗어났습니다.")
        for key in payload.video_keys:
            spec = _instantiate_camera(cameras[key], payload, episode_index)
            spec_bytes += len(_json(spec).encode())
            if spec_bytes > MAX_BATCH_SPEC_BYTES:
                raise ValueError(
                    "일괄 설정이 8 MiB를 초과합니다. 에피소드 또는 보정 점을 줄여 주세요."
                )
            hints = spec["prompts"] + spec["corrections"] + spec["manual_regions"]
            if any(
                hint.get("frame_index") is not None
                and hint["frame_index"] >= len(frame)
                for hint in hints
            ):
                raise ValueError(
                    "템플릿의 보정 프레임이 선택한 에피소드 범위를 벗어났습니다."
                )
            specs.append(spec)
    batch_id = str(uuid4())
    with database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            "SELECT id,request_json FROM segmentation_batches WHERE profile_id=? AND idempotency_key=?",
            (profile_id, payload.idempotency_key),
        ).fetchone()
        if existing:
            if existing["request_json"] != request_json:
                raise IdempotencyConflictError(payload.idempotency_key)
            batch_id = existing["id"]
        else:
            connection.execute(
                "INSERT INTO segmentation_batches(id,profile_id,template_id,dataset_id,fingerprint,source_registry_fingerprint,request_json,template_snapshot_json,idempotency_key,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    batch_id,
                    profile_id,
                    str(payload.template_id),
                    str(payload.dataset_id),
                    payload.fingerprint,
                    connection.execute(
                        "SELECT fingerprint FROM datasets WHERE id=?",
                        (str(payload.dataset_id),),
                    ).fetchone()["fingerprint"],
                    request_json,
                    _json(template),
                    payload.idempotency_key,
                    utc_now(),
                ),
            )
            connection.executemany(
                "INSERT INTO segmentation_batch_items(id,batch_id,ordinal,episode_index,video_key,spec_json) VALUES(?,?,?,?,?,?)",
                [
                    (
                        str(uuid4()),
                        batch_id,
                        index,
                        spec["episode_index"],
                        spec["video_key"],
                        _json(spec),
                    )
                    for index, spec in enumerate(specs)
                ],
            )
    dispatch_batch(database, dispatcher, settings, batch_id, profile_id)
    return get_batch(database, batch_id, profile_id)


def dispatch_batch(
    database: Database,
    dispatcher: QueueDispatcher,
    settings: Settings,
    batch_id: str,
    profile_id: str,
) -> None:
    from datasetui.segmentation.source import load_source

    batch = _batch_row(database, batch_id, profile_id)
    try:
        load_source(database, settings, batch["dataset_id"], batch["fingerprint"])
    except Exception:
        with database.connect() as connection:
            connection.execute(
                "UPDATE segmentation_batch_items SET dispatch_error=? WHERE batch_id=? AND preview_id IS NULL",
                (
                    "원본이 변경되었거나 사용할 수 없어 작업을 등록하지 않았습니다.",
                    batch_id,
                ),
            )
        raise
    for row in _item_rows(database, batch_id):
        try:
            if row["preview_id"]:
                job = database.get_job(row["preview_id"])
            else:
                job, _ = database.create_job(
                    kind="segmentation.preview",
                    queue_name="gpu"
                    if json.loads(row["spec_json"])["prompts"]
                    else "io",
                    profile_id=profile_id,
                    payload={"spec": json.loads(row["spec_json"])},
                    idempotency_key=f"seg-batch:{batch_id}:{row['id']}",
                )
                with database.connect() as connection:
                    connection.execute(
                        "UPDATE segmentation_batch_items SET preview_id=? WHERE id=? AND preview_id IS NULL",
                        (job["id"], row["id"]),
                    )
                # Another request may have bound a corrected preview meanwhile.
                with database.connect() as connection:
                    current = connection.execute(
                        "SELECT preview_id FROM segmentation_batch_items WHERE id=?",
                        (row["id"],),
                    ).fetchone()
                if current["preview_id"] != job["id"]:
                    continue
            if job["status"] == "queued":
                # Durable uncertainty marker closes the crash window between
                # storing an RQ id and actually sending it to Redis.
                database.record_dispatch_error(job["id"], "Unable to dispatch job")
                dispatch_registered_job(database, dispatcher, settings, job)
            with database.connect() as connection:
                connection.execute(
                    "UPDATE segmentation_batch_items SET dispatch_error=NULL WHERE id=?",
                    (row["id"],),
                )
        except Exception:
            with database.connect() as connection:
                connection.execute(
                    "UPDATE segmentation_batch_items SET dispatch_error=? WHERE id=?",
                    (
                        "작업 등록을 완료하지 못했습니다. 재시도하면 기존 작업을 이어서 등록합니다.",
                        row["id"],
                    ),
                )


def recover_segmentation_batches(
    database: Database, dispatcher: QueueDispatcher, settings: Settings
) -> None:
    with database.connect() as connection:
        rows = connection.execute("""SELECT DISTINCT b.id,b.profile_id FROM segmentation_batches b
            JOIN segmentation_batch_items i ON i.batch_id=b.id LEFT JOIN jobs j ON j.id=i.preview_id
            WHERE i.preview_id IS NULL OR (j.status='queued' AND (j.rq_job_id IS NULL OR j.error_code='queue_unavailable'))
            ORDER BY b.created_at LIMIT 20""").fetchall()
    for row in rows:
        try:
            dispatch_batch(database, dispatcher, settings, row["id"], row["profile_id"])
        except Exception:
            LOG.warning("Segmentation batch recovery delayed for %s", row["id"])


def _owned_item(database: Database, batch_id: str, item_id: str, profile_id: str):
    batch = _batch_row(database, batch_id, profile_id)
    with database.connect() as connection:
        row = connection.execute(
            "SELECT * FROM segmentation_batch_items WHERE id=? AND batch_id=?",
            (item_id, batch_id),
        ).fetchone()
    if row is None:
        raise FileNotFoundError("검토 항목을 찾을 수 없습니다.")
    return batch, row


def _matching_preview(
    database: Database, batch, item, preview_id: str, profile_id: str
):
    from datasetui.segmentation.preview import effective_preview_job

    job = database.get_job(preview_id)
    # Selection-token previews resolve their full fingerprint in the worker;
    # compare that verified identity, never the unresolved request field.
    spec = effective_preview_job(job)["payload"].get("spec", {})
    pending = spec.get("fingerprint") is None and job["status"] in {"queued", "running"}
    if (
        job["kind"] != "segmentation.preview"
        or job["profile_id"] != profile_id
        or spec.get("dataset_id") != batch["dataset_id"]
        or (not pending and spec.get("fingerprint") != batch["fingerprint"])
        or spec.get("episode_index") != item["episode_index"]
        or spec.get("video_key") != item["video_key"]
    ):
        raise ValueError("동일 프로필·원본·에피소드·카메라의 미리보기가 필요합니다.")
    return job


def bind_preview(
    database: Database,
    settings: Settings,
    batch_id: str,
    item_id: str,
    profile_id: str,
    preview_id: str,
) -> dict:
    from datasetui.segmentation.source import load_source

    batch, item = _owned_item(database, batch_id, item_id, profile_id)
    load_source(database, settings, batch["dataset_id"], batch["fingerprint"])
    job = _matching_preview(database, batch, item, preview_id, profile_id)
    if job["status"] not in {"queued", "running", "succeeded"}:
        raise ValueError("취소되거나 실패한 미리보기는 연결할 수 없습니다.")
    with database.connect() as connection:
        connection.execute(
            """UPDATE segmentation_batch_items SET preview_id=?,approved_recipe_hash=NULL,
            approved_artifact_fingerprint=NULL,approved_token_hash=NULL,approved_at=NULL,dispatch_error=NULL WHERE id=?""",
            (preview_id, item_id),
        )
    return get_batch(database, batch_id, profile_id)


def invalidate_item(
    database: Database, batch_id: str, item_id: str, profile_id: str
) -> dict:
    _owned_item(database, batch_id, item_id, profile_id)
    with database.connect() as connection:
        connection.execute(
            """UPDATE segmentation_batch_items SET approved_recipe_hash=NULL,
            approved_artifact_fingerprint=NULL,approved_token_hash=NULL,approved_at=NULL WHERE id=?""",
            (item_id,),
        )
    return get_batch(database, batch_id, profile_id)


def approve_item(
    database: Database,
    settings: Settings,
    batch_id: str,
    item_id: str,
    profile_id: str,
    recipe_hash: str,
) -> dict:
    from datasetui.segmentation.preview import verified_preview

    batch, item = _owned_item(database, batch_id, item_id, profile_id)
    if not item["preview_id"]:
        raise ValueError("완료된 미리보기가 필요합니다.")
    _matching_preview(database, batch, item, item["preview_id"], profile_id)
    # Batch export re-verifies the complete source before any output is written.
    _, result = verified_preview(
        database, settings, item["preview_id"], verify_source=False
    )
    if result.get("selection_required"):
        raise ValueError(
            "보존할 객체 후보를 명시적으로 선택하고 새 미리보기를 검토하세요."
        )
    if result.get("review_blocked"):
        raise ValueError(
            "영상 전체에서 대상 영역이 비어 있습니다. 프롬프트를 보정하세요."
        )
    if result["recipe_hash"] != recipe_hash:
        raise ValueError("현재 미리보기 설정을 확인하세요.")
    token = secrets.token_hex(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    with database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        updated = connection.execute(
            """UPDATE segmentation_batch_items SET approved_recipe_hash=?,approved_artifact_fingerprint=?,
            approved_token_hash=?,approved_at=? WHERE id=? AND preview_id=?""",
            (
                recipe_hash,
                result["artifact_fingerprint"],
                token_hash,
                utc_now(),
                item_id,
                item["preview_id"],
            ),
        ).rowcount
        if updated != 1:
            raise ValueError("검토 중 미리보기가 변경되었습니다. 다시 확인하세요.")
    return {"batch": get_batch(database, batch_id, profile_id), "approval_token": token}


def verify_batch_approvals(
    database: Database, settings: Settings, batch_id: str, profile_id: str
) -> tuple[list[str], list[str]]:
    from datasetui.segmentation.preview import verified_preview
    from datasetui.segmentation.source import load_source

    batch = _batch_row(database, batch_id, profile_id)
    load_source(database, settings, batch["dataset_id"], batch["fingerprint"])
    items = _item_rows(database, batch_id)
    request = json.loads(batch["request_json"])
    expected = {
        (episode, key)
        for episode in request["episode_indices"]
        for key in request["video_keys"]
    }
    if (
        not items
        or {(item["episode_index"], item["video_key"]) for item in items} != expected
    ):
        raise ValueError("원래 선택한 모든 에피소드·카메라가 필요합니다.")
    preview_ids, approval_hashes = [], []
    for item in items:
        if not item["approved_at"] or not item["preview_id"]:
            raise ValueError(
                "선택한 모든 결과를 검토하고 승인하세요. 누락 항목은 자동으로 제외하지 않습니다."
            )
        _matching_preview(database, batch, item, item["preview_id"], profile_id)
        _, result = verified_preview(
            database, settings, item["preview_id"], verify_source=False
        )
        if (
            result.get("selection_required")
            or result.get("review_blocked")
            or not item["approved_token_hash"]
            or result["recipe_hash"] != item["approved_recipe_hash"]
            or result["artifact_fingerprint"] != item["approved_artifact_fingerprint"]
        ):
            raise ValueError(
                "미리보기 또는 승인이 변경되었습니다. 전체 선택을 다시 검토하세요."
            )
        preview_ids.append(item["preview_id"])
        approval_hashes.append(item["approved_token_hash"])
    load_source(database, settings, batch["dataset_id"], batch["fingerprint"])
    fields = (
        "id",
        "preview_id",
        "approved_recipe_hash",
        "approved_artifact_fingerprint",
        "approved_token_hash",
        "approved_at",
    )
    if [
        tuple(row[key] for key in fields) for row in _item_rows(database, batch_id)
    ] != [tuple(row[key] for key in fields) for row in items]:
        raise ValueError("검증 중 검토 항목이 변경되었습니다. 다시 확인하세요.")
    return preview_ids, approval_hashes


def verify_batch_export(
    database: Database, settings: Settings, payload: dict
) -> list[str]:
    parsed = BatchExportPayload.model_validate(payload)
    preview_ids, hashes = verify_batch_approvals(
        database, settings, str(parsed.batch_id), str(parsed.profile_id)
    )
    if (
        preview_ids != [str(value) for value in parsed.preview_ids]
        or hashes != parsed.approval_hashes
    ):
        raise ValueError(
            "내보내기 등록 후 검토 항목이 변경되었습니다. 다시 승인해 새 작업을 등록하세요."
        )
    return preview_ids
