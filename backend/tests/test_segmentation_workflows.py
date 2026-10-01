from __future__ import annotations

import hashlib
from dataclasses import replace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from datasetui.content_integrity import dataset_content_fingerprint
from datasetui.database import IdempotencyConflictError, RecipeRevisionMismatchError
from datasetui.queueing import RecordingDispatcher
from datasetui.segmentation_workflow_api import create_segmentation_workflow_router
from datasetui.segmentation_workflow_contract import (
    BatchCreate,
    CameraTemplate,
    TemplateSave,
)
from datasetui import segmentation_workflows as workflow
from test_segmentation import _registered


@pytest.fixture
def context(tmp_path):
    settings, database, root, dataset = _registered(tmp_path)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixture only")
    settings = replace(
        settings, sam3_checkpoint=checkpoint, sam3_checkpoint_sha256="a" * 64
    )
    profile = database.create_profile("Segmentation owner")
    dispatcher = RecordingDispatcher()
    workflow.initialize_segmentation_workflows(database)
    return settings, database, root, dataset, profile, dispatcher


def _template(context, *, wrist=False):
    settings, database, _, dataset, profile, _ = context
    cameras = [
        {
            "video_key": key,
            "camera_mode": "wrist" if wrist else "fixed",
            "prompts": [
                {
                    "frame_index": 1,
                    "target": "protect",
                    "text": "gripper",
                    "points": [{"x": 0.5, "y": 0.5}],
                    "box": [0.1, 0.1, 0.4, 0.4],
                }
            ],
            "corrections": [
                {
                    "frame_index": 1,
                    "target": "protect",
                    "radius": 0.1,
                    "points": [{"x": 0.2, "y": 0.2}],
                }
            ],
            "manual_regions": [{"box": [0.1, 0.2, 0.3, 0.4]}],
        }
        for key in ("observation.images.top", "observation.images.side")
    ]
    return workflow.save_template(
        database,
        settings,
        TemplateSave.model_validate(
            {
                "profile_id": profile["id"],
                "name": "Work area",
                "dataset_id": dataset["id"],
                "fingerprint": dataset["fingerprint"],
                "cameras": cameras,
            }
        ),
    )


def _request(context, template, **overrides):
    _, _, _, dataset, profile, _ = context
    return BatchCreate.model_validate(
        {
            "profile_id": profile["id"],
            "idempotency_key": "batch-1",
            "template_id": template["id"],
            "dataset_id": dataset["id"],
            "fingerprint": dataset["fingerprint"],
            "episode_indices": [0],
            "video_keys": ["observation.images.top", "observation.images.side"],
            "same_camera_setup_confirmed": True,
            **overrides,
        }
    )


def _batch(context, **overrides):
    settings, database, _, _, _, dispatcher = context
    template = _template(context)
    return workflow.create_batch(
        database, dispatcher, settings, _request(context, template, **overrides)
    )


def _finish(context, preview_id, *, selection_required=False, review_blocked=False):
    settings, database, _, dataset, _, _ = context
    root = settings.jobs_root / "segmentation" / preview_id
    root.mkdir(parents=True)
    (root / "manifest.json").write_text("{}")
    database.claim_job(preview_id, worker_id="test")
    result = {
        "preview_id": preview_id,
        "fingerprint": dataset["fingerprint"],
        "recipe_hash": hashlib.sha256(preview_id.encode()).hexdigest(),
        "artifact_fingerprint": dataset_content_fingerprint(root),
        "selection_required": selection_required,
        "review_blocked": review_blocked,
    }
    database.succeed_job(preview_id, result, worker_id="test")
    return result


def _approve_all(context, batch):
    settings, database, _, _, profile, _ = context
    for item in batch["items"]:
        result = _finish(context, item["preview_id"])
        workflow.approve_item(
            database,
            settings,
            batch["id"],
            item["id"],
            profile["id"],
            result["recipe_hash"],
        )
    return workflow.get_batch(database, batch["id"], profile["id"])


def test_templates_have_owner_boundaries_and_immutable_batch_snapshots(context):
    settings, database, _, _, profile, dispatcher = context
    template = _template(context)
    other = database.create_profile("Other")
    assert workflow.list_templates(database, other["id"]) == []
    with pytest.raises(FileNotFoundError):
        workflow.get_template(database, template["id"], other["id"])
    batch = workflow.create_batch(
        database, dispatcher, settings, _request(context, template)
    )
    updated = {
        key: value
        for key, value in template.items()
        if key not in {"id", "created_at", "updated_at", "archived_at"}
    }
    updated["name"] = "Changed template"
    workflow.save_template(
        database, settings, TemplateSave.model_validate(updated), template["id"]
    )
    assert (
        workflow.get_batch(database, batch["id"], profile["id"])["template_snapshot"][
            "name"
        ]
        == "Work area"
    )
    workflow.archive_template(database, template["id"], profile["id"])
    assert workflow.list_templates(database, profile["id"]) == []
    assert len(workflow.get_batch(database, batch["id"], profile["id"])["items"]) == 2
    with pytest.raises(FileNotFoundError):
        workflow.get_batch(database, batch["id"], other["id"])


def test_wrist_transfers_only_semantics_not_screen_coordinates(context):
    settings, database, _, _, _, dispatcher = context
    template = _template(context, wrist=True)
    batch = workflow.create_batch(
        database,
        dispatcher,
        settings,
        _request(context, template, same_camera_setup_confirmed=False),
    )
    for item in batch["items"]:
        spec = item["spec"]
        assert spec["camera_mode"] == "wrist"
        assert spec["render_mode"] == "black" and spec.get("background_base64") is None
        assert spec["manual_regions"] == [] and spec["corrections"] == []
        assert spec["prompts"][0]["frame_index"] == 0
        assert spec["prompts"][0]["points"] == [] and spec["prompts"][0]["box"] is None
        assert spec["prompts"][0]["text"] == "gripper"
    with pytest.raises(ValidationError):
        CameraTemplate.model_validate(
            {
                "video_key": "camera",
                "camera_mode": "wrist",
                "prompts": [
                    {"frame_index": 0, "target": "protect", "box": [0, 0, 1, 1]}
                ],
            }
        )


def test_wrist_reinitializes_one_semantic_prompt_per_corrected_object(context):
    settings, database, _, _, _, dispatcher = context
    template = _template(context, wrist=True)
    camera = template["cameras"][0]
    camera["prompts"][0]["object_id"] = 1
    camera["prompts"].append(
        {**camera["prompts"][0], "frame_index": 2, "points": [{"x": 0.9, "y": 0.9}]}
    )
    payload = {
        key: value
        for key, value in template.items()
        if key not in {"id", "created_at", "updated_at", "archived_at"}
    }
    saved = workflow.save_template(
        database, settings, TemplateSave.model_validate(payload), template["id"]
    )
    batch = workflow.create_batch(
        database,
        dispatcher,
        settings,
        _request(context, saved, same_camera_setup_confirmed=False),
    )
    prompts = batch["items"][0]["spec"]["prompts"]
    assert len(prompts) == 1
    assert prompts[0]["object_id"] == 1 and prompts[0]["frame_index"] == 0
    assert prompts[0]["points"] == [] and prompts[0]["box"] is None


def test_fixed_coordinates_require_explicit_matching_camera_setup(context):
    settings, database, _, _, _, dispatcher = context
    template = _template(context)
    with pytest.raises(ValueError, match="설치 위치"):
        workflow.create_batch(
            database,
            dispatcher,
            settings,
            _request(context, template, same_camera_setup_confirmed=False),
        )
    assert database.list_jobs() == []
    batch = workflow.create_batch(
        database, dispatcher, settings, _request(context, template)
    )
    assert batch["items"][0]["spec"]["manual_regions"]
    assert batch["items"][0]["spec"]["prompts"][0]["frame_index"] == 1


def test_batch_limits_finite_unique_selection_and_frame_bounds(context):
    settings, database, _, _, _, dispatcher = context
    template = _template(context)
    for changes in (
        {"episode_indices": [0, 0]},
        {"video_keys": ["camera", "camera"]},
        {"episode_indices": list(range(129))},
        {"episode_indices": [-1]},
        {"episode_indices": [0.1]},
    ):
        with pytest.raises(ValidationError):
            _request(context, template, **changes)
    with pytest.raises(ValueError, match="에피소드"):
        workflow.create_batch(
            database,
            dispatcher,
            settings,
            _request(context, template, episode_indices=[2]),
        )
    with pytest.raises(ValueError, match="모든 카메라"):
        workflow.create_batch(
            database,
            dispatcher,
            settings,
            _request(context, template, video_keys=["unknown"]),
        )
    with pytest.raises(ValidationError):
        CameraTemplate.model_validate(
            {
                "video_key": "camera",
                "camera_mode": "fixed",
                "prompts": [
                    {
                        "frame_index": 0,
                        "target": "protect",
                        "points": [{"x": float("nan"), "y": 0}],
                    }
                ],
            }
        )


def test_dispatch_failure_is_visible_and_recovery_does_not_duplicate_or_resurrect(
    context,
):
    settings, database, _, _, profile, dispatcher = context
    template = _template(context)
    dispatcher.available = False
    request = _request(context, template)
    batch = workflow.create_batch(database, dispatcher, settings, request)
    ids = [item["preview_id"] for item in batch["items"]]
    assert len(set(ids)) == 2 and len(database.list_jobs()) == 2
    assert all(item["dispatch_error"] for item in batch["items"])
    database.request_job_cancellation(ids[0], profile_id=profile["id"])
    dispatcher.available = True
    workflow.recover_segmentation_batches(database, dispatcher, settings)
    retried = workflow.create_batch(database, dispatcher, settings, request)
    assert [item["preview_id"] for item in retried["items"]] == ids
    assert retried["items"][0]["job_status"] == "cancelled"
    assert dispatcher.enqueued == [(ids[1], "gpu")]
    assert len(database.list_jobs()) == 2
    assert not retried["ready_to_export"]
    with pytest.raises(IdempotencyConflictError):
        workflow.create_batch(
            database,
            dispatcher,
            settings,
            _request(context, template, video_keys=["observation.images.top"]),
        )


def test_recovery_finds_job_created_before_batch_item_link(context):
    settings, database, _, _, profile, dispatcher = context
    batch = _batch(context)
    item = batch["items"][0]
    with database.connect() as connection:
        connection.execute(
            "UPDATE segmentation_batch_items SET preview_id=NULL WHERE id=?",
            (item["id"],),
        )
    workflow.recover_segmentation_batches(database, dispatcher, settings)
    result = workflow.get_batch(database, batch["id"], profile["id"])
    assert result["items"][0]["preview_id"] == item["preview_id"]
    assert len(database.list_jobs()) == 2


def test_stale_source_rejects_template_batch_and_export(context):
    settings, database, root, _, profile, dispatcher = context
    batch = _approve_all(context, _batch(context))
    (root / "README.md").write_text("changed source")
    with pytest.raises(RecipeRevisionMismatchError):
        workflow.verify_batch_approvals(database, settings, batch["id"], profile["id"])
    with pytest.raises(RecipeRevisionMismatchError):
        workflow.dispatch_batch(
            database, dispatcher, settings, batch["id"], profile["id"]
        )
    with pytest.raises(RecipeRevisionMismatchError):
        _template(context)


def test_export_requires_every_approval_and_worker_rechecks_invalidation(context):
    settings, database, _, _, profile, _ = context
    batch = _batch(context)
    first = batch["items"][0]
    result = _finish(context, first["preview_id"])
    workflow.approve_item(
        database,
        settings,
        batch["id"],
        first["id"],
        profile["id"],
        result["recipe_hash"],
    )
    with pytest.raises(ValueError, match="모든 결과"):
        workflow.verify_batch_approvals(database, settings, batch["id"], profile["id"])
    second = batch["items"][1]
    result = _finish(context, second["preview_id"])
    workflow.approve_item(
        database,
        settings,
        batch["id"],
        second["id"],
        profile["id"],
        result["recipe_hash"],
    )
    ids, hashes = workflow.verify_batch_approvals(
        database, settings, batch["id"], profile["id"]
    )
    payload = {
        "profile_id": profile["id"],
        "batch_id": batch["id"],
        "output_name": "segmented",
        "preview_ids": ids,
        "approval_hashes": hashes,
    }
    assert workflow.verify_batch_export(database, settings, payload) == ids
    workflow.invalidate_item(database, batch["id"], first["id"], profile["id"])
    with pytest.raises(ValueError):
        workflow.verify_batch_export(database, settings, payload)


def test_candidates_require_explicit_selection_before_approval(context):
    settings, database, _, _, profile, _ = context
    batch = _batch(context)
    item = batch["items"][0]
    result = _finish(context, item["preview_id"], selection_required=True)
    with pytest.raises(ValueError, match="객체 후보"):
        workflow.approve_item(
            database,
            settings,
            batch["id"],
            item["id"],
            profile["id"],
            result["recipe_hash"],
        )


def test_empty_clip_cannot_be_batch_approved_or_marked_export_ready(context):
    settings, database, _, _, profile, _ = context
    batch = _batch(context)
    item = batch["items"][0]
    result = _finish(context, item["preview_id"], review_blocked=True)
    with pytest.raises(ValueError, match="비어"):
        workflow.approve_item(
            database,
            settings,
            batch["id"],
            item["id"],
            profile["id"],
            result["recipe_hash"],
        )
    # Even a pre-fix persisted approval must remain invalid on reads and export.
    with database.connect() as connection:
        connection.execute(
            "UPDATE segmentation_batch_items SET approved_at='earlier',approved_token_hash=?,approved_recipe_hash=?,approved_artifact_fingerprint=? WHERE id=?",
            (
                "a" * 64,
                result["recipe_hash"],
                result["artifact_fingerprint"],
                item["id"],
            ),
        )
    reviewed = workflow.get_batch(database, batch["id"], profile["id"])
    assert reviewed["items"][0]["review_status"] == "pending"
    assert not reviewed["ready_to_export"]
    with pytest.raises(ValueError, match="변경"):
        workflow.verify_batch_approvals(database, settings, batch["id"], profile["id"])


def test_preview_binding_ownership_scope_and_approval_reset(context):
    settings, database, _, _, profile, _ = context
    batch = _approve_all(context, _batch(context))
    item = batch["items"][0]
    other = database.create_profile("Other owner")
    spec = item["spec"]
    unrelated, _ = database.create_job(
        kind="segmentation.preview",
        queue_name="gpu",
        profile_id=other["id"],
        payload={"spec": spec},
        idempotency_key="other",
    )
    with pytest.raises(ValueError, match="동일 프로필"):
        workflow.bind_preview(
            database, settings, batch["id"], item["id"], profile["id"], unrelated["id"]
        )
    with pytest.raises(ValueError):
        workflow.bind_preview(
            database,
            settings,
            batch["id"],
            item["id"],
            profile["id"],
            batch["items"][1]["preview_id"],
        )
    edited, _ = database.create_job(
        kind="segmentation.preview",
        queue_name="gpu",
        profile_id=profile["id"],
        payload={
            "spec": {
                **spec,
                "source_preview_id": item["preview_id"],
                "selected_candidate_ids": ["0-1"],
            }
        },
        idempotency_key="edited",
    )
    bound = workflow.bind_preview(
        database, settings, batch["id"], item["id"], profile["id"], edited["id"]
    )
    assert not bound["ready_to_export"]
    assert bound["items"][0]["review_status"] == "pending"
    assert bound["items"][0]["spec"]["selected_candidate_ids"] == ["0-1"]


def test_tampered_artifacts_and_missing_selection_fail_closed(context):
    settings, database, _, _, profile, _ = context
    batch = _approve_all(context, _batch(context))
    item = batch["items"][0]
    path = settings.jobs_root / "segmentation" / item["preview_id"] / "manifest.json"
    path.write_text("changed")
    with pytest.raises(ValueError, match="변경"):
        workflow.verify_batch_approvals(database, settings, batch["id"], profile["id"])
    with database.connect() as connection:
        connection.execute(
            "DELETE FROM segmentation_batch_items WHERE id=?", (item["id"],)
        )
    with pytest.raises(ValueError, match="모든 에피소드"):
        workflow.verify_batch_approvals(database, settings, batch["id"], profile["id"])


def test_api_export_is_one_job_for_entire_selection_and_idempotent(context):
    settings, database, _, _, profile, dispatcher = context
    app = FastAPI()
    app.include_router(
        create_segmentation_workflow_router(database, dispatcher, settings)
    )
    client = TestClient(app)
    batch = _approve_all(context, _batch(context))
    payload = {
        "profile_id": profile["id"],
        "idempotency_key": "export-one",
        "output_name": "black-scenes",
    }
    url = f"/api/v1/segmentation/batches/{batch['id']}/exports"
    response = client.post(url, json=payload)
    assert response.status_code == 202, response.text
    job = response.json()
    assert job["kind"] == "segmentation.batch_export"
    actual = database.get_job(job["id"])
    assert len(actual["payload"]["preview_ids"]) == 2
    assert actual["payload"]["recompute_statistics"] is False
    assert client.post(url, json=payload).json()["id"] == job["id"]
    assert client.post(url, json={**payload, "output_name": "other"}).status_code == 409
    assert (
        client.get(
            f"/api/v1/segmentation/batches/{batch['id']}",
            params={"profile_id": str(uuid4())},
        ).status_code
        == 404
    )


def test_api_dispatch_failure_returns_visible_batch_not_partial_success(context):
    settings, database, _, _, profile, dispatcher = context
    app = FastAPI()
    app.include_router(
        create_segmentation_workflow_router(database, dispatcher, settings)
    )
    client = TestClient(app)
    template = _template(context)
    dispatcher.available = False
    request = _request(context, template)
    response = client.post(
        "/api/v1/segmentation/batches", json=request.model_dump(mode="json")
    )
    assert response.status_code == 202, response.text
    assert len(response.json()["items"]) == 2
    assert all(item["dispatch_error"] for item in response.json()["items"])
    assert (
        len(
            client.get(
                "/api/v1/segmentation/batches", params={"profile_id": profile["id"]}
            ).json()["batches"]
        )
        == 1
    )


def test_partial_enqueue_ack_loss_recovers_same_jobs(context, monkeypatch):
    settings, database, _, _, profile, dispatcher = context
    original = RecordingDispatcher.enqueue
    calls = 0

    def ack_lost(self, **kwargs):
        nonlocal calls
        original(self, **kwargs)
        calls += 1
        if calls == 1:
            raise ConnectionError("acknowledgement lost after enqueue")

    monkeypatch.setattr(RecordingDispatcher, "enqueue", ack_lost)
    batch = _batch(context)
    ids = [item["preview_id"] for item in batch["items"]]
    assert batch["items"][0]["dispatch_error"]
    workflow.recover_segmentation_batches(database, dispatcher, settings)
    assert len(database.list_jobs()) == len(dispatcher.enqueued) == 2
    assert [
        item["preview_id"]
        for item in workflow.get_batch(database, batch["id"], profile["id"])["items"]
    ] == ids


def test_failed_jobs_remain_required_and_can_be_explicitly_replaced(context):
    settings, database, _, _, profile, dispatcher = context
    batch = _batch(context)
    item = batch["items"][0]
    database.claim_job(item["preview_id"], worker_id="test")
    database.fail_job(
        item["preview_id"], "test_failure", "test failure", worker_id="test"
    )
    workflow.dispatch_batch(database, dispatcher, settings, batch["id"], profile["id"])
    assert database.get_job(item["preview_id"])["status"] == "failed"
    with pytest.raises(ValueError):
        workflow.verify_batch_approvals(database, settings, batch["id"], profile["id"])
    with pytest.raises(ValueError):
        workflow.bind_preview(
            database,
            settings,
            batch["id"],
            item["id"],
            profile["id"],
            item["preview_id"],
        )


def test_rebind_and_reapproval_invalidate_queued_export_snapshot(context):
    settings, database, _, _, profile, _ = context
    batch = _approve_all(context, _batch(context))
    ids, hashes = workflow.verify_batch_approvals(
        database, settings, batch["id"], profile["id"]
    )
    payload = {
        "profile_id": profile["id"],
        "batch_id": batch["id"],
        "output_name": "scene",
        "preview_ids": ids,
        "approval_hashes": hashes,
    }
    first = batch["items"][0]
    rerender, _ = database.create_job(
        kind="segmentation.preview",
        queue_name="gpu",
        profile_id=profile["id"],
        payload={"spec": first["spec"]},
        idempotency_key="rerender",
    )
    workflow.bind_preview(
        database, settings, batch["id"], first["id"], profile["id"], rerender["id"]
    )
    result = _finish(context, rerender["id"])
    workflow.approve_item(
        database,
        settings,
        batch["id"],
        first["id"],
        profile["id"],
        result["recipe_hash"],
    )
    assert workflow.get_batch(database, batch["id"], profile["id"])["ready_to_export"]
    with pytest.raises(ValueError, match="등록 후"):
        workflow.verify_batch_export(database, settings, payload)


def test_template_and_expanded_batch_byte_budgets(context, monkeypatch):
    from datasetui import segmentation_workflow_contract

    settings, database, _, _, _, dispatcher = context
    template = _template(context)
    monkeypatch.setattr(segmentation_workflow_contract, "MAX_TEMPLATE_BYTES", 10)
    with pytest.raises(ValidationError, match="MiB"):
        _template(context)
    monkeypatch.setattr(workflow, "MAX_BATCH_SPEC_BYTES", 10)
    with pytest.raises(ValueError, match="MiB"):
        workflow.create_batch(
            database, dispatcher, settings, _request(context, template)
        )
    assert database.list_jobs() == []


def test_metadata_only_registry_marker_is_distinct_from_content_revision(context):
    settings, database, root, dataset, profile, _ = context
    marker = hashlib.sha256((root / "meta/info.json").read_bytes()).hexdigest()
    assert marker != dataset["fingerprint"]
    with database.connect() as connection:
        connection.execute(
            "UPDATE datasets SET fingerprint=? WHERE id=?", (marker, dataset["id"])
        )
    batch = _approve_all(context, _batch(context))
    assert batch["fingerprint"] == dataset["fingerprint"]
    assert not batch["source_stale"] and batch["ready_to_export"]
    assert (
        len(
            workflow.verify_batch_approvals(
                database, settings, batch["id"], profile["id"]
            )[0]
        )
        == 2
    )


def test_approval_drift_marks_review_pending_even_before_export(context):
    settings, database, _, _, profile, _ = context
    batch = _approve_all(context, _batch(context))
    with database.connect() as connection:
        connection.execute(
            "UPDATE segmentation_batch_items SET approved_artifact_fingerprint=? WHERE id=?",
            ("f" * 64, batch["items"][0]["id"]),
        )
    assert not workflow.get_batch(database, batch["id"], profile["id"])[
        "ready_to_export"
    ]
    with pytest.raises(ValueError):
        workflow.verify_batch_approvals(database, settings, batch["id"], profile["id"])


def test_independent_batches_can_review_same_preview_without_revoking_each_other(
    context,
):
    settings, database, _, _, profile, dispatcher = context
    first = _approve_all(context, _batch(context))
    template = _template(context)
    second = workflow.create_batch(
        database,
        dispatcher,
        settings,
        _request(context, template, idempotency_key="batch-two"),
    )
    for original, target in zip(first["items"], second["items"], strict=True):
        workflow.bind_preview(
            database,
            settings,
            second["id"],
            target["id"],
            profile["id"],
            original["preview_id"],
        )
        workflow.approve_item(
            database,
            settings,
            second["id"],
            target["id"],
            profile["id"],
            original["result"]["recipe_hash"],
        )
    assert workflow.get_batch(database, first["id"], profile["id"])["ready_to_export"]
    assert workflow.get_batch(database, second["id"], profile["id"])["ready_to_export"]
    first_ids, first_hashes = workflow.verify_batch_approvals(
        database, settings, first["id"], profile["id"]
    )
    second_ids, second_hashes = workflow.verify_batch_approvals(
        database, settings, second["id"], profile["id"]
    )
    assert first_ids == second_ids and first_hashes != second_hashes
    workflow.invalidate_item(
        database, second["id"], second["items"][0]["id"], profile["id"]
    )
    assert workflow.get_batch(database, first["id"], profile["id"])["ready_to_export"]


def test_invalidation_during_expensive_verification_fails_closed(context, monkeypatch):
    from datasetui import segmentation_api

    settings, database, _, _, profile, _ = context
    batch = _approve_all(context, _batch(context))
    original = segmentation_api.verified_preview
    invalidated = False

    def verify_then_edit(*args, **kwargs):
        nonlocal invalidated
        result = original(*args, **kwargs)
        if not invalidated:
            invalidated = True
            workflow.invalidate_item(
                database, batch["id"], batch["items"][0]["id"], profile["id"]
            )
        return result

    monkeypatch.setattr(segmentation_api, "verified_preview", verify_then_edit)
    with pytest.raises(ValueError, match="검증 중"):
        workflow.verify_batch_approvals(database, settings, batch["id"], profile["id"])


def test_fixed_manual_only_template_batches_without_model_but_wrist_rejects(context):
    settings, database, _, dataset, profile, dispatcher = context
    settings = replace(settings, sam3_checkpoint=None, sam3_checkpoint_sha256="")
    camera = {
        "video_key": "observation.images.top",
        "camera_mode": "fixed",
        "manual_regions": [{"box": [0, 0, 0.5, 0.5]}],
    }
    template = workflow.save_template(
        database,
        settings,
        TemplateSave.model_validate(
            {
                "profile_id": profile["id"],
                "dataset_id": dataset["id"],
                "fingerprint": dataset["fingerprint"],
                "name": "Manual protected work area",
                "cameras": [camera],
            }
        ),
    )
    app = FastAPI()
    app.include_router(
        create_segmentation_workflow_router(database, dispatcher, settings)
    )
    request = _request(context, template, video_keys=[camera["video_key"]])
    response = TestClient(app).post(
        "/api/v1/segmentation/batches", json=request.model_dump(mode="json")
    )
    assert response.status_code == 202, response.text
    assert response.json()["items"][0]["spec"]["prompts"] == []
    assert (
        database.get_job(response.json()["items"][0]["preview_id"])["queue_name"]
        == "io"
    )
    with pytest.raises(ValidationError, match="손목"):
        CameraTemplate.model_validate({**camera, "camera_mode": "wrist"})
