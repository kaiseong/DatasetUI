from fastapi import FastAPI
from fastapi.testclient import TestClient
from datasetui.queueing import RecordingDispatcher
from datasetui.segmentation_api import create_segmentation_router
from datasetui.segmentation_workflow_api import create_segmentation_workflow_router
from datasetui.segmentation_catalog import dataset_catalog, selected_scope
from datasetui.segmentation_workflows import _instantiate_camera
from datasetui.segmentation_workflow_contract import BatchCreate
from test_segmentation import _registered


def test_lazy_save_preview_and_batch_enqueue_do_not_full_scan(tmp_path, monkeypatch):
    settings, database, _, dataset = _registered(tmp_path)
    profile = database.create_profile("lazy owner")
    dispatcher = RecordingDispatcher()
    app = FastAPI()
    app.include_router(create_segmentation_router(database, dispatcher, settings))
    app.include_router(
        create_segmentation_workflow_router(database, dispatcher, settings)
    )
    catalog = dataset_catalog(database, settings, dataset["id"])
    scope = selected_scope(
        database, settings, dataset["id"], 0, "observation.images.top"
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("full scan on HTTP request")

    monkeypatch.setattr("datasetui.segmentation.load_source", forbidden)
    with TestClient(app) as client:
        camera = {
            "video_key": "observation.images.top",
            "camera_mode": "fixed",
            "manual_regions": [{"frame_index": 0, "box": [0.1, 0.1, 0.2, 0.2]}],
            "reuse_policy": "text_by_default",
        }
        result = client.post(
            "/api/v1/segmentation/templates",
            json={
                "profile_id": profile["id"],
                "dataset_id": dataset["id"],
                "metadata_revision": catalog["metadata_revision"],
                "name": "sample",
                "cameras": [camera],
            },
        )
        assert result.status_code == 201, result.text
        template = result.json()
        response = client.post(
            "/api/v1/segmentation/previews",
            json={
                "profile_id": profile["id"],
                "idempotency_key": "lazy-preview",
                "spec": {
                    "dataset_id": dataset["id"],
                    "frame_token": scope["frame_token"],
                    "episode_index": 0,
                    "video_key": camera["video_key"],
                    "manual_regions": camera["manual_regions"],
                },
            },
        )
        assert response.status_code == 202, response.text
        assert response.json()["payload"]["spec"]["fingerprint"] is None
        response = client.post(
            "/api/v1/segmentation/batches/prepare",
            json={
                "profile_id": profile["id"],
                "idempotency_key": "lazy-batch",
                "template_id": template["id"],
                "dataset_id": dataset["id"],
                "metadata_revision": catalog["metadata_revision"],
                "episode_indices": [0],
                "video_keys": [camera["video_key"]],
            },
        )
        assert response.status_code == 202, response.text
        assert response.json()["kind"] == "segmentation.batch_prepare"
        assert len(dispatcher.enqueued) == 2


def test_new_templates_redetect_without_spatial_hints(tmp_path):
    import uuid

    settings, database, _, dataset = _registered(tmp_path)
    payload = BatchCreate.model_validate(
        {
            "profile_id": str(uuid.uuid4()),
            "template_id": str(uuid.uuid4()),
            "idempotency_key": "test",
            "dataset_id": dataset["id"],
            "fingerprint": dataset["fingerprint"],
            "episode_indices": [0],
            "video_keys": ["observation.images.top"],
        }
    )
    camera = {
        "video_key": "observation.images.top",
        "camera_mode": "fixed",
        "reuse_policy": "text_by_default",
        "mode": "protect_foreground",
        "prompts": [
            {
                "frame_index": 3,
                "target": "protect",
                "text": "plug",
                "points": [{"x": 0.1, "y": 0.2}],
            }
        ],
    }
    spec = _instantiate_camera(camera, payload, 0)
    assert spec["prompts"][0]["text"] == "plug"
    assert spec["prompts"][0]["frame_index"] == 0
    assert spec["prompts"][0]["points"] == []
    assert spec["corrections"] == []


def test_pending_execution_exposes_real_fingerprint_without_mutating_request(tmp_path):
    from datasetui.segmentation import create_preview
    from datasetui.segmentation_api import verified_preview
    from test_segmentation import DeterministicEngine

    settings, database, _, dataset = _registered(tmp_path)
    profile = database.create_profile("pending owner")
    scope = selected_scope(
        database, settings, dataset["id"], 0, "observation.images.top"
    )
    spec = {
        "dataset_id": dataset["id"],
        "frame_token": scope["frame_token"],
        "episode_index": 0,
        "video_key": "observation.images.top",
        "prompts": [{"target": "protect", "frame_index": 0, "text": "plug"}],
    }
    job, _ = database.create_job(
        kind="segmentation.preview",
        queue_name="gpu",
        profile_id=profile["id"],
        payload={"spec": spec},
        idempotency_key="pending",
    )
    database.claim_job(job["id"], worker_id="gpu", lease_seconds=120)
    result = create_preview(
        database,
        settings,
        job_id=job["id"],
        worker_id="gpu",
        spec=spec,
        engine=DeterministicEngine(),
    )
    database.succeed_job(job["id"], result, worker_id="gpu")
    effective, _ = verified_preview(database, settings, job["id"])
    assert effective["payload"]["spec"]["fingerprint"] == dataset["fingerprint"]
    assert "frame_token" not in effective["payload"]["spec"]
    assert not database.get_job(job["id"])["payload"]["spec"].get("fingerprint")
