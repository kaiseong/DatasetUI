from __future__ import annotations

import base64
import hashlib
from dataclasses import replace
from io import BytesIO

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from datasetui.content_integrity import dataset_content_fingerprint
from datasetui.queueing import RecordingDispatcher
from datasetui.segmentation_api import create_segmentation_router
from datasetui.segmentation_contract import SegmentationSpec, decode_background
from test_transforms import _settings


def _image():
    buffer = BytesIO()
    Image.new("RGB", (8, 8), "blue").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _spec(dataset_id):
    return {
        "dataset_id": dataset_id,
        "fingerprint": "a" * 64,
        "episode_index": 0,
        "video_key": "observation.images.top",
        "mode": "replace_background",
        "background_base64": _image(),
        "prompts": [{"frame_index": 0, "target": "replace", "text": "wall"}],
        "corrections": [],
    }


def test_segmentation_inputs_reject_invalid_regions():
    from uuid import uuid4

    spec = _spec(str(uuid4()))
    assert SegmentationSpec.model_validate(spec).prompts[0].text == "wall"
    for replacement in (
        {"mode": "protect_foreground"},
        {"video_key": "../escape"},
        {"prompts": [{"frame_index": 0, "target": "replace", "box": [0.9, 0, 0.2, 1]}]},
        {
            "prompts": [
                {
                    "frame_index": 0,
                    "target": "replace",
                    "points": [{"x": float("nan"), "y": 0}],
                }
            ]
        },
    ):
        with pytest.raises(ValueError):
            SegmentationSpec.model_validate({**spec, **replacement})
    with pytest.raises(ValueError):
        decode_background("invalid%base64")


@pytest.fixture
def segmentation_client(tmp_path, database, monkeypatch):
    settings = _settings(tmp_path)
    checkpoint = tmp_path / "test-model.pt"
    checkpoint.write_bytes(b"test only")
    settings = replace(
        settings, sam3_checkpoint=checkpoint, sam3_checkpoint_sha256="b" * 64
    )
    dispatcher = RecordingDispatcher()
    app = FastAPI()
    app.include_router(create_segmentation_router(database, dispatcher, settings))
    monkeypatch.setattr(
        "datasetui.segmentation.load_source", lambda *args, **kwargs: None
    )
    return TestClient(app), settings, dispatcher


def test_preview_queue_failure_can_retry_same_request(segmentation_client, database):
    from uuid import uuid4

    client, _, dispatcher = segmentation_client
    profile = database.create_profile("Preview author")
    payload = {
        "profile_id": profile["id"],
        "idempotency_key": "preview-request",
        "spec": _spec(str(uuid4())),
    }
    dispatcher.available = False
    assert client.post("/api/v1/segmentation/previews", json=payload).status_code == 503
    dispatcher.available = True
    response = client.post("/api/v1/segmentation/previews", json=payload)
    assert response.status_code == 202, response.text
    assert response.json()["kind"] == "segmentation.preview"
    assert len(database.list_jobs()) == 1
    changed = {**payload, "spec": {**payload["spec"], "episode_index": 1}}
    assert client.post("/api/v1/segmentation/previews", json=changed).status_code == 409


def _finished_preview(database, settings, profile_id):
    from uuid import uuid4

    job, _ = database.create_job(
        kind="segmentation.preview",
        queue_name="gpu",
        profile_id=profile_id,
        payload={"spec": _spec(str(uuid4()))},
        idempotency_key="ready-preview",
    )
    root = settings.jobs_root / "segmentation" / job["id"]
    root.mkdir(parents=True)
    (root / "manifest.json").write_text("{}")
    (root / "composite.mp4").write_bytes(b"test artifact")
    database.claim_job(job["id"], worker_id="test")
    result = {
        "preview_id": job["id"],
        "recipe_hash": "c" * 64,
        "fingerprint": "a" * 64,
        "artifact_fingerprint": dataset_content_fingerprint(root),
        "artifact_hashes": {
            "composite.mp4": hashlib.sha256(
                (root / "composite.mp4").read_bytes()
            ).hexdigest()
        },
    }
    database.succeed_job(job["id"], result, worker_id="test")
    return job, root


def test_approval_binds_author_recipe_and_artifact(segmentation_client, database):
    client, settings, _ = segmentation_client
    author = database.create_profile("Author")
    other = database.create_profile("Other")
    job, root = _finished_preview(database, settings, author["id"])
    url = f"/api/v1/segmentation/previews/{job['id']}/approve"
    assert (
        client.post(
            url, json={"profile_id": other["id"], "recipe_hash": "c" * 64}
        ).status_code
        == 409
    )
    assert (
        client.post(
            url, json={"profile_id": author["id"], "recipe_hash": "d" * 64}
        ).status_code
        == 409
    )
    response = client.post(
        url, json={"profile_id": author["id"], "recipe_hash": "c" * 64}
    )
    assert response.status_code == 200
    token = response.json()["approval_token"]
    artifact_url = f"/api/v1/segmentation/previews/{job['id']}/artifacts/composite.mp4"
    assert client.get(artifact_url).status_code == 200
    payload = {
        "profile_id": author["id"],
        "preview_id": job["id"],
        "approval_token": token,
        "output_name": "my-augmented-dataset",
        "idempotency_key": "export-request",
    }
    assert client.post("/api/v1/segmentation/exports", json=payload).status_code == 202
    (root / "composite.mp4").write_bytes(b"changed after approval")
    assert client.get(artifact_url).status_code == 409
    assert (
        client.post(
            "/api/v1/segmentation/exports",
            json={**payload, "idempotency_key": "another-export"},
        ).status_code
        == 409
    )
    assert (
        client.post(
            "/api/v1/segmentation/exports", json={**payload, "output_name": "../escape"}
        ).status_code
        == 422
    )


def test_unconfigured_gpu_is_explicitly_unavailable(tmp_path, database):
    app = FastAPI()
    app.include_router(
        create_segmentation_router(database, RecordingDispatcher(), _settings(tmp_path))
    )
    client = TestClient(app)
    assert client.get("/api/v1/segmentation/capabilities").json()["configured"] is False
    profile = database.create_profile("Author")
    from uuid import uuid4

    response = client.post(
        "/api/v1/segmentation/previews",
        json={
            "profile_id": profile["id"],
            "idempotency_key": "disabled",
            "spec": _spec(str(uuid4())),
        },
    )
    assert response.status_code == 503
    assert database.list_jobs() == []
