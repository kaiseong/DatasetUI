"""Application -> queue entrypoint -> real encoded fixture -> approved export.

Only SAM mask prediction is injected. This is not a CUDA/checkpoint test.
"""

from dataclasses import replace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from datasetui.application import create_application
from datasetui.config import Settings
from datasetui.queueing import RecordingDispatcher
from datasetui.tasks import run_job
from test_segmentation import DeterministicEngine, _registered, _spec


def test_complete_approved_augmentation_flow(tmp_path, monkeypatch):
    settings, database, source, dataset = _registered(tmp_path)
    checkpoint = tmp_path / "fixture-model.pt"
    checkpoint.write_bytes(b"synthetic test engine, not a real SAM checkpoint")
    settings = replace(
        settings, sam3_checkpoint=checkpoint, sam3_checkpoint_sha256="a" * 64
    )
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(
        "datasetui.segmentation._default_engine",
        lambda settings, **kwargs: DeterministicEngine(),
    )
    app = create_application(
        database=database,
        dispatcher=RecordingDispatcher(),
        allowed_origins=["https://example.test"],
        legacy_app=FastAPI(),
        settings=settings,
    )
    client = TestClient(app)
    profile = client.post("/api/v1/profiles", json={"name": "E2E researcher"}).json()
    scope = client.get(f"/api/v1/segmentation/datasets/{dataset['id']}/scope")
    assert scope.status_code == 200, scope.text
    assert scope.json()["fingerprint"] == dataset["fingerprint"]
    frame = client.get(
        f"/api/v1/segmentation/datasets/{dataset['id']}/frame",
        params={
            "frame_token": scope.json()["frame_token"],
            "episode_index": 0,
            "frame_index": 1,
            "video_key": "observation.images.top",
        },
    )
    assert frame.status_code == 200
    assert frame.content.startswith(b"\x89PNG")
    created = client.post(
        "/api/v1/segmentation/previews",
        json={
            "profile_id": profile["id"],
            "idempotency_key": "e2e-preview",
            "spec": _spec(dataset),
        },
    )
    assert created.status_code == 202, created.text
    preview_id = created.json()["id"]
    assert run_job(preview_id)["status"] == "succeeded"
    result = client.get(f"/api/v1/jobs/{preview_id}").json()["result"]
    assert isinstance(
        result["model"], str
    ), "frontend expects a display name, not a provenance object"
    assert (
        client.get(
            f"/api/v1/segmentation/previews/{preview_id}/artifacts/composite.mp4"
        ).status_code
        == 200
    )
    approval = client.post(
        f"/api/v1/segmentation/previews/{preview_id}/approve",
        json={"profile_id": profile["id"], "recipe_hash": result["recipe_hash"]},
    )
    assert approval.status_code == 200, approval.text
    exported = client.post(
        "/api/v1/segmentation/exports",
        json={
            "profile_id": profile["id"],
            "preview_id": preview_id,
            "approval_token": approval.json()["approval_token"],
            "output_name": "researcher-chosen-name",
            "idempotency_key": "e2e-export",
        },
    )
    assert exported.status_code == 202, exported.text
    export_id = exported.json()["id"]
    assert run_job(export_id)["status"] == "succeeded"
    final = client.get(f"/api/v1/jobs/{export_id}").json()["result"]
    assert final["output_name"] == "researcher-chosen-name"
    replay = client.post(
        "/api/v1/segmentation/exports",
        json={
            "profile_id": profile["id"],
            "preview_id": preview_id,
            "approval_token": approval.json()["approval_token"],
            "output_name": "researcher-chosen-name",
            "idempotency_key": "e2e-export",
        },
    )
    assert replay.status_code == 202, replay.text
    assert replay.json()["id"] == export_id
    assert replay.json()["status"] == "succeeded"
    registered = client.get(f"/api/v1/datasets/{final['dataset_id']}").json()
    assert registered["readiness"] == "ready"
    output = settings.nas_root / "derived" / "researcher-chosen-name"
    relative = "data/chunk-000/episode_000000.parquet"
    assert (source / relative).read_bytes() == (output / relative).read_bytes()
    assert (
        client.post(
            "/api/v1/segmentation/exports",
            json={
                "profile_id": profile["id"],
                "preview_id": preview_id,
                "approval_token": approval.json()["approval_token"],
                "output_name": "researcher-chosen-name",
                "idempotency_key": "no-overwrite",
            },
        ).status_code
        == 409
    )
