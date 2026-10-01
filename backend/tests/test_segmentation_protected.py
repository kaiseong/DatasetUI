from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest
from PIL import Image

from datasetui.content_integrity import dataset_content_fingerprint
from datasetui.deferred_statistics import read_deferred_statistics
from datasetui.segmentation import (
    SegmentationError,
    _iter_video_arrays,
    create_preview,
    export_preview,
)
from datasetui.segmentation_contract import SegmentationSpec
from test_segmentation import DeterministicEngine, _registered, _spec, _write_video


def make_preview(settings, database, profile, spec, *, engine=None):
    job, _ = database.create_job(
        kind="segmentation.preview",
        queue_name="gpu",
        profile_id=profile["id"],
        payload={"spec": spec},
        idempotency_key=str(uuid4()),
    )
    database.claim_job(job["id"], worker_id="gpu", lease_seconds=120)
    result = create_preview(
        database,
        settings,
        job_id=job["id"],
        worker_id="gpu",
        spec=spec,
        engine=engine or DeterministicEngine(),
    )
    database.succeed_job(job["id"], result, worker_id="gpu")
    return job, result


def black_spec(dataset, **kwargs):
    return {
        **_spec(dataset),
        "mode": "protect_foreground",
        "render_mode": "black",
        "background_base64": None,
        "prompts": [
            {
                "object_id": 1,
                "frame_index": 0,
                "target": "protect",
                "points": [{"x": 0.25, "y": 0.5}],
            }
        ],
        "corrections": [],
        **kwargs,
    }


class CandidateEngine(DeterministicEngine):
    def propagate(self, **kwargs):
        result = super().propagate(**kwargs)
        root, count = kwargs["output_dir"], kwargs["frame_count"]
        candidates = []
        for number in (1, 2):
            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[:, :8] = 255 if number == 1 else 0
            mask[:, 8:] = 255 if number == 2 else 0
            directory = root / "instances" / f"1-{number}"
            directory.mkdir(parents=True)
            for index in range(count):
                Image.fromarray(mask).save(directory / f"{index:06d}.png")
            Image.fromarray(mask).save(root / f"candidate-1-{number}.png")
            candidates.append(
                {
                    "candidate_id": f"1-{number}",
                    "object_id": 1,
                    "sam_object_id": number,
                    "target": "protect",
                    "frame_index": 0,
                    "area_pixels": 128,
                    "artifact_name": f"candidate-1-{number}.png",
                    "requires_selection": True,
                }
            )
        return {**result, "candidates": candidates}


class NeverInfer:
    def propagate(self, **kwargs):
        pytest.fail("Cached mask rendering must not run inference")


def test_black_default_legacy_image_and_wrist_region_validation():
    base = {
        "dataset_id": str(uuid4()),
        "fingerprint": "a" * 64,
        "episode_index": 0,
        "video_key": "top",
        "prompts": [{"frame_index": 0, "target": "protect", "text": "plug"}],
    }
    assert SegmentationSpec.model_validate(base).render_mode == "black"
    legacy = _spec({"id": base["dataset_id"], "fingerprint": base["fingerprint"]})
    assert SegmentationSpec.model_validate(legacy).render_mode == "image"
    with pytest.raises(ValueError, match="배경 이미지"):
        SegmentationSpec.model_validate({**base, "render_mode": "image"})
    with pytest.raises(ValueError, match="손목"):
        SegmentationSpec.model_validate(
            {
                **base,
                "camera_mode": "wrist",
                "manual_regions": [{"box": [0, 0, 0.5, 0.5]}],
            }
        )
    with pytest.raises(ValueError):
        SegmentationSpec.model_validate(
            {**base, "manual_regions": [{"points": [{"x": 0, "y": 0}]}]}
        )


def test_cache_selection_black_render_and_changed_mask_inputs_fail(tmp_path):
    settings, database, source, dataset = _registered(tmp_path)
    original = dataset_content_fingerprint(source)
    profile = database.create_profile("Candidate reviewer")
    spec = black_spec(
        dataset,
        prompts=[
            {"object_id": 1, "frame_index": 0, "target": "protect", "text": "plug"}
        ],
    )
    first, discovered = make_preview(
        settings, database, profile, spec, engine=CandidateEngine()
    )
    assert discovered["selection_required"] is True
    assert len(discovered["candidates"]) == 2
    selected_spec = {
        **spec,
        "source_preview_id": first["id"],
        "selected_candidate_ids": ["1-1"],
    }
    second, selected = make_preview(
        settings, database, profile, selected_spec, engine=NeverInfer()
    )
    assert selected["selection_required"] is False
    root = settings.jobs_root / "segmentation" / second["id"]
    with Image.open(root / "background.png") as image:
        assert not np.asarray(image).any()
    with Image.open(root / "protect/000000.png") as image:
        mask = np.asarray(image)
        assert (mask[:, :8] == 255).all() and not mask[:, 8:].any()
    with pytest.raises(SegmentationError, match="Mask inputs changed"):
        make_preview(
            settings,
            database,
            profile,
            {**selected_spec, "manual_regions": [{"box": [0, 0, 0.5, 0.5]}]},
            engine=NeverInfer(),
        )
    assert dataset_content_fingerprint(source) == original
    other = database.create_profile("Other mask owner")
    with pytest.raises(SegmentationError, match="original preview profile"):
        make_preview(settings, database, other, selected_spec, engine=NeverInfer())


@pytest.mark.parametrize("camera_mode", ["fixed", "wrist"])
def test_manual_protection_obeys_camera_coordinate_scope(tmp_path, camera_mode):
    settings, database, _, dataset = _registered(tmp_path)
    profile = database.create_profile("Region reviewer")
    spec = black_spec(
        dataset,
        camera_mode=camera_mode,
        manual_regions=[{"frame_index": 1, "box": [0.5, 0.1, 0.4, 0.4]}],
    )
    job, _ = make_preview(settings, database, profile, spec)
    root = settings.jobs_root / "segmentation" / job["id"]
    for index in range(4):
        mask = np.asarray(Image.open(root / "protect" / f"{index:06d}.png"))
        assert bool(mask[3, 10]) == (camera_mode == "fixed" or index == 1)


def test_manual_only_preview_and_legacy_registry_identity(tmp_path):
    import hashlib
    from datasetui.segmentation import dataset_scope, load_source
    from datasetui.database import RecipeRevisionMismatchError

    settings, database, source, dataset = _registered(tmp_path)
    legacy = hashlib.sha256((source / "meta/info.json").read_bytes()).hexdigest()
    with database.connect() as connection:
        connection.execute(
            "UPDATE datasets SET fingerprint = ? WHERE id = ?", (legacy, dataset["id"])
        )
    scope = dataset_scope(database, settings, dataset["id"])
    assert scope["fingerprint"] != legacy
    profile = database.create_profile("Manual reviewer")
    job, result = make_preview(
        settings,
        database,
        profile,
        black_spec(
            dataset,
            fingerprint=scope["fingerprint"],
            prompts=[],
            manual_regions=[{"box": [0, 0, 0.5, 0.5]}],
        ),
        engine=NeverInfer(),
    )
    assert result["model"] == "manual-protection"
    assert result["selection_required"] is False
    assert (
        len(
            list(
                _iter_video_arrays(
                    settings.jobs_root / "segmentation" / job["id"] / "composite.mp4"
                )
            )
        )
        == 4
    )
    (source / "README.md").write_text("mutation after mask inference")
    with pytest.raises(RecipeRevisionMismatchError):
        load_source(database, settings, dataset["id"], scope["fingerprint"])


def test_multicamera_export_deferred_and_no_source_mutation(tmp_path, monkeypatch):
    settings, database, source, dataset = _registered(tmp_path)
    fingerprint = dataset_content_fingerprint(source)
    profile = database.create_profile("Multi camera")
    previews = [
        make_preview(settings, database, profile, black_spec(dataset, video_key=key))[
            0
        ]["id"]
        for key in ("observation.images.top", "observation.images.side")
    ]
    import datasetui.segmentation as implementation

    monkeypatch.setattr(
        implementation,
        "_update_image_statistics",
        lambda *a, **k: pytest.fail("Statistics must remain deferred"),
    )
    job, _ = database.create_job(
        kind="segmentation.export",
        queue_name="io",
        profile_id=profile["id"],
        payload={},
        idempotency_key="multi",
    )
    database.claim_job(job["id"], worker_id="io", lease_seconds=120)
    export_preview(
        database,
        settings,
        job_id=job["id"],
        worker_id="io",
        preview_ids=previews,
        output_name="multi",
    )
    output = settings.nas_root / "derived/multi"
    assert read_deferred_statistics(output)
    assert dataset_content_fingerprint(source) == fingerprint
    for video in source.rglob("*.mp4"):
        assert len(list(_iter_video_arrays(output / video.relative_to(source)))) == 4
        assert video.read_bytes() != (output / video.relative_to(source)).read_bytes()
    assert (source / "data/chunk-000/episode_000000.parquet").read_bytes() == (
        output / "data/chunk-000/episode_000000.parquet"
    ).read_bytes()


def test_api_requires_explicit_text_selection_and_serves_verified_candidates(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from datasetui.queueing import RecordingDispatcher
    from datasetui.segmentation_api import create_segmentation_router

    settings, database, _, dataset = _registered(tmp_path)
    profile = database.create_profile("Reviewer")
    spec = black_spec(
        dataset,
        prompts=[
            {"object_id": 1, "frame_index": 0, "target": "protect", "text": "plug"}
        ],
    )
    first, result = make_preview(
        settings, database, profile, spec, engine=CandidateEngine()
    )
    app = FastAPI()
    app.include_router(
        create_segmentation_router(database, RecordingDispatcher(), settings)
    )
    client = TestClient(app)
    url = f"/api/v1/segmentation/previews/{first['id']}"
    approve_payload = {
        "profile_id": profile["id"],
        "recipe_hash": result["recipe_hash"],
    }
    assert client.post(url + "/approve", json=approve_payload).status_code == 409
    assert client.get(url, params={"profile_id": profile["id"]}).json()["spec"] == spec
    artifact = client.get(url + "/artifacts/candidate-1-1.png")
    assert (
        artifact.status_code == 200 and artifact.headers["content-type"] == "image/png"
    )
    rerender = client.post(
        "/api/v1/segmentation/previews",
        json={
            "profile_id": profile["id"],
            "idempotency_key": "cache-without-checkpoint",
            "spec": {
                **spec,
                "source_preview_id": first["id"],
                "selected_candidate_ids": ["1-1"],
            },
        },
    )
    assert rerender.status_code == 202, rerender.text
    assert rerender.json()["queue_name"] == "io"
    candidate = settings.jobs_root / "segmentation" / first["id"] / "candidate-1-1.png"
    candidate.write_bytes(candidate.read_bytes() + b"changed")
    assert client.get(url + "/artifacts/candidate-1-1.png").status_code == 409


def test_empty_track_cannot_be_approved_even_with_explicit_review(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from datasetui.queueing import RecordingDispatcher
    from datasetui.segmentation_api import create_segmentation_router

    class EmptyEngine(DeterministicEngine):
        def propagate(self, **kwargs):
            result = super().propagate(**kwargs)
            for path in (kwargs["output_dir"] / "protect").glob("*.png"):
                Image.new("L", (16, 16), 0).save(path)
            return result

    settings, database, _, dataset = _registered(tmp_path)
    profile = database.create_profile("Empty mask reviewer")
    job, result = make_preview(
        settings, database, profile, black_spec(dataset), engine=EmptyEngine()
    )
    assert result["review_blocked"] is True
    assert result["review_signals"] == [
        {"frame_index": index, "reason": "empty_mask"} for index in range(4)
    ]
    app = FastAPI()
    app.include_router(
        create_segmentation_router(database, RecordingDispatcher(), settings)
    )
    response = TestClient(app).post(
        f"/api/v1/segmentation/previews/{job['id']}/approve",
        json={"profile_id": profile["id"], "recipe_hash": result["recipe_hash"]},
    )
    assert response.status_code == 409


def test_long_object_brush_covers_full_stroke_and_keeps_radius_at_keyframe(tmp_path):
    settings, database, _, dataset = _registered(tmp_path)
    profile = database.create_profile("Stroke reviewer")
    received = []

    class InspectSeeds(DeterministicEngine):
        def propagate(self, **kwargs):
            received.extend(kwargs["prompts"])
            return super().propagate(**kwargs)

    points = [{"x": index / 199, "y": 0.2} for index in range(200)]
    job, result = make_preview(
        settings,
        database,
        profile,
        black_spec(
            dataset,
            corrections=[
                {
                    "object_id": 1,
                    "frame_index": 1,
                    "target": "protect",
                    "operation": "add",
                    "radius": 0.1,
                    "points": points,
                }
            ],
        ),
        engine=InspectSeeds(),
    )
    assert len(received[-1]["points"]) == 64
    assert received[-1]["points"][0]["x"] == 0
    assert received[-1]["points"][-1]["x"] == 1
    mask = np.asarray(
        Image.open(
            settings.jobs_root / "segmentation" / job["id"] / "protect/000001.png"
        )
    )
    assert (mask[3, :] == 255).all()
    assert (
        mask[4, 15] == 255
    )  # Full radius, including the end beyond the first64 points.
    assert "exact-keyframe" in result["model_provenance"]["correction_policy"]


def test_real_api_accepts_editor_black_empty_background_without_checkpoint(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from datasetui.queueing import RecordingDispatcher
    from datasetui.segmentation_api import create_segmentation_router

    settings, database, _, dataset = _registered(tmp_path)
    profile = database.create_profile("Black editor")
    app = FastAPI()
    app.include_router(
        create_segmentation_router(database, RecordingDispatcher(), settings)
    )
    client = TestClient(app)
    request = {
        "profile_id": profile["id"],
        "idempotency_key": "manual-editor-payload",
        "spec": black_spec(
            dataset,
            background_base64="",
            prompts=[],
            manual_regions=[{"box": [0, 0, 0.5, 0.5]}],
        ),
    }
    response = client.post("/api/v1/segmentation/previews", json=request)
    assert response.status_code == 202, response.text
    assert (
        database.get_job(response.json()["id"])["payload"]["spec"]["background_base64"]
        is None
    )
    assert response.json()["queue_name"] == "io"
    request["spec"]["render_mode"] = "image"
    assert client.post("/api/v1/segmentation/previews", json=request).status_code == 422


def test_object_erase_preserves_other_overlapping_protected_track(tmp_path):
    from datasetui.segmentation_masks import apply_object_corrections, apply_selection

    spec = SegmentationSpec.model_validate(
        {
            "dataset_id": str(uuid4()),
            "fingerprint": "a" * 64,
            "episode_index": 0,
            "video_key": "top",
            "prompts": [
                {
                    "object_id": number,
                    "frame_index": 0,
                    "target": "protect",
                    "points": [{"x": 0, "y": 0}],
                }
                for number in (1, 2)
            ],
            "corrections": [
                {
                    "object_id": 1,
                    "frame_index": 0,
                    "target": "protect",
                    "operation": "erase",
                    "radius": 0.25,
                    "points": [{"x": 0, "y": 0}],
                }
            ],
        }
    )
    candidates = []
    for number in (1, 2):
        candidate_id = f"{number}-1"
        directory = tmp_path / "instances" / candidate_id
        directory.mkdir(parents=True)
        Image.fromarray(np.asarray([[255, 0], [0, 0]], dtype=np.uint8)).save(
            directory / "000000.png"
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "object_id": number,
                "target": "protect",
                "area_pixels": 1,
                "frame_index": 0,
                "requires_selection": False,
                "artifact_name": f"candidate-{candidate_id}.png",
            }
        )
    provenance = {"candidates": candidates}
    apply_object_corrections(tmp_path, spec, provenance, 1, 2, 2)
    apply_selection(tmp_path, spec, provenance, 1, 2, 2)
    assert np.asarray(Image.open(tmp_path / "instances/1-1/000000.png"))[0, 0] == 0
    assert np.asarray(Image.open(tmp_path / "protect/000000.png"))[0, 0] == 255


@pytest.mark.parametrize("current_policy", [False, True])
def test_new_mode_mask_reuse_requires_mixed_hint_policy(tmp_path, current_policy):
    from datasetui.sam3_engine import MIXED_HINT_POLICY

    settings, database, _, dataset = _registered(tmp_path)
    profile = database.create_profile("Mixed hint cache")
    spec = black_spec(
        dataset,
        mode="object_selection",
        prompts=[
            {"object_id": 1, "frame_index": 0, "target": "protect", "text": "plug"}
        ],
    )

    class VersionedEngine(CandidateEngine):
        def propagate(self, **kwargs):
            result = super().propagate(**kwargs)
            if current_policy:
                result["hint_policy"] = MIXED_HINT_POLICY
            return result

    first, _ = make_preview(settings, database, profile, spec, engine=VersionedEngine())
    selected = {
        **spec,
        "source_preview_id": first["id"],
        "selected_candidate_ids": ["1-1"],
    }
    if current_policy:
        _, result = make_preview(
            settings, database, profile, selected, engine=NeverInfer()
        )
        assert not result["selection_required"]
    else:
        with pytest.raises(SegmentationError, match="혼합 힌트 처리 방식"):
            make_preview(settings, database, profile, selected, engine=NeverInfer())
