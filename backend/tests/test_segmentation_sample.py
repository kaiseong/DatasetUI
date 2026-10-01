import av
import pytest

from datasetui.segmentation_catalog import selected_scope
from datasetui.segmentation_sample import SampleSpec, create_sample, frame_guidance
from datasetui.segmentation_api import verified_preview
from test_segmentation import _registered, DeterministicEngine


def test_sample_rebases_one_frame_and_rejects_full_preview(tmp_path, monkeypatch):
    settings, database, _, dataset = _registered(tmp_path)
    scope = selected_scope(
        database, settings, dataset["id"], 0, "observation.images.top"
    )
    spec = {
        "dataset_id": dataset["id"],
        "frame_token": scope["frame_token"],
        "episode_index": 0,
        "video_key": "observation.images.top",
        "frame_index": 2,
        "prompts": [
            {"frame_index": 0, "target": "protect", "text": "plug", "object_id": 1},
            {
                "frame_index": 2,
                "target": "protect",
                "object_id": 1,
                "points": [{"x": 0.5, "y": 0.5, "label": 1}],
            },
        ],
    }
    rebased = frame_guidance(SampleSpec.model_validate(spec))
    assert len(rebased.prompts) == 1
    assert rebased.prompts[0].frame_index == 0
    assert rebased.prompts[0].text == "plug"
    assert len(rebased.prompts[0].points) == 1

    class OneFrameEngine(DeterministicEngine):
        def propagate(self, **kwargs):
            assert kwargs["frame_count"] == 1
            with av.open(str(kwargs["video_path"])) as video:
                assert len(list(video.decode(video=0))) == 1
            return super().propagate(**kwargs)

    monkeypatch.setattr(database, "assert_job_lease", lambda *a, **k: None)
    monkeypatch.setattr(database, "update_job_progress", lambda *a, **k: None)
    import uuid

    job_id = str(uuid.uuid4())
    result = create_sample(
        database,
        settings,
        job_id=job_id,
        worker_id="test",
        spec=spec,
        engine=OneFrameEngine(),
    )
    assert result["frame_count"] == 1
    assert result["source_frame_index"] == 2
    assert set(result["artifacts"]) == {"original.png", "mask.png", "composite.png"}
    monkeypatch.setattr(
        database,
        "get_job",
        lambda _: {"kind": "segmentation.sample", "status": "succeeded"},
    )
    with pytest.raises(ValueError, match="미리보기"):
        verified_preview(database, settings, job_id)


def test_sample_never_reuses_other_frame_spatial_hints(tmp_path):
    settings, database, _, dataset = _registered(tmp_path)
    scope = selected_scope(
        database, settings, dataset["id"], 0, "observation.images.top"
    )
    parsed = SampleSpec.model_validate(
        {
            "dataset_id": dataset["id"],
            "frame_token": scope["frame_token"],
            "episode_index": 0,
            "video_key": "observation.images.top",
            "frame_index": 2,
            "prompts": [
                {
                    "frame_index": 0,
                    "target": "protect",
                    "text": "plug",
                    "points": [{"x": 0.5, "y": 0.5, "label": 1}],
                }
            ],
        }
    )
    assert not frame_guidance(parsed).prompts[0].points


def test_selected_candidate_on_another_frame_is_tracked_through_a_short_clip(tmp_path):
    from uuid import uuid4

    import numpy as np
    from PIL import Image

    from datasetui.segmentation_sample import _keep_single_frame, tracking_frames

    settings, database, _, dataset = _registered(tmp_path)
    scope = selected_scope(
        database, settings, dataset["id"], 0, "observation.images.top"
    )
    parsed = SampleSpec.model_validate(
        {
            "dataset_id": dataset["id"],
            "frame_token": scope["frame_token"],
            "episode_index": 0,
            "video_key": "observation.images.top",
            "frame_index": 300,
            "mode": "object_selection",
            "prompts": [
                {
                    "object_id": 1,
                    "frame_index": 10,
                    "target": "protect",
                    "text": "board",
                    "confidence_threshold": 0.5,
                    "selected_candidates": [
                        {"sample_id": str(uuid4()), "candidate_id": "1-3"}
                    ],
                },
                {
                    "object_id": 1,
                    "frame_index": 300,
                    "target": "protect",
                    "points": [{"x": 0.5, "y": 0.5, "label": 1}],
                },
            ],
        }
    )
    frames = tracking_frames(parsed)
    assert frames[0] == 10 and frames[-1] == 300 and len(frames) <= 25
    index = {frame: position for position, frame in enumerate(frames)}
    guidance = parsed.model_copy()
    from datasetui.segmentation_sample import frame_guidance as rebase

    rebased = rebase(guidance, index)
    by_text = {bool(p.text): p.frame_index for p in rebased.prompts}
    assert by_text == {True: 0, False: len(frames) - 1}

    staging = tmp_path / "staging"
    for directory in ("protect", "replace", "instances/1-3"):
        (staging / directory).mkdir(parents=True)
        for frame in range(3):
            mask = np.zeros((4, 4), dtype=np.uint8)
            if frame == 2 and directory != "replace":
                mask[1:3, 1:3] = 255
            Image.fromarray(mask).save(staging / directory / f"{frame:06d}.png")
    provenance = {
        "candidates": [
            {"candidate_id": "1-3", "artifact_name": "candidate-1-3.png", "area_pixels": 4}
        ]
    }
    _keep_single_frame(staging, 2, 3, provenance, 300)
    assert sorted(path.name for path in (staging / "protect").iterdir()) == ["000000.png"]
    [candidate] = provenance["candidates"]
    assert candidate["area_pixels"] == 4 and candidate["frame_index"] == 300
    assert (staging / "candidate-1-3.png").is_file()
    assert tracking_frames(parsed.model_copy(update={"frame_index": 10})) == [10]
