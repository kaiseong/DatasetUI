import numpy as np
import pytest

from datasetui.segmentation.selection import retained_mask, brush_hints
from datasetui.segmentation.contract import Correction


@pytest.mark.parametrize(
    "has_keep,expected",
    [(True, [False, True, False, False]), (False, [True, True, False, False])],
)
def test_remove_wins_and_remove_only_keeps_rest(has_keep, expected):
    keep = np.array([False, True, True, False])
    remove = np.array([False, False, True, True])
    assert (
        retained_mask("object_selection", keep, remove, has_keep=has_keep).tolist()
        == expected
    )


def test_legacy_composition_is_unchanged():
    keep = np.array([False, True, True, False])
    remove = np.array([False, False, True, True])
    assert retained_mask("protect_foreground", keep, remove).tolist() == keep.tolist()
    assert retained_mask("replace_background", keep, remove).tolist() == [
        True,
        True,
        True,
        False,
    ]


def test_grow_mask_is_a_disk_dilation():
    from datasetui.segmentation.selection import grow_mask

    mask = np.zeros((40, 50), dtype=bool)
    mask[3, 4] = mask[20, 25] = mask[39, 49] = True
    mask[10:13, 30:45] = True
    ys, xs = np.mgrid[:40, :50]
    for radius in (1, 3, 7):
        expected = np.zeros_like(mask)
        for y, x in zip(*np.nonzero(mask)):
            expected |= (ys - y) ** 2 + (xs - x) ** 2 <= radius * radius
        assert np.array_equal(grow_mask(mask, radius), expected)
    assert grow_mask(mask, 0) is mask


def test_edge_margin_only_claims_unowned_pixels_and_removal_wins():
    keep = np.zeros((1, 12), dtype=bool)
    remove = np.zeros((1, 12), dtype=bool)
    keep[0, 2:5] = True
    remove[0, 5:7] = True
    remove[0, 10] = True
    grown = retained_mask("object_selection", keep, remove, margin_px=2)
    # The kept object grows left only: its right side touches the removal,
    # whose own margin (7-8) wins over the keep margin; the removal never
    # eats into the kept object itself.
    assert np.nonzero(grown[0])[0].tolist() == [0, 1, 2, 3, 4]
    replaced = retained_mask("replace_background", keep, remove, margin_px=1)
    assert np.nonzero(~replaced[0])[0].tolist() == [5, 6, 7, 9, 10, 11]
    assert np.array_equal(
        retained_mask("protect_foreground", keep, remove, margin_px=1)[0],
        np.isin(np.arange(12), [1, 2, 3, 4]),
    )


def test_edge_margin_is_render_only_and_keeps_old_recipe_hashes():
    from uuid import uuid4
    from datasetui.segmentation.contract import SegmentationSpec
    from datasetui.segmentation.selection import mask_inputs

    base = {
        "dataset_id": str(uuid4()),
        "fingerprint": "a" * 64,
        "episode_index": 0,
        "video_key": "cam",
        "mode": "object_selection",
        "prompts": [{"object_id": 1, "target": "protect", "frame_index": 0, "text": "arm"}],
    }
    plain = SegmentationSpec.model_validate(base).model_dump(mode="json")
    assert "edge_margin_px" not in plain
    assert "edge_margin_px" not in SegmentationSpec.model_validate(
        {**base, "edge_margin_px": 0}
    ).model_dump(mode="json")
    wide = SegmentationSpec.model_validate({**base, "edge_margin_px": 6})
    assert wide.model_dump(mode="json")["edge_margin_px"] == 6
    assert mask_inputs(wide.model_dump(mode="json")) == mask_inputs(plain)
    for bad in (-1, 65, 2.5):
        with pytest.raises(ValueError):
            SegmentationSpec.model_validate({**base, "edge_margin_px": bad})


def test_brush_hints_are_bounded_radius_aware_and_have_correct_polarity():
    stroke = Correction.model_validate(
        {
            "object_id": 1,
            "frame_index": 0,
            "target": "replace",
            "operation": "erase",
            "radius": 0.1,
            "points": [{"x": 0.5, "y": 0.5}],
        }
    )
    hints = brush_hints(stroke, 640, 480)
    assert 1 < len(hints) <= 64
    assert all(point["label"] == 0 for point in hints)
    assert max(p["x"] for p in hints) - min(p["x"] for p in hints) > 0.1
    assert max(p["y"] for p in hints) - min(p["y"] for p in hints) > 0.1
    long = stroke.model_copy(update={"points": stroke.points * 1000})
    assert len(brush_hints(long, 640, 480)) <= 64
    assert brush_hints(stroke, 640, 480) == hints


@pytest.mark.parametrize("margin", [0, 2])
@pytest.mark.parametrize("keep,remove", [(True, False), (False, True), (True, True)])
def test_sample_video_and_export_use_identical_selection(tmp_path, keep, remove, margin):
    import shutil
    from PIL import Image
    from datasetui.segmentation.preview import create_preview, read_manifest
    from datasetui.segmentation.media import iter_video_arrays
    from datasetui.segmentation.sample import create_sample
    from datasetui.segmentation.catalog import selected_scope
    from datasetui.segmentation.export import write_segmented_videos
    from test_segmentation import _registered

    settings, database, root, dataset = _registered(tmp_path)
    profile = database.create_profile("selection test")
    selection = selected_scope(
        database, settings, dataset["id"], 0, "observation.images.top"
    )
    prompts = []
    if keep:
        prompts.append(
            {
                "object_id": 1,
                "target": "protect",
                "frame_index": 0,
                "points": [{"x": 0.2, "y": 0.5}],
            }
        )
    if remove:
        prompts.append(
            {
                "object_id": 2,
                "target": "replace",
                "frame_index": 0,
                "points": [{"x": 0.6, "y": 0.5}],
            }
        )
    spec = {
        "dataset_id": dataset["id"],
        "episode_index": 0,
        "video_key": "observation.images.top",
        "mode": "object_selection",
        "frame_token": selection["frame_token"],
        "prompts": prompts,
        "edge_margin_px": margin,
    }

    class Engine:
        def propagate(
            self, *, video_path, prompts, frame_count, output_dir, check_lease
        ):
            candidates = []
            for target in ["protect", "replace"]:
                (output_dir / target).mkdir(exist_ok=True)
            for obj, target, active in [(1, "protect", keep), (2, "replace", remove)]:
                mask = np.zeros((16, 16), dtype=np.uint8)
                if active:
                    mask[:, : 8 if obj == 1 else 12] = 255
                if obj == 2:
                    mask[:, :4] = 0
                if active:
                    cid = f"{obj}-1"
                    directory = output_dir / "instances" / cid
                    directory.mkdir(parents=True)
                    for i in range(frame_count):
                        Image.fromarray(mask).save(directory / f"{i:06d}.png")
                    Image.fromarray(mask).save(output_dir / f"candidate-{cid}.png")
                    candidates.append(
                        {
                            "candidate_id": cid,
                            "object_id": obj,
                            "target": target,
                            "area_pixels": int((mask > 0).sum()),
                            "frame_index": 0,
                            "artifact_name": f"candidate-{cid}.png",
                        }
                    )
                for i in range(frame_count):
                    Image.fromarray(mask).save(output_dir / target / f"{i:06d}.png")
            return {"engine": "fixture", "candidates": candidates}

    results = {}
    for kind, run, request in [
        ("segmentation.sample", create_sample, {**spec, "frame_index": 0}),
        ("segmentation.preview", create_preview, spec),
    ]:
        job, _ = database.create_job(
            kind=kind,
            queue_name="gpu",
            profile_id=profile["id"],
            payload={"spec": request},
            idempotency_key=kind,
        )
        database.claim_job(job["id"], worker_id="test", lease_seconds=120)
        result = run(
            database,
            settings,
            job_id=job["id"],
            worker_id="test",
            spec=request,
            engine=Engine(),
        )
        database.succeed_job(job["id"], result, worker_id="test")
        results[kind] = job["id"]
    sample = (
        settings.jobs_root / "segmentation-samples" / results["segmentation.sample"]
    )
    preview = settings.jobs_root / "segmentation" / results["segmentation.preview"]
    k = np.zeros((16, 16), dtype=bool)
    k[:, :8] = keep
    r = np.zeros((16, 16), dtype=bool)
    if remove:
        r[:, 4:12] = True
    expected = retained_mask("object_selection", k, r, has_keep=keep, margin_px=margin)
    if margin and keep and not remove:
        assert expected[:, :10].all() and not expected[:, 10:].any()
    assert np.array_equal(np.asarray(Image.open(sample / "mask.png")) > 0, expected)
    original = np.asarray(Image.open(sample / "original.png"))
    assert np.array_equal(
        np.asarray(Image.open(sample / "composite.png")),
        np.where(expected[:, :, None], original, 0),
    )
    exported_root = tmp_path / "exported"
    shutil.copytree(root, exported_root)
    exported = (
        exported_root / "videos/chunk-000/observation.images.top/episode_000000.mp4"
    )
    before = list(iter_video_arrays(exported))
    write_segmented_videos(
        exported_root, [(preview, read_manifest(preview))], lambda: None
    )
    after = list(iter_video_arrays(exported))
    assert len(after) == 4
    # Source-codec re-encode: visually identical, not bit-exact.
    target = np.where(expected[:, :, None], before[0], 0).astype(int)
    assert np.abs(after[0].astype(int) - target).mean() < 8
    rendered = list(iter_video_arrays(preview / "composite.mp4"))
    assert len(rendered) == 4
    assert np.abs(rendered[0].astype(int) - after[0].astype(int)).mean() < 8


def test_new_mode_rejects_manual_and_missing_detections():
    from uuid import uuid4
    from datasetui.segmentation.contract import SegmentationSpec
    from datasetui.segmentation.selection import validate_detections

    base = {
        "dataset_id": str(uuid4()),
        "fingerprint": "a" * 64,
        "episode_index": 0,
        "video_key": "camera",
        "mode": "object_selection",
        "prompts": [
            {"object_id": 1, "frame_index": 0, "target": "replace", "text": "plug"}
        ],
    }
    parsed = SegmentationSpec.model_validate(base)
    with pytest.raises(ValueError, match="수동 보호"):
        SegmentationSpec.model_validate(
            {**base, "manual_regions": [{"box": [0, 0, 0.5, 0.5]}]}
        )
    with pytest.raises(ValueError, match="찾지 못했습니다"):
        validate_detections(parsed, {"candidates": []})


def test_new_brush_does_not_rasterize_masks(tmp_path):
    from PIL import Image
    from uuid import uuid4
    from datasetui.segmentation.contract import SegmentationSpec
    from datasetui.segmentation.selection import apply_legacy_corrections
    from datasetui.segmentation.selection import apply_object_corrections

    parsed = SegmentationSpec.model_validate(
        {
            "dataset_id": str(uuid4()),
            "fingerprint": "a" * 64,
            "episode_index": 0,
            "video_key": "camera",
            "mode": "object_selection",
            "corrections": [
                {
                    "object_id": 1,
                    "target": "replace",
                    "frame_index": 0,
                    "radius": 0.2,
                    "operation": "add",
                    "points": [{"x": 0.5, "y": 0.5}],
                }
            ],
        }
    )
    assert len(parsed.prompts) == 1
    for target in ("protect", "replace"):
        (tmp_path / target).mkdir()
        Image.new("L", (16, 16)).save(tmp_path / target / "000000.png")
    apply_object_corrections(tmp_path, parsed, {"candidates": []}, 1, 16, 16)
    apply_legacy_corrections(tmp_path, parsed, 1, 16, 16)
    assert not np.asarray(Image.open(tmp_path / "replace/000000.png")).any()


def test_negative_only_object_is_rejected_before_inference():
    from uuid import uuid4
    from datasetui.segmentation.contract import SegmentationSpec
    from datasetui.segmentation.workflow_contract import CameraTemplate
    from datasetui.segmentation.sample import SampleSpec, frame_guidance

    prompts = [
        {
            "object_id": 1,
            "frame_index": 0,
            "target": "protect",
            "points": [{"x": 0.5, "y": 0.5, "label": 1}],
        },
        {
            "object_id": 2,
            "frame_index": 0,
            "target": "protect",
            "points": [{"x": 0.7, "y": 0.8, "label": 0}],
        },
    ]
    base = dict(
        dataset_id=str(uuid4()),
        fingerprint="a" * 64,
        episode_index=0,
        video_key="camera",
        mode="object_selection",
        prompts=prompts,
    )
    with pytest.raises(ValueError, match="객체 2.*제외"):
        SegmentationSpec.model_validate(base)
    with pytest.raises(ValueError, match="객체 2.*제외"):
        CameraTemplate.model_validate(
            dict(
                video_key="camera",
                camera_mode="fixed",
                mode="object_selection",
                prompts=prompts,
            )
        )
    # Negative keyframes for an already identified object remain valid.
    prompts[1].update(object_id=1, frame_index=3)
    SegmentationSpec.model_validate(base)
    sample = SampleSpec.model_validate(
        {**base, "fingerprint": None, "frame_token": str(uuid4()), "frame_index": 3}
    )
    with pytest.raises(ValueError, match="객체 1.*제외"):
        frame_guidance(sample)


def test_detection_error_is_safe_and_user_visible():
    from types import SimpleNamespace
    from datasetui.segmentation.selection import validate_detections
    from datasetui.tasks import _public_failure

    parsed = SimpleNamespace(
        mode="object_selection",
        prompts=[SimpleNamespace(object_id=2, target="protect")],
    )
    with pytest.raises(ValueError) as error:
        validate_detections(parsed, {"candidates": []})
    code, message = _public_failure(error.value)
    assert code != "job_failed"
    assert "객체 2" in message and "찾지 못했습니다" in message
