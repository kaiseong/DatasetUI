"""Objects that are off-camera on the prompt frame, or leave and re-enter."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from datasetui.sam3_engine import _propagate_masks, _resolve_late_tracks
from datasetui.segmentation_contract import RegionPrompt

H, W = 8, 8


def _outputs(objects: dict[int, np.ndarray], scores: dict[int, float] | None = None):
    ids = sorted(objects)
    masks = (
        np.stack([objects[key] for key in ids])
        if ids
        else np.zeros((0, H, W), dtype=bool)
    )
    return {
        "out_obj_ids": np.asarray(ids, dtype=np.int64),
        "out_binary_masks": masks,
        "out_probs": np.asarray([(scores or {}).get(key, 0.9) for key in ids]),
    }


def _block(column: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    mask[2:5, column : column + 2] = True
    return mask


class LateConceptPredictor:
    """Text prompt finds nothing on frame 0; a track appears on frames 3-5."""

    def __init__(self, timeline: dict[int, dict[int, np.ndarray]]):
        self.timeline = timeline

    def handle_request(self, request):
        if request["type"] == "start_session":
            return {"session_id": "s"}
        if request["type"] == "add_prompt":
            return {"frame_index": request["frame_index"], "outputs": _outputs({})}
        return {}

    def handle_stream_request(self, request):
        for frame in range(len(self.timeline)):
            yield {"frame_index": frame, "outputs": _outputs(self.timeline[frame])}


def _directories(tmp_path):
    result = {}
    for target in ("replace", "protect"):
        (tmp_path / target).mkdir()
        result[target] = tmp_path / target
    return result


def test_text_object_absent_on_prompt_frame_is_found_later(tmp_path):
    timeline = {frame: {} for frame in range(6)}
    for frame in (3, 4, 5):
        timeline[frame] = {7: _block(2)}
    provenance = _propagate_masks(
        predictor=LateConceptPredictor(timeline),
        video_path=tmp_path / "unused.mp4",
        prompts=[
            RegionPrompt(
                object_id=1,
                frame_index=0,
                target="protect",
                text="robot gripper",
                confidence_threshold=0.5,
            )
        ],
        frame_count=6,
        target_directories=_directories(tmp_path),
        checkpoint_sha256="a" * 64,
        check_lease=lambda: None,
        mixed_spatial=True,
    )
    [candidate] = provenance["candidates"]
    assert candidate["candidate_id"] == "1-7"
    assert candidate["late_track"] is True
    assert candidate["first_visible_frame"] == 3
    assert candidate["visible_ranges"] == [[3, 5]]
    assert candidate["requires_selection"] is True
    with Image.open(tmp_path / "protect" / "000004.png") as image:
        assert np.asarray(image).any()
    with Image.open(tmp_path / "protect" / "000001.png") as image:
        assert not np.asarray(image).any()


def _areas(tracks: dict[int, list[int]]):
    return {
        (sam_id, frame): (12 if frame in frames else 0)
        for sam_id, frames in tracks.items()
        for frame in range(10)
    }


def _candidate(tmp_path, sam_id):
    candidate_id = f"1-{sam_id}"
    (tmp_path / "instances" / candidate_id).mkdir(parents=True)
    (tmp_path / f"candidate-{candidate_id}.png").write_bytes(b"x")
    return candidate_id, {
        "candidate_id": candidate_id,
        "object_id": 1,
        "sam_object_id": sam_id,
        "artifact_name": f"candidate-{candidate_id}.png",
        "requires_selection": False,
    }


def test_selected_object_reentry_is_accepted_only_while_it_is_absent(tmp_path):
    candidates = dict(
        _candidate(tmp_path, sam_id) for sam_id in (3, 8, 9)
    )
    # 3 = selected object (frames 0-3), 8 = same object back after leaving
    # (frames 6-9), 9 = a second instance that coexists with it (frames 2-4).
    _resolve_late_tracks(
        candidates,
        group_id=1,
        late_ids={8, 9},
        initial_ids={3},
        capacity=1,
        areas=_areas({3: [0, 1, 2, 3], 8: [6, 7, 8, 9], 9: [2, 3, 4]}),
        instances=tmp_path / "instances",
        output_root=tmp_path,
    )
    assert set(candidates) == {"1-3", "1-8"}
    assert candidates["1-8"]["reentry"] is True
    assert candidates["1-8"]["first_visible_frame"] == 6
    assert not (tmp_path / "instances" / "1-9").exists()
    assert not (tmp_path / "candidate-1-9.png").exists()


def test_unselected_text_object_keeps_late_tracks_as_choices(tmp_path):
    candidates = dict(_candidate(tmp_path, sam_id) for sam_id in (4, 5))
    _resolve_late_tracks(
        candidates,
        group_id=1,
        late_ids={4, 5},
        initial_ids=set(),
        capacity=None,
        areas=_areas({4: [1, 2], 5: [2, 3]}),
        instances=tmp_path / "instances",
        output_root=tmp_path,
    )
    assert set(candidates) == {"1-4", "1-5"}
    assert all(item["late_track"] for item in candidates.values())
    assert not any(item.get("reentry") for item in candidates.values())


def test_tracking_reports_frame_progress(tmp_path):
    events = []
    timeline = {frame: {7: _block(2)} for frame in range(4)}
    _propagate_masks(
        predictor=LateConceptPredictor(timeline),
        video_path=tmp_path / "unused.mp4",
        prompts=[RegionPrompt(object_id=1, frame_index=0, target="protect", text="gripper", confidence_threshold=0.5)],
        frame_count=4,
        target_directories=_directories(tmp_path),
        checkpoint_sha256="a" * 64,
        check_lease=lambda: None,
        mixed_spatial=True,
        on_progress=events.append,
    )
    assert [event["completed"] for event in events] == [1, 2, 3, 4]
    assert all(event["stage"] == "segment" and event["total"] >= event["completed"] for event in events)
    assert events[-1]["current_item"] == "SAM 추적 · 객체 1/1"


def test_weighted_progress_is_whole_job_and_never_moves_back():
    from datasetui.job_progress import WeightedProgress

    seen = []
    progress = WeightedProgress(seen.append, {"read": 10, "segment": 80, "write": 10})
    progress({"stage": "read", "completed": 10, "total": 10, "unit": "frames"})
    progress({"stage": "segment", "completed": 40, "total": 100, "unit": "frames"})
    progress({"stage": "segment", "completed": 150, "total": 100, "unit": "frames"})
    progress({"stage": "write", "completed": 0, "total": 10, "unit": "frames"})
    progress({"stage": "complete", "completed": 1, "total": 1, "unit": "items"})
    overall = [event["overall"] for event in seen]
    assert overall[0] == pytest.approx(0.1) and overall[1] == pytest.approx(0.42)
    assert overall == sorted(overall) and overall[-1] == 1.0
