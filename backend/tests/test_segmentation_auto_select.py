"""A text object with one plausible physical match is selected automatically."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

from datasetui.segmentation.selection import apply_selection, selection_review_signals

W = H = 4


def _write(root, candidate_id, frames, count=4):
    directory = root / "instances" / candidate_id
    directory.mkdir(parents=True)
    for index in range(count):
        mask = np.zeros((H, W), dtype=np.uint8)
        if index in frames:
            mask[1:3, 1:3] = 255
        Image.fromarray(mask).save(directory / f"{index:06d}.png")


def _candidate(candidate_id, ranges, **extra):
    return {
        "candidate_id": candidate_id,
        "object_id": 1,
        "target": "protect",
        "area_pixels": 4,
        "frame_index": ranges[0][0],
        "requires_selection": True,
        "visible_ranges": ranges,
        **extra,
    }


def _parsed(selected=()):
    prompt = SimpleNamespace(text="board", selected_candidates=[])
    return SimpleNamespace(prompts=[prompt], selected_candidate_ids=list(selected))


def _protect(root, index):
    with Image.open(root / "protect" / f"{index:06d}.png") as image:
        return np.asarray(image) > 0


def test_single_text_candidate_is_auto_selected(tmp_path):
    _write(tmp_path, "1-3", {0, 1, 2, 3})
    provenance = {"candidates": [_candidate("1-3", [[0, 3]])]}
    required = apply_selection(tmp_path, _parsed(), provenance, 4, W, H)
    assert required is False
    assert provenance["candidates"][0]["auto_selected"] is True
    assert _protect(tmp_path, 2).any()
    assert selection_review_signals(provenance) == [
        {"frame_index": 0, "reason": "auto_selected", "candidate_id": "1-3"}
    ]


def test_time_disjoint_tracks_are_one_object_reentering(tmp_path):
    _write(tmp_path, "1-3", {0, 1})
    _write(tmp_path, "1-8", {3})
    provenance = {
        "candidates": [
            _candidate("1-3", [[0, 1]]),
            _candidate("1-8", [[3, 3]], late_track=True, first_visible_frame=3),
        ]
    }
    assert apply_selection(tmp_path, _parsed(), provenance, 4, W, H) is False
    assert _protect(tmp_path, 0).any() and _protect(tmp_path, 3).any()
    assert {"frame_index": 3, "reason": "reentry", "candidate_id": "1-8"} in (
        selection_review_signals(provenance)
    )


def test_coexisting_candidates_still_require_a_choice(tmp_path):
    _write(tmp_path, "1-3", {0, 1, 2})
    _write(tmp_path, "1-4", {1, 2, 3})
    provenance = {
        "candidates": [_candidate("1-3", [[0, 2]]), _candidate("1-4", [[1, 3]])]
    }
    assert apply_selection(tmp_path, _parsed(), provenance, 4, W, H) is True
    assert not any(item.get("auto_selected") for item in provenance["candidates"])
    assert apply_selection(tmp_path, _parsed(["1-4"]), provenance, 4, W, H) is False
    assert not _protect(tmp_path, 0).any() and _protect(tmp_path, 3).any()


def test_object_coverage_reports_frames_where_an_object_is_missing():
    from datasetui.segmentation.selection import coverage_signals, object_coverage

    provenance = {
        "candidates": [
            # Board: chosen, visible 0-3 and 7-9 (re-entry) -> missing 4-6.
            _candidate("4-1", [[0, 3]]) | {"object_id": 4, "requires_selection": False},
            _candidate("4-9", [[7, 9]]) | {"object_id": 4, "requires_selection": False},
            # Gripper: auto-selected, always visible.
            _candidate("2-1", [[0, 9]]) | {"object_id": 2, "auto_selected": True},
            # Unchosen proposal never counts.
            _candidate("2-5", [[0, 1]]) | {"object_id": 2},
        ]
    }
    coverage = object_coverage(provenance, [], 10)
    assert [(c["object_id"], c["visible_frames"], c["missing_ranges"]) for c in coverage] == [
        (2, 10, []),
        (4, 7, [[4, 6]]),
    ]
    assert coverage_signals(coverage) == [
        {"frame_index": 4, "reason": "object_gap", "object_id": 4, "missing_frames": 3}
    ]
    legacy = {"candidates": [{**_candidate("1-1", [[0, 0]]), "requires_selection": False}]}
    del legacy["candidates"][0]["visible_ranges"]
    assert object_coverage(legacy, [], 10) == []
