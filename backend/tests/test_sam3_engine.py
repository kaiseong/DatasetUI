from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import datasetui.segmentation.engine as sam3_engine
from datasetui.segmentation.engine import Sam3Engine, Sam3InferenceError, Sam3UnavailableError


def _settings(checkpoint: Path, *, expected_sha: str | None = None):
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return SimpleNamespace(
        sam3_checkpoint=checkpoint,
        sam3_checkpoint_sha256=expected_sha or sha,
        segmentation_max_frames=8,
    )


class FakePredictor:
    def __init__(self, outputs_by_session: list[list[dict]]):
        self.outputs_by_session = outputs_by_session
        self.sessions: list[str] = []
        self.start_requests: list[dict] = []
        self.closed: list[str] = []
        self.prompts: list[dict] = []
        self.stream_requests: list[dict] = []
        self.shutdown_called = False

    def handle_request(self, request: dict):
        if request["type"] == "start_session":
            self.start_requests.append(request)
            session_id = f"session-{len(self.sessions)}"
            self.sessions.append(session_id)
            return {"session_id": session_id}
        if request["type"] == "add_prompt":
            self.prompts.append(request)
            return {
                "frame_index": request["frame_index"],
                "outputs": {"out_obj_ids": np.asarray([1])},
            }
        if request["type"] == "close_session":
            self.closed.append(request["session_id"])
            return {"is_success": True}
        raise AssertionError(request)

    def handle_stream_request(self, request: dict):
        self.stream_requests.append(request)
        index = int(request["session_id"].split("-")[-1])
        yield from self.outputs_by_session[index]

    def shutdown(self):
        self.shutdown_called = True


def _frame(
    frame_index: int,
    mask: np.ndarray,
    *,
    object_id: int = 1,
    discovered_mask: np.ndarray | None = None,
) -> dict:
    object_ids = [object_id]
    masks = [mask]
    if discovered_mask is not None:
        object_ids.append(999)
        masks.append(discovered_mask)
    return {
        "frame_index": frame_index,
        "outputs": {
            "out_obj_ids": np.asarray(object_ids),
            "out_binary_masks": np.stack(masks),
        },
    }


def test_propagate_unions_independent_prompt_sessions_and_writes_all_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "sam3.1.pt"
    checkpoint.write_bytes(b"licensed checkpoint")
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"predictor-owned fake video")
    left = np.asarray([[1, 0], [0, 0]], dtype=bool)
    right = np.asarray([[0, 1], [0, 0]], dtype=bool)
    empty = np.zeros((2, 2), dtype=bool)
    predictor = FakePredictor(
        [
            [
                _frame(0, left),
                _frame(1, empty, discovered_mask=np.ones_like(empty)),
                _frame(2, empty),
            ],
            [
                _frame(0, right, discovered_mask=np.ones_like(right)),
                _frame(1, right),
                _frame(2, empty),
            ],
        ]
    )
    monkeypatch.setattr(sam3_engine, "_build_predictor", lambda checkpoint: predictor)
    lease_checks: list[None] = []

    result = Sam3Engine(_settings(checkpoint)).propagate(
        video_path=video,
        prompts=[
            {
                "frame_index": 0,
                "target": "replace",
                "text": "table",
                "box": None,
            },
            {
                "frame_index": 1,
                "target": "replace",
                "text": "",
                "points": [
                    {"x": 0.75, "y": 0.25, "label": 1},
                    {"x": 0.25, "y": 0.75, "label": 0},
                ],
                "box": [0.5, 0.0, 0.5, 0.5],
            },
        ],
        frame_count=3,
        output_dir=tmp_path / "masks",
        check_lease=lambda: lease_checks.append(None),
    )

    assert result["engine"] == "sam3.1-multiplex"
    assert result["upstream_commit"] == sam3_engine.SAM3_UPSTREAM_COMMIT
    assert result["dimensions"] == {"width": 2, "height": 2}
    assert predictor.closed == predictor.sessions == ["session-0", "session-1"]
    assert [request["start_frame_index"] for request in predictor.stream_requests] == [
        0,
        1,
        1,
    ]
    assert all(
        request["offload_video_to_cpu"] is True for request in predictor.start_requests
    )
    assert predictor.shutdown_called is True
    assert predictor.prompts[2]["point_labels"] == [2, 3, 1, 0]
    assert predictor.prompts[1]["bounding_boxes"] == [[0.5, 0.0, 0.5, 0.5]]
    assert predictor.prompts[2]["bounding_boxes"] is None
    assert predictor.prompts[2]["text"] is None
    assert len(lease_checks) >= 6

    replace = np.asarray(Image.open(tmp_path / "masks/replace/000000.png"))
    protect = np.asarray(Image.open(tmp_path / "masks/protect/000000.png"))
    # Text sessions accept a newly discovered matching object on a later frame,
    # while the point-only session remains fenced to its prompted object ID.
    assert replace.tolist() == [[255, 255], [0, 0]]
    assert np.asarray(Image.open(tmp_path / "masks/replace/000001.png")).tolist() == [
        [255, 255],
        [255, 255],
    ]
    assert set(np.unique(replace)) <= {0, 255}
    assert np.count_nonzero(protect) == 0
    assert sorted(path.name for path in (tmp_path / "masks/replace").glob("*.png")) == [
        "000000.png",
        "000001.png",
        "000002.png",
    ]
    assert sorted(path.name for path in (tmp_path / "masks/protect").glob("*.png")) == [
        "000000.png",
        "000001.png",
        "000002.png",
    ]


def test_checkpoint_is_verified_before_predictor_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "sam3.1.pt"
    checkpoint.write_bytes(b"wrong checkpoint")
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"video")
    loaded = False

    def load(_checkpoint: Path):
        nonlocal loaded
        loaded = True

    monkeypatch.setattr(sam3_engine, "_build_predictor", load)
    with pytest.raises(Sam3UnavailableError, match="SHA-256"):
        Sam3Engine(_settings(checkpoint, expected_sha="0" * 64)).propagate(
            video_path=video,
            prompts=[{"frame_index": 0, "target": "replace", "text": "table"}],
            frame_count=1,
            output_dir=tmp_path / "masks",
            check_lease=lambda: None,
        )
    assert loaded is False


def test_same_object_keyframes_use_one_track_and_latest_frame_result(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"test checkpoint")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"test clip")
    old = np.asarray([[1, 0], [0, 0]], dtype=bool)
    corrected = np.asarray([[0, 1], [0, 0]], dtype=bool)
    predictor = FakePredictor(
        [[_frame(0, old), _frame(1, corrected), _frame(0, corrected)]]
    )
    monkeypatch.setattr(sam3_engine, "_build_predictor", lambda _: predictor)
    result = Sam3Engine(_settings(checkpoint)).propagate(
        video_path=video,
        prompts=[
            {
                "object_id": 7,
                "frame_index": 0,
                "target": "protect",
                "points": [{"x": 0.1, "y": 0.1}],
            },
            {
                "object_id": 7,
                "frame_index": 1,
                "target": "protect",
                "points": [{"x": 0.8, "y": 0.1}],
            },
        ],
        frame_count=2,
        output_dir=tmp_path / "masks",
        check_lease=lambda: None,
    )
    assert result["session_count"] == 1
    assert len(predictor.prompts) == 2 and len(predictor.stream_requests) == 1
    assert predictor.stream_requests[0]["start_frame_index"] == 1
    assert predictor.prompts[0]["session_id"] == predictor.prompts[1]["session_id"]
    assert predictor.prompts[0]["obj_id"] == predictor.prompts[1]["obj_id"]
    actual = np.asarray(Image.open(tmp_path / "masks/protect/000000.png")) > 0
    assert np.array_equal(actual, corrected)
    assert result["candidates"][0]["candidate_id"] == "7-1"


def test_text_correction_resolves_one_instance_from_current_session_geometry(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"test checkpoint")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"test clip")
    left = np.asarray([[1, 0], [0, 0]], dtype=bool)
    right = np.asarray([[0, 1], [0, 0]], dtype=bool)

    class MultiplePredictor(FakePredictor):
        def handle_request(self, request):
            response = super().handle_request(request)
            if request["type"] == "add_prompt" and request["text"]:
                response["outputs"]["out_obj_ids"] = np.asarray([1, 999])
            return response

    predictor = MultiplePredictor(
        [
            [
                _frame(0, left, discovered_mask=right),
                _frame(1, left, discovered_mask=right),
            ]
        ]
    )
    monkeypatch.setattr(sam3_engine, "_build_predictor", lambda _: predictor)
    result = Sam3Engine(_settings(checkpoint)).propagate(
        video_path=video,
        prompts=[
            {"object_id": 1, "frame_index": 0, "target": "protect", "text": "plug"},
            {
                "object_id": 1,
                "frame_index": 1,
                "target": "protect",
                "points": [{"x": 0.99, "y": 0.01}],
            },
        ],
        frame_count=2,
        output_dir=tmp_path / "masks",
        check_lease=lambda: None,
    )
    assert len(predictor.sessions) == 1
    assert predictor.prompts[1]["obj_id"] == 999
    assert len(predictor.stream_requests) == 3
    assert [candidate["candidate_id"] for candidate in result["candidates"]] == [
        "1-999"
    ]
    assert np.array_equal(
        np.asarray(Image.open(tmp_path / "masks/protect/000000.png")) > 0, right
    )


def test_missing_frame_is_rejected_and_session_is_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "sam3.1.pt"
    checkpoint.write_bytes(b"licensed checkpoint")
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"video")
    mask = np.ones((2, 2), dtype=bool)
    predictor = FakePredictor([[_frame(0, mask)]])
    monkeypatch.setattr(sam3_engine, "_build_predictor", lambda checkpoint: predictor)

    with pytest.raises(Sam3InferenceError, match="every requested frame"):
        Sam3Engine(_settings(checkpoint)).propagate(
            video_path=video,
            prompts=[{"frame_index": 0, "target": "replace", "text": "table"}],
            frame_count=2,
            output_dir=tmp_path / "masks",
            check_lease=lambda: None,
        )
    assert predictor.closed == ["session-0"]
    assert predictor.shutdown_called is True


def test_runtime_rejects_host_without_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "sam3.1.pt"
    checkpoint.write_bytes(b"licensed checkpoint")
    monkeypatch.setitem(
        __import__("sys").modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    with pytest.raises(Sam3UnavailableError, match="CUDA GPU"):
        sam3_engine._build_predictor(checkpoint)


@pytest.mark.parametrize("reject_weights", [False, True])
def test_runtime_uses_pinned_multiplex_builder_without_weight_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reject_weights: bool
) -> None:
    checkpoint = tmp_path / "sam3.1.pt"
    checkpoint.write_bytes(b"licensed checkpoint")
    captured: dict = {}
    state_loads = []
    closed = []

    def load_state(state, strict):
        state_loads.append((state, strict))
        if reject_weights:
            raise RuntimeError("Missing trained checkpoint parameters")

    expected_predictor = SimpleNamespace(
        model=SimpleNamespace(load_state_dict=load_state),
        shutdown=lambda: closed.append(True),
    )
    tracker_loads = []

    def build(**kwargs):
        captured.update(kwargs)
        model_builder.build_sam3_multiplex_video_model(
            checkpoint_path=kwargs["checkpoint_path"], load_from_HF=False
        )
        return expected_predictor

    sam3_package = ModuleType("sam3")
    sam3_package.__path__ = []
    model_builder = ModuleType("sam3.model_builder")
    model_builder.build_sam3_multiplex_video_predictor = build

    def original_tracker(**kwargs):
        tracker_loads.append(kwargs)

    model_builder.build_sam3_multiplex_video_model = original_tracker
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True),
            load=lambda *args, **kwargs: {
                "model": {"detector.weight": "verified", "tracker.weight": "verified"}
            },
        ),
    )
    monkeypatch.setitem(sys.modules, "sam3", sam3_package)
    monkeypatch.setitem(sys.modules, "sam3.model_builder", model_builder)
    monkeypatch.setenv("DATASETUI_SAM3_COMMIT", sam3_engine.SAM3_UPSTREAM_COMMIT)

    if reject_weights:
        with pytest.raises(Sam3UnavailableError, match="verified checkpoint"):
            sam3_engine._build_predictor(checkpoint)
        assert closed == [True]
    else:
        adapter = sam3_engine._build_predictor(checkpoint)
        assert adapter.predictor is expected_predictor
    assert tracker_loads == [{"checkpoint_path": None, "load_from_HF": False}]
    assert state_loads == [
        ({"detector.weight": "verified", "tracker.weight": "verified"}, True)
    ]
    assert model_builder.build_sam3_multiplex_video_model is original_tracker
    assert captured == {
        "checkpoint_path": str(checkpoint),
        "use_fa3": False,
        "use_rope_real": False,
        "compile": False,
        "warm_up": False,
        "async_loading_frames": False,
    }


def test_multiplex_adapter_does_not_forward_unsupported_state_offload():
    received = []
    exited = []

    def init_state(*, resource_path, offload_video_to_cpu, async_loading_frames):
        received.append((resource_path, offload_video_to_cpu, async_loading_frames))
        return {"frames": 4}

    predictor = SimpleNamespace(
        model=SimpleNamespace(init_state=init_state),
        async_loading_frames=False,
        _all_inference_states={},
        shutdown=lambda: None,
        bf16_context=SimpleNamespace(__exit__=lambda *args: exited.append(args)),
    )
    adapter = sam3_engine._MultiplexCompatibilityAdapter(predictor)
    result = adapter.handle_request(
        {
            "type": "start_session",
            "resource_path": "/clip.mp4",
            "offload_video_to_cpu": True,
        }
    )
    assert received == [("/clip.mp4", True, False)]
    assert predictor._all_inference_states[result["session_id"]]["state"] == {
        "frames": 4
    }
    with pytest.raises(Sam3InferenceError, match="state offload"):
        adapter.handle_request(
            {
                "type": "start_session",
                "resource_path": "/clip.mp4",
                "offload_state_to_cpu": True,
            }
        )
    adapter.shutdown()
    adapter.shutdown()
    assert len(exited) == 1


def test_box_keyframe_becomes_instance_corner_points_without_semantic_reset(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"test checkpoint")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"test clip")
    predictor = FakePredictor(
        [
            [
                _frame(0, np.ones((2, 2), dtype=bool)),
                _frame(1, np.ones((2, 2), dtype=bool)),
            ]
        ]
    )
    monkeypatch.setattr(sam3_engine, "_build_predictor", lambda _: predictor)
    Sam3Engine(_settings(checkpoint)).propagate(
        video_path=video,
        prompts=[
            {
                "object_id": 1,
                "frame_index": 0,
                "target": "protect",
                "points": [{"x": 0.5, "y": 0.5}],
            },
            {
                "object_id": 1,
                "frame_index": 1,
                "target": "protect",
                "box": [0.25, 0.25, 0.5, 0.5],
            },
        ],
        frame_count=2,
        output_dir=tmp_path / "masks",
        check_lease=lambda: None,
    )
    corrected = predictor.prompts[1]
    assert corrected["bounding_boxes"] is None and corrected["text"] is None
    assert corrected["points"] == [[0.25, 0.25], [0.75, 0.75]]
    assert corrected["point_labels"] == [2, 3]
    assert corrected["obj_id"] == predictor.prompts[0]["obj_id"]


def test_point_only_adapter_preserves_real_masks_without_semantic_cache():
    states = {
        "session": {
            "state": {"num_frames": 3, "cached_frame_outputs": {0: {7: "original"}}}
        }
    }

    def handle(request):
        state = states[request["session_id"]]["state"]
        # Exact upstream merge contract: absent frame returns {} before refine.
        return {
            index: (
                {**state["cached_frame_outputs"][index], 1: "real-model-mask"}
                if index in state["cached_frame_outputs"]
                else {}
            )
            for index in range(3)
        }

    adapter = sam3_engine._MultiplexCompatibilityAdapter(
        SimpleNamespace(_all_inference_states=states, handle_request=handle)
    )
    outputs = adapter.handle_request(
        {"type": "add_prompt", "session_id": "session", "points": [[0.4, 0.5]]}
    )
    assert all(output[1] == "real-model-mask" for output in outputs.values())
    assert outputs[0][7] == "original"
    assert states["session"]["state"]["cached_frame_outputs"][1] == {}


def test_point_track_uses_requested_id_when_initial_postprocessing_is_empty(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"test checkpoint")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"test clip")

    class DelayedPromptOutput(FakePredictor):
        def handle_request(self, request):
            result = super().handle_request(request)
            if request["type"] == "add_prompt":
                result["outputs"]["out_obj_ids"] = np.asarray([], dtype=np.int64)
            return result

    predictor = DelayedPromptOutput(
        [
            [
                _frame(0, np.ones((2, 2), dtype=bool)),
                _frame(1, np.ones((2, 2), dtype=bool)),
            ]
        ]
    )
    monkeypatch.setattr(sam3_engine, "_build_predictor", lambda _: predictor)
    result = Sam3Engine(_settings(checkpoint)).propagate(
        video_path=video,
        prompts=[
            {
                "object_id": 1,
                "frame_index": 0,
                "target": "protect",
                "points": [{"x": 0.5, "y": 0.5}],
            }
        ],
        frame_count=2,
        output_dir=tmp_path / "masks",
        check_lease=lambda: None,
    )
    assert result["candidates"][0]["candidate_id"] == "1-1"
    assert np.asarray(Image.open(tmp_path / "masks/protect/000001.png")).all()


def test_last_endpoint_runs_backwards_first_but_interior_keeps_bidirectional():
    predictor = SimpleNamespace(
        _all_inference_states={"session": {"state": {"num_frames": 3}}},
        handle_stream_request=lambda request: iter([request["propagation_direction"]]),
    )
    adapter = sam3_engine._MultiplexCompatibilityAdapter(predictor)
    request = {
        "type": "propagate_in_video",
        "session_id": "session",
        "propagation_direction": "both",
        "start_frame_index": 2,
    }
    assert list(adapter.handle_stream_request(request)) == ["backward", "forward"]
    assert list(adapter.handle_stream_request({**request, "start_frame_index": 1})) == [
        "both"
    ]


def test_mixed_spatial_hints_use_one_track_and_accumulate_brush(tmp_path, monkeypatch):
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"test checkpoint")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"test clip")
    predictor = FakePredictor(
        [
            [
                _frame(0, np.ones((2, 2), dtype=bool)),
                _frame(1, np.ones((2, 2), dtype=bool)),
            ]
        ]
    )
    monkeypatch.setattr(sam3_engine, "_build_predictor", lambda _: predictor)
    prompts = [
        {
            "object_id": 2,
            "frame_index": 0,
            "target": "replace",
            "box": [0, 0, 1, 0.2],
            "points": [
                {"x": 0.3, "y": 0.6, "label": 1},
                {"x": 0.9, "y": 0.9, "label": 0},
            ],
        },
        {
            "object_id": 2,
            "frame_index": 0,
            "target": "replace",
            "points": [{"x": 0.4, "y": 0.6, "label": 1}],
        },
        {
            "object_id": 2,
            "frame_index": 0,
            "target": "replace",
            "points": [{"x": 0.8, "y": 0.8, "label": 0}],
        },
        {
            "object_id": 2,
            "frame_index": 1,
            "target": "replace",
            "box": [0.2, 0.2, 0.3, 0.3],
        },
    ]
    result = Sam3Engine(_settings(checkpoint), mixed_spatial=True).propagate(
        video_path=video,
        prompts=prompts,
        frame_count=2,
        output_dir=tmp_path / "out",
        check_lease=lambda: None,
    )
    assert len(predictor.sessions) == 1
    assert len(predictor.prompts) == 4
    assert predictor.prompts[0]["point_labels"] == [2, 3, 1, 0]
    assert predictor.prompts[0]["points"] == [[0, 0], [1, 0.2], [0.3, 0.6], [0.9, 0.9]]
    assert all(
        p["bounding_boxes"] is None and p["text"] is None for p in predictor.prompts
    )
    assert [p.get("clear_old_points", True) for p in predictor.prompts] == [
        True,
        False,
        False,
        True,
    ]
    assert (
        len(predictor.stream_requests) == 1
    )  # no semantic candidate-resolution prerequisite
    assert result["hint_policy"] == sam3_engine.MIXED_HINT_POLICY


def test_mixed_box_only_is_instance_geometry_not_semantic_detection():
    from datasetui.segmentation.contract import RegionPrompt

    predictor = FakePredictor([])
    prompt = RegionPrompt(frame_index=0, target="protect", box=(0, 0, 1, 0.2))
    _, ids = sam3_engine._add_initial_prompt(
        predictor, "session", prompt, 1, lambda: None, mixed_spatial=True
    )
    assert ids == {1}
    assert predictor.prompts[0]["point_labels"] == [2, 3]
    assert predictor.prompts[0]["bounding_boxes"] is None


@pytest.mark.parametrize("matches", [0, 2])
def test_semantic_hint_conflict_reports_object_and_action(matches):
    from datasetui.segmentation.contract import RegionPrompt
    from datasetui.tasks import _public_failure

    mask = np.ones((2, 2), dtype=bool) if matches else np.zeros((2, 2), dtype=bool)
    predictor = FakePredictor(
        [[_frame(0, mask, discovered_mask=mask if matches else None)]]
    )
    prompt = RegionPrompt(
        object_id=2,
        frame_index=0,
        target="replace",
        text="board",
        points=[{"x": 0.5, "y": 0.5}],
    )
    with pytest.raises(Sam3InferenceError) as error:
        sam3_engine._resolve_correction_object(
            predictor, "session-0", prompt, prompt, 1, lambda: None
        )
    code, message = _public_failure(error.value)
    assert code == "segmentation_guidance"
    assert "객체 2" in message and "텍스트" in message
    assert ("없습니다" if matches == 0 else "여러 개") in message
    assert (
        _public_failure(Sam3InferenceError("/private/arbitrary error"))[1]
        == "SAM 3.1 did not produce a complete valid mask sequence"
    )


def test_detection_score_alignment_and_invalid_values():
    assert sam3_engine._detection_scores({'outputs': {'out_obj_ids': [8, 2], 'out_probs': [.2, .8]}}) == {8:.2, 2:.8}
    for scores in ([.2], [float('nan'), .8], [-1,.8], [1.1,.8]):
        with pytest.raises(Sam3InferenceError):
            sam3_engine._detection_scores({'outputs': {'out_obj_ids': [8,2], 'out_probs': scores}})
    assert sam3_engine._detection_scores({'original_detection_scores': {8:.6}, 'outputs': {'out_obj_ids':[8], 'out_probs':[1.]}}) == {8:.6}


@pytest.mark.parametrize("singleton", [False, True])
def test_selected_group_matches_masks_not_ephemeral_ids_and_refines_one_member(singleton):
    from datasetui.segmentation.contract import RegionPrompt
    left = np.array([[1,0],[1,0]], dtype=bool)
    right = ~left
    response = {'frame_index':0, 'outputs':{'out_obj_ids':np.array([70,91]),
        'out_probs':np.array([.7,.9]), 'out_binary_masks':np.array([left,right])}}
    class Predictor:
        def __init__(self): self.requests=[]
        def handle_request(self, request):
            self.requests.append(request)
            return response
        def handle_stream_request(self, request): yield response
    predictor=Predictor()
    sample='00000000-0000-0000-0000-000000000001'
    prompt=RegionPrompt(object_id=1, frame_index=0, target='protect', text='wire', confidence_threshold=.5,
        selected_candidates=[{'sample_id':sample,'candidate_id':'1-2'},{'sample_id':sample,'candidate_id':'1-3'}],
        member_candidate_id='1-3', points=[{'x':.8,'y':.8,'label':0}])
    if singleton:
        prompt = prompt.model_copy(update={'selected_candidates': prompt.selected_candidates[1:], 'member_candidate_id': None})
    result, ids=sam3_engine._add_initial_prompt(predictor,'session',prompt,1,lambda:None,
        reference_masks={(sample,'1-2'):left, (sample,'1-3'):right})
    assert ids == ({91} if singleton else {70,91})
    assert result['member_ids'] == ({'1-3':91} if singleton else {'1-2':70,'1-3':91})
    assert predictor.requests[-1]['obj_id']==91
    assert result['original_detection_scores']=={70:.7,91:.9}


def test_detector_pruning_gates_follow_requested_threshold():
    model=SimpleNamespace(score_threshold_detection=.4,new_det_thresh=.65,image_only_det_thresh=.5)
    adapter=sam3_engine._MultiplexCompatibilityAdapter(SimpleNamespace(model=model))
    adapter.configure_detection_threshold(None)
    assert model.score_threshold_detection==.4 and model.new_det_thresh==.65
    adapter.configure_detection_threshold(.1)
    assert model.score_threshold_detection==model.new_det_thresh==model.image_only_det_thresh==.1
    adapter.configure_detection_threshold(None)
    assert model.score_threshold_detection==.4 and model.new_det_thresh==.65 and model.image_only_det_thresh==.5


def test_redetected_group_correction_does_not_drop_other_members(tmp_path, monkeypatch):
    from datasetui.segmentation.contract import RegionPrompt
    left=np.array([[1,0],[1,0]],dtype=bool)
    right=~left
    response={'frame_index':0,'outputs':{'out_obj_ids':np.array([1,2]),'out_probs':np.array([.8,.9]),'out_binary_masks':np.array([left,right])}}
    class Predictor(FakePredictor):
        def handle_request(self,request):
            if request['type']=='add_prompt':
                self.prompts.append(request)
                return response
            return super().handle_request(request)
    predictor=Predictor([[response]])
    targets=sam3_engine._prepare_output_directories(tmp_path/'masks')
    result=sam3_engine._propagate_masks(predictor=predictor,video_path=tmp_path/'clip.mp4',frame_count=1,
        target_directories=targets,checkpoint_sha256='a'*64,check_lease=lambda:None,mixed_spatial=True,
        prompts=[RegionPrompt(object_id=1,frame_index=0,target='protect',text='wire',confidence_threshold=.5),
                 RegionPrompt(object_id=1,frame_index=0,target='protect',points=[{'x':0,'y':0,'label':1}])])
    assert {c['sam_object_id'] for c in result['candidates']}=={1,2}
    assert {c['sam_object_id'] for c in result['candidates'] if c['manually_refined']}=={1}


def test_unassigned_member_only_resolves_singleton():
    from datasetui.segmentation.errors import SegmentationGuidanceError
    assert sam3_engine._selected_member({'4-0': 91}, None) == 91
    assert sam3_engine._selected_member({'4-0': 91, '4-1': 70}, '4-1') == 70
    for members, candidate in [({}, None), ({'4-0': 91, '4-1': 70}, None), ({'4-0': 91}, '4-2')]:
        with pytest.raises(SegmentationGuidanceError):
            sam3_engine._selected_member(members, candidate)


def test_grounding_identity_survives_corrected_display_mask(tmp_path):
    from datasetui.segmentation.contract import RegionPrompt
    from PIL import Image
    sample = '00000000-0000-0000-0000-000000000001'
    root = tmp_path / 'segmentation-samples' / sample
    root.mkdir(parents=True)
    original = np.array([[1, 0], [0, 0]], dtype=bool)
    corrected = np.ones((2, 2), dtype=bool)
    Image.fromarray(corrected.astype('uint8') * 255).save(root / 'candidate-1-0.png')
    Image.fromarray(original.astype('uint8') * 255).save(root / 'grounding-1-0.png')
    prompt = RegionPrompt(object_id=1, frame_index=0, target='protect', text='board',
        selected_candidates=[dict(sample_id=sample, candidate_id='1-0')])
    masks = sam3_engine._load_candidate_masks(SimpleNamespace(jobs_root=tmp_path), [prompt])
    assert np.array_equal(masks[(sample, '1-0')], original)
    (root / 'grounding-1-0.png').unlink()
    legacy = sam3_engine._load_candidate_masks(SimpleNamespace(jobs_root=tmp_path), [prompt])
    assert np.array_equal(legacy[(sample, '1-0')], corrected)


def test_grounding_snapshot_is_not_mutated_by_refinement():
    mask = np.array([[1, 0], [0, 0]], dtype=bool)
    response = {'frame_index': 0, 'outputs': {'out_obj_ids': np.array([7]),
        'out_binary_masks': np.array([mask])}}
    saved = sam3_engine._grounding_masks(response, 1)
    response['outputs']['out_binary_masks'][:] = True
    assert np.array_equal(saved[7], mask)


@pytest.mark.parametrize("fails", [False, True])
def test_finished_jobs_return_cached_gpu_memory(tmp_path, monkeypatch, fails):
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"test checkpoint")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"test clip")
    mask = np.asarray([[1, 0], [0, 0]], dtype=bool)
    predictor = FakePredictor([[_frame(0, mask), _frame(1, mask)]])
    built = []

    def build(_):
        built.append(True)
        return predictor

    monkeypatch.setattr(sam3_engine, "_build_predictor", build)
    released = []
    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: True, empty_cache=lambda: released.append(True)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def lease():
        # Fail mid-inference, after the predictor holds GPU memory.
        if fails and built:
            raise RuntimeError("lease lost")

    run = lambda: Sam3Engine(_settings(checkpoint)).propagate(  # noqa: E731
        video_path=video,
        prompts=[
            {"object_id": 1, "frame_index": 0, "target": "protect",
             "points": [{"x": 0.1, "y": 0.1}]},
        ],
        frame_count=2,
        output_dir=tmp_path / "masks",
        check_lease=lease,
    )
    if fails:
        with pytest.raises(RuntimeError):
            run()
    else:
        run()
    assert released == [True]
