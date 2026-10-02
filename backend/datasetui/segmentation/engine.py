"""SAM 3.1 inference engine and its progress/engine factories."""

from __future__ import annotations

import hashlib
import hmac
import importlib.metadata
import json
import logging
import os
import re
import shutil
import stat
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from datasetui.config import Settings
from datasetui.segmentation.contract import RegionPrompt, SegmentationSpec


def estimated_sam_passes(parsed: SegmentationSpec) -> int:

    groups: dict[Any, list[Any]] = {}
    for index, prompt in enumerate(parsed.prompts):
        groups.setdefault(prompt.object_id or f"p{index}", []).append(prompt)
    for correction in parsed.corrections:
        if correction.object_id in groups:
            groups[correction.object_id].append(groups[correction.object_id][0])
    return _expected_passes(groups) if groups else 1


def engine_progress(engine: Any, progress: Any) -> dict[str, Any]:
    """Forward SAM frame progress when the engine supports it (fixtures may not)."""
    import inspect

    try:
        accepts = "on_progress" in inspect.signature(engine.propagate).parameters
    except (TypeError, ValueError):
        accepts = False
    if not accepts or progress is None:
        return {}

    return {"on_progress": progress}


def default_engine(settings: Settings, *, mixed_spatial: bool = False) -> Any:

    return Sam3Engine(settings, mixed_spatial=mixed_spatial)


SAM3_UPSTREAM_COMMIT = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"


MIXED_HINT_POLICY = "instance-box-points-grounding-identity-v3"


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


_TARGETS = ("replace", "protect")


logger = logging.getLogger("datasetui.sam3")


_BUILD_LOCK = threading.Lock()


# A dedicated GPU worker (rq SimpleWorker, no per-job fork) may keep one loaded
# predictor and the identity of the checkpoint it already verified.
_CACHE_LOCK = threading.Lock()


_CACHED_PREDICTOR: dict[str, Any] = {}


_VERIFIED_CHECKPOINTS: dict[tuple, str] = {}


def _reuse_predictor() -> bool:
    return os.environ.get("DATASETUI_SAM3_REUSE_PREDICTOR", "").strip() == "1"


class Sam3UnavailableError(RuntimeError):
    """SAM 3.1 cannot safely run with the configured local runtime."""


class Sam3InferenceError(RuntimeError):
    """SAM 3.1 returned incomplete or malformed inference output."""


class Sam3PromptMatchError(Sam3InferenceError):
    """Allowlisted guidance failure, never an arbitrary predictor exception."""

    def __init__(self, object_id, matches):
        detail = (
            "일치하는 후보가 없습니다"
            if matches == 0
            else "일치하는 후보가 여러 개입니다"
        )
        super().__init__(
            f"객체 {object_id or '?'}: 텍스트·Box로 찾은 대상과 포함점이 "
            f"하나의 객체로 연결되지 않습니다 ({detail}). "
            "텍스트와 라벨이 같은 대상을 가리키도록 보정하세요. "
            "Point·Box·Brush만 사용하려면 이 객체의 Instruction에서 텍스트를 지우세요."
        )


class Sam3Engine:
    def __init__(self, settings: Settings, *, mixed_spatial: bool = False):
        self.settings = settings
        self.mixed_spatial = mixed_spatial

    def propagate(
        self,
        *,
        video_path: Path,
        prompts: list[dict[str, Any]],
        frame_count: int,
        output_dir: Path,
        check_lease: Callable[[], None],
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if (
            isinstance(frame_count, bool)
            or not isinstance(frame_count, int)
            or frame_count < 1
            or frame_count > self.settings.segmentation_max_frames
        ):
            raise ValueError("frame count exceeds the configured segmentation limit")
        if not 1 <= len(prompts) <= 132:
            raise ValueError(
                "segmentation requires bounded prompts and keyframe corrections"
            )
        if video_path.is_symlink() or not video_path.is_file():
            raise ValueError("segmentation video is unavailable")

        normalized_prompts = [RegionPrompt.model_validate(prompt) for prompt in prompts]
        if any(prompt.frame_index >= frame_count for prompt in normalized_prompts):
            raise ValueError("prompt frame is outside the requested video range")

        checkpoint, checkpoint_sha256 = self._verified_checkpoint(check_lease)
        target_directories = _prepare_output_directories(output_dir)
        check_lease()
        reuse = _reuse_predictor()
        key = f"{checkpoint}:{checkpoint_sha256}"
        predictor = None
        with _CACHE_LOCK:
            cached = list(_CACHED_PREDICTOR.items())
            _CACHED_PREDICTOR.clear()
        for cached_key, cached_predictor in cached:
            if reuse and cached_key == key:
                predictor = cached_predictor
            else:
                try:
                    cached_predictor.shutdown()
                except Exception:
                    logger.error("failed to shut down a stale SAM 3.1 predictor")
        if predictor is None:
            predictor = _build_predictor(checkpoint)
        succeeded = False
        try:
            result = _propagate_masks(
                predictor=predictor,
                video_path=video_path,
                prompts=normalized_prompts,
                frame_count=frame_count,
                target_directories=target_directories,
                checkpoint_sha256=checkpoint_sha256,
                check_lease=check_lease,
                mixed_spatial=self.mixed_spatial,
                candidate_masks=_load_candidate_masks(self.settings, normalized_prompts),
                on_progress=on_progress,
            )
            succeeded = True
            return result
        finally:
            kept = False
            if reuse and succeeded:
                try:
                    # Restore the builder's detection gates for the next job.
                    configure = getattr(predictor, "configure_detection_threshold", None)
                    if callable(configure):
                        configure(None)
                    with _CACHE_LOCK:
                        _CACHED_PREDICTOR[key] = predictor
                    kept = True
                except Exception:
                    logger.error("failed to keep the SAM 3.1 predictor; reloading next job")
            if not kept:
                # Failed or interrupted jobs never leave partial state behind.
                shutdown = getattr(predictor, "shutdown", None)
                if callable(shutdown):
                    try:
                        shutdown()
                    except Exception:
                        logger.error("failed to shut down the SAM 3.1 predictor")
            _release_cached_gpu_memory()

    def _verified_checkpoint(self, check_lease: Callable[[], None]) -> tuple[Path, str]:
        checkpoint = self.settings.sam3_checkpoint
        expected = self.settings.sam3_checkpoint_sha256.strip().lower()
        if checkpoint is None:
            raise Sam3UnavailableError("SAM 3.1 checkpoint is not configured")
        if not _SHA256_PATTERN.fullmatch(expected):
            raise Sam3UnavailableError("SAM 3.1 checkpoint SHA-256 is not configured")
        try:
            metadata = checkpoint.lstat()
        except OSError as exc:
            raise Sam3UnavailableError("SAM 3.1 checkpoint is unavailable") from exc
        if checkpoint.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise Sam3UnavailableError("SAM 3.1 checkpoint path is unsafe")
        identity = (
            str(checkpoint),
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        if _reuse_predictor() and hmac.compare_digest(
            _VERIFIED_CHECKPOINTS.get(identity, ""), expected
        ):
            check_lease()
            return checkpoint, expected

        digest = hashlib.sha256()
        try:
            descriptor = os.open(checkpoint, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                before = os.fstat(descriptor)
                while chunk := os.read(descriptor, 8 * 1024 * 1024):
                    check_lease()
                    digest.update(chunk)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise Sam3UnavailableError("SAM 3.1 checkpoint could not be read") from exc
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise Sam3UnavailableError("SAM 3.1 checkpoint changed during verification")
        actual = digest.hexdigest()
        if not hmac.compare_digest(actual, expected):
            raise Sam3UnavailableError("SAM 3.1 checkpoint SHA-256 does not match")
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == identity[1:5]:
            _VERIFIED_CHECKPOINTS[identity] = actual
        check_lease()
        return checkpoint, actual


def _release_cached_gpu_memory() -> None:
    """Return a finished job's cached CUDA blocks: an idle worker keeps only the model.

    The caching allocator otherwise holds the largest job's peak (tens of GB)
    for the life of the worker process.
    """
    try:
        import torch
    except (ImportError, OSError):
        return
    try:
        if torch.cuda.is_available():
            import gc

            gc.collect()
            torch.cuda.empty_cache()
    except Exception:
        logger.error("failed to release cached SAM 3.1 GPU memory")


def _build_predictor(checkpoint: Path) -> Any:
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise Sam3UnavailableError("SAM 3.1 CUDA runtime is not installed") from exc
    if not torch.cuda.is_available():
        raise Sam3UnavailableError("SAM 3.1 requires an available CUDA GPU")
    _verify_runtime_commit()
    try:
        import sam3.model_builder as model_builder
    except (ImportError, OSError) as exc:
        raise Sam3UnavailableError("SAM 3.1 runtime is not installed") from exc
    predictor = None
    try:
        # This pinned upstream builder passes the COMBINED detector+tracker
        # checkpoint to a bare tracker first. Avoid that partial non-strict load;
        # the official final combined model still loads the verified local file.
        with _BUILD_LOCK:
            tracker_builder = model_builder.build_sam3_multiplex_video_model

            def build_uninitialized_tracker(*args, **kwargs):
                kwargs.update(checkpoint_path=None, load_from_HF=False)
                return tracker_builder(*args, **kwargs)

            model_builder.build_sam3_multiplex_video_model = build_uninitialized_tracker
            try:
                predictor = model_builder.build_sam3_multiplex_video_predictor(
                    checkpoint_path=str(checkpoint),
                    use_fa3=False,
                    use_rope_real=False,
                    compile=False,
                    warm_up=False,
                    async_loading_frames=False,
                )
            finally:
                model_builder.build_sam3_multiplex_video_model = tracker_builder
        state = torch.load(
            str(checkpoint), map_location="cpu", weights_only=True, mmap=True
        )
        if isinstance(state, dict) and isinstance(state.get("model"), dict):
            state = state["model"]
        if not isinstance(state, dict):
            raise Sam3UnavailableError("SAM 3.1 checkpoint is not a state dictionary")
        # Upstream uses strict=False at the final load too. Do not accept missing,
        # unexpected, or differently shaped trained weights in production.
        predictor.model.load_state_dict(state, strict=True)
        del state
        return _MultiplexCompatibilityAdapter(predictor)
    except Sam3UnavailableError:
        if predictor is not None:
            _MultiplexCompatibilityAdapter(predictor).shutdown()
        raise
    except Exception as exc:
        if predictor is not None:
            _MultiplexCompatibilityAdapter(predictor).shutdown()
        raise Sam3UnavailableError(
            "SAM 3.1 predictor could not load the verified checkpoint"
        ) from exc


class _MultiplexCompatibilityAdapter:
    """Narrow fixes for the pinned predictor's unsupported init kwarg/lifecycle."""

    def __init__(self, predictor):
        self.predictor = predictor
        self.closed = False
        self.original_detection_thresholds = None

    def configure_detection_threshold(self, threshold):
        # Pinned upstream has separate pruning and new-track admission gates.
        # Set both before starting each independent object session.
        if threshold is None and self.original_detection_thresholds is None:
            return  # Existing recipes retain the pinned builder's own defaults.
        names = ("score_threshold_detection", "new_det_thresh", "image_only_det_thresh")
        if self.original_detection_thresholds is None:
            if not all(hasattr(self.predictor.model, name) for name in names):
                raise Sam3UnavailableError("SAM detector threshold contract changed")
            self.original_detection_thresholds = {name: getattr(self.predictor.model, name) for name in names}
        for name in names:
            setattr(self.predictor.model, name,
                    self.original_detection_thresholds[name] if threshold is None else threshold)

    def handle_request(self, request):
        if request.get("type") != "start_session":
            if (
                request.get("type") == "add_prompt"
                and request.get("points") is not None
            ):
                session = self.predictor._all_inference_states.get(
                    request.get("session_id"), {}
                )
                state = session.get("state", {})
                # Pinned _build_sam2_output erroneously returns {} for a frame
                # absent from this map, discarding VALID refined mask logits.
                # Point-only sessions have no prior VG cache. Empty containers
                # permit its normal merge; no image/mask predictions are added.
                cache = state.setdefault("cached_frame_outputs", {})
                for index in range(int(state.get("num_frames", 0))):
                    cache.setdefault(index, {})
            return self.predictor.handle_request(request)
        if request.get("offload_state_to_cpu", False):
            raise Sam3InferenceError(
                "Pinned multiplex model does not support state offload"
            )
        state = self.predictor.model.init_state(
            resource_path=request["resource_path"],
            offload_video_to_cpu=request.get("offload_video_to_cpu", False),
            async_loading_frames=self.predictor.async_loading_frames,
        )
        session_id = request.get("session_id") or str(uuid.uuid4())
        now = time.time()
        self.predictor._all_inference_states[session_id] = {
            "state": state,
            "session_id": session_id,
            "start_time": now,
            "last_use_time": now,
        }
        return {"session_id": session_id}

    def handle_stream_request(self, request):
        session = self.predictor._all_inference_states.get(
            request.get("session_id"), {}
        )
        count = session.get("state", {}).get("num_frames")
        if (
            request.get("propagation_direction", "both") == "both"
            and count
            and request.get("start_frame_index") == count - 1
        ):
            # The pinned action-history parser treats a propagation begun at an
            # endpoint as complete. Start at the LAST endpoint backwards, not
            # forwards (which would process only its single conditioning frame).
            yield from self.predictor.handle_stream_request(
                {**request, "propagation_direction": "backward"}
            )
            yield from self.predictor.handle_stream_request(
                {**request, "propagation_direction": "forward"}
            )
        else:
            yield from self.predictor.handle_stream_request(request)

    def shutdown(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.predictor.shutdown()
        finally:
            context = getattr(self.predictor, "bf16_context", None)
            if context is not None:
                context.__exit__(None, None, None)


def _verify_runtime_commit() -> None:
    installed_commit = os.environ.get("DATASETUI_SAM3_COMMIT", "").strip().lower()
    if not installed_commit:
        try:
            direct_url = importlib.metadata.distribution("sam3").read_text(
                "direct_url.json"
            )
            document = json.loads(direct_url or "{}")
            installed_commit = str(
                document.get("vcs_info", {}).get("commit_id", "")
            ).lower()
        except (importlib.metadata.PackageNotFoundError, OSError, ValueError):
            installed_commit = ""
    if installed_commit != SAM3_UPSTREAM_COMMIT:
        raise Sam3UnavailableError(
            "SAM 3.1 runtime does not match the pinned upstream commit"
        )


def _propagate_masks(
    *,
    predictor: Any,
    video_path: Path,
    prompts: list[RegionPrompt],
    frame_count: int,
    target_directories: dict[str, Path],
    checkpoint_sha256: str,
    check_lease: Callable[[], None],
    mixed_spatial: bool = False,
    candidate_masks=None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    dimensions: tuple[int, int] | None = None
    grouped: dict[int, list[RegionPrompt]] = {}
    for index, prompt in enumerate(prompts):
        grouped.setdefault(prompt.object_id or index + 33, []).append(prompt)
    tracker = None
    if on_progress is not None:
        tracker = _StreamProgress(on_progress, frame_count, _expected_passes(grouped), len(grouped))
        predictor = _CountingPredictor(predictor, tracker)
    candidates: dict[str, dict[str, Any]] = {}
    output_root = next(iter(target_directories.values())).parent
    instances = output_root / "instances"
    instances.mkdir(exist_ok=True)
    for group_number, (group_id, keyframes) in enumerate(grouped.items(), start=1):
        prompt = keyframes[0]
        if tracker is not None:
            tracker.group = group_number
        check_lease()
        session_id: str | None = None
        detection_scores = {}
        threshold = prompt.confidence_threshold if prompt.text else None
        configure = getattr(predictor, "configure_detection_threshold", None)
        if callable(configure):
            # Chosen candidates passed the threshold when they were selected.
            # Re-identifying them (re-encoded clip, other frames) must not fail
            # on small score jitter; identity is decided by mask IoU instead.
            configure(
                _reidentify_threshold(threshold)
                if threshold is not None and prompt.selected_candidates
                else threshold
            )
        try:
            started = predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": str(video_path),
                    "offload_video_to_cpu": True,
                }
            )
            session_id = _session_id(started)
            prompted, explicit_ids = _add_initial_prompt(
                predictor,
                session_id,
                prompt,
                frame_count,
                check_lease,
                mixed_spatial=mixed_spatial,
                reference_masks=candidate_masks or {},
            )
            grounding_masks = prompted.get("grounding_masks", {}) if isinstance(prompted, dict) else {}
            for sam_id, grounding_mask in grounding_masks.items():
                _write_mask(output_root / f"grounding-{group_id}-{sam_id}.png", grounding_mask)
            matching_object_ids = (
                explicit_ids
                if explicit_ids is not None
                else (None if prompt.text else _prompt_object_ids(prompted))
            )
            member_ids = prompted.get("member_ids", {}) if isinstance(prompted, dict) else {}
            refined_ids = set(prompted.get("refined_ids", [])) if isinstance(prompted, dict) else set()
            if threshold is not None and not member_ids and (prompt.points or prompt.box is not None):
                refined_ids.update(explicit_ids or ())
            if threshold is not None:
                detection_scores = _detection_scores(prompted)
                gate = _reidentify_threshold(threshold) if member_ids else threshold
                eligible = {key for key, value in detection_scores.items() if value >= gate}
                matching_object_ids = eligible if matching_object_ids is None else matching_object_ids & eligible
                # A text-only object may be outside the camera on its prompt
                # frame (other episodes re-detect from frame 0). Let tracks the
                # detector admits later (new_det_thresh == threshold) qualify.
                if not matching_object_ids and not (
                    prompt.text and not member_ids and len(keyframes) == 1
                ):
                    from datasetui.segmentation.errors import SegmentationGuidanceError
                    raise SegmentationGuidanceError(
                        f"객체 {group_id}: 최소 SAM 탐지 점수 {threshold:.2f} 이상인 후보가 없습니다. 문구나 기준을 조정하세요.")
            # Corrections belong to the same predictor session/object. Apply every
            # keyframe before propagating once; never union an old track with its
            # correction from an independent inference session.
            correction_ids = (
                explicit_ids
                if explicit_ids is not None
                else _prompt_object_ids(prompted)
            )
            if threshold is not None:
                correction_ids &= matching_object_ids
            requires_selection = not bool(member_ids) and (bool(prompt.text) or len(correction_ids) > 1)
            if len(keyframes) > 1 and (
                prompt.text or (prompt.box is not None and not mixed_spatial)
            ):
                _prime_semantic_track(
                    predictor, session_id, prompt.frame_index, frame_count, check_lease
                )
            prompted_frames = {prompt.frame_index}
            for keyframe in keyframes[1:]:
                if member_ids:
                    member = _selected_member(member_ids, keyframe.member_candidate_id)
                    request = _instance_prompt_request(session_id, keyframe, member)
                    request["clear_old_points"] = keyframe.frame_index not in prompted_frames
                    predictor.handle_request(request)
                    refined_ids.add(member)
                    prompted_frames.add(keyframe.frame_index)
                    continue
                if threshold is not None and not member_ids:
                    # Cross-episode reuse has no durable instance IDs. Resolve
                    # each correction geometrically without dropping its peers.
                    member = _resolve_correction_object(predictor, session_id, prompt,
                        keyframe, frame_count, check_lease)
                    if member not in matching_object_ids:
                        raise Sam3PromptMatchError(group_id, 0)
                    request = _instance_prompt_request(session_id, keyframe, member)
                    request["clear_old_points"] = keyframe.frame_index not in prompted_frames
                    predictor.handle_request(request)
                    refined_ids.add(member)
                    prompted_frames.add(keyframe.frame_index)
                    continue
                if len(correction_ids) != 1:
                    correction_ids = {
                        _resolve_correction_object(
                            predictor,
                            session_id,
                            prompt,
                            keyframe,
                            frame_count,
                            check_lease,
                        )
                    }
                    matching_object_ids = correction_ids
                request = _instance_prompt_request(
                    session_id, keyframe, next(iter(correction_ids))
                )
                if mixed_spatial:
                    request["clear_old_points"] = (
                        keyframe.frame_index not in prompted_frames
                    )
                    prompted_frames.add(keyframe.frame_index)
                matching_object_ids = correction_ids
                predictor.handle_request(request)
            seen_frames: set[int] = set()
            # Concept tracks that appear only later (entering or re-entering the
            # camera) receive new SAM IDs. Record them; decide after the stream.
            admit_late = bool(prompt.text) and matching_object_ids is not None
            # IDs SAM already returned at the prompt are decided (kept or
            # deliberately excluded); only genuinely new tracks are "late".
            try:
                prompt_ids = _prompt_object_ids(prompted) | set(detection_scores)
            except Sam3InferenceError:
                prompt_ids = set(detection_scores)
            prompt_ids |= set(member_ids.values()) if member_ids else set()
            late_ids: set[int] = set()
            areas: dict[tuple[int, int], int] = {}
            stream = predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "start_frame_index": keyframes[-1].frame_index,
                    "max_frame_num_to_track": frame_count,
                    "propagation_direction": "both",
                }
            )
            for response in stream:
                check_lease()
                frame_index, mask = _response_mask(
                    response, frame_count, matching_object_ids
                )
                seen_frames.add(frame_index)
                current_dimensions = (mask.shape[1], mask.shape[0])
                if dimensions is None:
                    dimensions = current_dimensions
                elif dimensions != current_dimensions:
                    raise Sam3InferenceError(
                        "SAM 3.1 returned inconsistent mask dimensions"
                    )
                # A stream may revisit a frame in the reverse pass. Overwrite
                # this track's last result rather than preserving rejected pixels.
                for candidate in candidates.values():
                    if candidate["object_id"] == group_id:
                        _write_mask(
                            instances
                            / candidate["candidate_id"]
                            / f"{frame_index:06d}.png",
                            np.zeros(mask.shape, dtype=bool),
                        )
                        areas[(candidate["sam_object_id"], frame_index)] = 0
                outputs = _field(response, "outputs")
                ids = _to_numpy(_field(outputs, "out_obj_ids")).reshape(-1)
                for raw_id in ids.tolist():
                    sam_id = int(raw_id)
                    if sam_id < 0:
                        raise Sam3InferenceError(
                            "SAM 3.1 returned a negative object ID"
                        )
                    if (
                        matching_object_ids is not None
                        and sam_id not in matching_object_ids
                    ):
                        if not admit_late or sam_id in prompt_ids:
                            continue
                        late_ids.add(sam_id)
                    _, instance_mask = _response_mask(response, frame_count, {sam_id})
                    candidate_id = f"{group_id}-{sam_id}"
                    directory = instances / candidate_id
                    directory.mkdir(exist_ok=True)
                    _write_mask(directory / f"{frame_index:06d}.png", instance_mask)
                    area = int(instance_mask.sum())
                    areas[(sam_id, frame_index)] = area
                    if (
                        candidate_id not in candidates
                        or area > candidates[candidate_id]["area_pixels"]
                    ):
                        artifact = f"candidate-{candidate_id}.png"
                        _write_mask(output_root / artifact, instance_mask)
                        candidates[candidate_id] = {
                            "candidate_id": candidate_id,
                            "detection_score": detection_scores.get(sam_id),
                            "score_source": "text_detection" if sam_id in detection_scores else None,
                            "detection_frame_index": prompt.frame_index if sam_id in detection_scores else None,
                            "manually_refined": sam_id in refined_ids if (member_ids or threshold is not None) else bool(prompt.points or prompt.box or len(keyframes) > 1),
                            "object_id": group_id,
                            "sam_object_id": sam_id,
                            "target": prompt.target,
                            "frame_index": frame_index,
                            "area_pixels": area,
                            "artifact_name": artifact,
                            "requires_selection": requires_selection,
                        }
            if seen_frames != set(range(frame_count)):
                raise Sam3InferenceError("SAM 3.1 did not return every requested frame")
            if late_ids:
                _resolve_late_tracks(
                    candidates,
                    group_id=group_id,
                    late_ids=late_ids,
                    initial_ids=set(matching_object_ids or ()),
                    capacity=len(set(member_ids.values())) if member_ids else None,
                    areas=areas,
                    instances=instances,
                    output_root=output_root,
                )
        finally:
            if session_id is not None:
                try:
                    predictor.handle_request(
                        {"type": "close_session", "session_id": session_id}
                    )
                except Exception:
                    logger.error("failed to close a SAM 3.1 predictor session")

    if dimensions is None:
        raise Sam3InferenceError("SAM 3.1 returned no mask dimensions")
    width, height = dimensions
    zero_mask = np.zeros((height, width), dtype=bool)
    # Rebuild proposal thumbnails from FINAL tracks, not an earlier forward pass.
    visible: dict[str, list[int]] = {key: [] for key in candidates}
    for candidate in candidates.values():
        candidate["area_pixels"] = -1
    for frame_index in range(frame_count):
        check_lease()
        for target in _TARGETS:
            path = target_directories[target] / f"{frame_index:06d}.png"
            aggregate = zero_mask.copy()
            for candidate in candidates.values():
                if candidate["target"] != target:
                    continue
                instance = (
                    instances / candidate["candidate_id"] / f"{frame_index:06d}.png"
                )
                if not instance.exists():
                    _write_mask(instance, zero_mask)
                with Image.open(instance) as image:
                    instance_mask = np.asarray(image) > 0
                aggregate |= instance_mask
                area = int(instance_mask.sum())
                if area > 0:
                    visible[candidate["candidate_id"]].append(frame_index)
                if area > candidate["area_pixels"]:
                    candidate["area_pixels"] = area
                    candidate["frame_index"] = frame_index
                    _write_mask(output_root / candidate["artifact_name"], instance_mask)
            _write_mask(path, aggregate)
    for key, frames in visible.items():
        candidates[key]["visible_ranges"] = _frame_ranges(frames)

    check_lease()
    return {
        "engine": "sam3.1-multiplex",
        "hint_policy": MIXED_HINT_POLICY if mixed_spatial else "semantic-resolve-v1",
        "upstream_commit": SAM3_UPSTREAM_COMMIT,
        "checkpoint_sha256": checkpoint_sha256,
        "frame_count": frame_count,
        "prompt_count": len(prompts),
        "session_count": len(grouped),
        "candidates": list(candidates.values()),
        "dimensions": {"width": width, "height": height},
        "mask_values": [0, 255],
        "targets": list(_TARGETS),
    }


def _expected_passes(grouped: dict[int, list[RegionPrompt]]) -> int:
    """Estimated full-clip propagation passes (for progress only)."""
    passes = 0
    for keyframes in grouped.values():
        prompt = keyframes[0]
        passes += 1
        if len(keyframes) > 1 or (prompt.text and (prompt.points or prompt.box is not None)):
            passes += 1
        if prompt.text and not prompt.selected_candidates:
            passes += len(keyframes) - 1
    return max(1, passes)


class _StreamProgress:
    def __init__(self, report, frame_count: int, passes: int, groups: int):
        self.report, self.frame_count, self.groups = report, frame_count, groups
        self.total = frame_count * passes
        self.done = 0
        self.group = 1

    def frame(self) -> None:
        self.done += 1
        self.total = max(self.total, self.done)
        self.report(
            {
                "stage": "segment",
                "completed": self.done,
                "total": self.total,
                "unit": "frames",
                "current_item": f"SAM 추적 · 객체 {self.group}/{self.groups}",
            }
        )


class _CountingPredictor:
    """Count streamed frames without changing predictor behaviour."""

    def __init__(self, predictor, tracker: _StreamProgress):
        self._predictor = predictor
        self._tracker = tracker

    def __getattr__(self, name):
        return getattr(self._predictor, name)

    def handle_request(self, request):
        return self._predictor.handle_request(request)

    def handle_stream_request(self, request):
        for response in self._predictor.handle_stream_request(request):
            self._tracker.frame()
            yield response


def _reidentify_threshold(threshold: float) -> float:
    """Detection gate used only to find already-selected candidates again."""
    return max(0.05, float(threshold) * 0.5)


def _frame_ranges(frames: list[int]) -> list[list[int]]:
    """Inclusive [start, end] runs of visible frames."""
    ranges: list[list[int]] = []
    for frame in frames:
        if ranges and frame == ranges[-1][1] + 1:
            ranges[-1][1] = frame
        else:
            ranges.append([frame, frame])
    return ranges


def _resolve_late_tracks(
    candidates: dict[str, dict[str, Any]],
    *,
    group_id: int,
    late_ids: set[int],
    initial_ids: set[int],
    capacity: int | None,
    areas: dict[tuple[int, int], int],
    instances: Path,
    output_root: Path,
) -> None:
    """Keep concept tracks that first appear after the prompt frame.

    Selected groups (``capacity`` = number of chosen objects) accept a new track
    only while it never coexists with ``capacity`` already accepted tracks: the
    object left the camera and SAM re-detected it under a new ID. Unselected
    text objects keep every late track as an ordinary selectable candidate.
    Accepted re-entries are flagged so review can show them explicitly.
    """

    def frames(sam_id: int) -> set[int]:
        return {frame for (key, frame), area in areas.items() if key == sam_id and area > 0}

    occupancy: dict[int, int] = {}
    for sam_id in initial_ids:
        for frame in frames(sam_id):
            occupancy[frame] = occupancy.get(frame, 0) + 1
    rejected = []
    for sam_id in sorted(late_ids, key=lambda key: (min(frames(key), default=1 << 30), key)):
        candidate = candidates.get(f"{group_id}-{sam_id}")
        if candidate is None:
            continue
        visible = frames(sam_id)
        accepted = bool(visible) and (
            capacity is None or all(occupancy.get(frame, 0) < capacity for frame in visible)
        )
        if not accepted:
            rejected.append(candidate["candidate_id"])
            continue
        for frame in visible:
            occupancy[frame] = occupancy.get(frame, 0) + 1
        candidate["late_track"] = True
        candidate["first_visible_frame"] = min(visible)
        if capacity is not None:
            candidate["reentry"] = True
            candidate["requires_selection"] = False
    for candidate_id in rejected:
        candidate = candidates.pop(candidate_id)
        shutil.rmtree(instances / candidate_id, ignore_errors=True)
        (output_root / candidate["artifact_name"]).unlink(missing_ok=True)


def _prepare_output_directories(output_dir: Path) -> dict[str, Path]:
    if output_dir.is_symlink():
        raise ValueError("segmentation output path is unsafe")
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for target in _TARGETS:
        directory = output_dir / target
        if directory.is_symlink():
            raise ValueError("segmentation output path is unsafe")
        directory.mkdir(exist_ok=True)
        if any(directory.iterdir()):
            raise ValueError("segmentation output directory must be empty")
        result[target] = directory
    return result


def _resolve_correction_object(
    predictor, session_id, initial, correction, frame_count, check_lease
):
    """Resolve a geometric correction IN this session, never by stale cached IDs."""
    positive = [point for point in correction.points if point.label == 1]
    if not positive and correction.box is None:
        from datasetui.segmentation.errors import SegmentationGuidanceError

        raise SegmentationGuidanceError(
            f"객체 {initial.object_id or '?'}: 제외 힌트만으로는 어느 후보를 보정할지 알 수 없습니다. "
            "해당 프레임에서 대상 안쪽에 포함점이나 좁은 Box를 함께 지정하세요."
        )
    response_at_frame = None
    for response in predictor.handle_stream_request(
        {
            "type": "propagate_in_video",
            "session_id": session_id,
            "start_frame_index": initial.frame_index,
            "max_frame_num_to_track": frame_count,
            "propagation_direction": "both",
        }
    ):
        check_lease()
        if int(_field(response, "frame_index")) == correction.frame_index:
            response_at_frame = response
    if response_at_frame is None:
        raise Sam3InferenceError("SAM 3.1 omitted the object correction frame")
    matches = []
    for object_id in _prompt_object_ids(response_at_frame):
        _, mask = _response_mask(response_at_frame, frame_count, {object_id})
        height, width = mask.shape
        hit = bool(positive) and all(
            mask[round(point.y * (height - 1)), round(point.x * (width - 1))]
            for point in positive
        )
        if not positive and correction.box is not None:
            x, y, box_width, box_height = correction.box
            hit = bool(
                mask[
                    round(y * (height - 1)) : round((y + box_height) * (height - 1))
                    + 1,
                    round(x * (width - 1)) : round((x + box_width) * (width - 1)) + 1,
                ].any()
            )
        if hit:
            matches.append(object_id)
    if len(matches) != 1:
        raise Sam3PromptMatchError(initial.object_id, len(matches))
    return matches[0]


def _session_id(response: Any) -> str:
    value = _field(response, "session_id")
    if not isinstance(value, str) or not value:
        raise Sam3InferenceError("SAM 3.1 did not return a session identifier")
    return value


def _prompt_request(session_id: str, prompt: RegionPrompt) -> dict[str, Any]:
    request: dict[str, Any] = {
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": prompt.frame_index,
        "text": prompt.text or None,
        "points": [[point.x, point.y] for point in prompt.points] or None,
        "point_labels": [point.label for point in prompt.points] or None,
        "bounding_boxes": [list(prompt.box)] if prompt.box is not None else None,
        "bounding_box_labels": [1] if prompt.box is not None else None,
        "obj_id": 1,
        "rel_coordinates": True,
    }
    return request


def _instance_prompt_request(session_id, prompt, object_id):
    """Instance corrections must not reset semantic state with another box prompt."""
    request = _prompt_request(session_id, prompt)
    request.update(
        text=None, bounding_boxes=None, bounding_box_labels=None, obj_id=object_id
    )
    points = request["points"] or []
    labels = request["point_labels"] or []
    if prompt.box is not None:
        x, y, width, height = prompt.box
        # Official SAM prompt encoder embeds labels2/3 as the box corners.
        points = [[x, y], [x + width, y + height]] + points
        labels = [2, 3] + labels
    request["points"], request["point_labels"] = points, labels
    return request


def _add_initial_prompt(
    predictor, session_id, prompt, frame_count, check_lease, *, mixed_spatial=False,
    reference_masks=None,
):
    if prompt.selected_candidates:
        request = _prompt_request(session_id, prompt)
        request.update(points=None, point_labels=None, bounding_boxes=None, bounding_box_labels=None)
        grounded = predictor.handle_request(request)
        grounding_masks = _grounding_masks(grounded, frame_count)
        scores = _detection_scores(grounded)
        eligible = {
            key
            for key, score in scores.items()
            if score >= _reidentify_threshold(prompt.confidence_threshold or 0)
        }
        members = {}
        for ref in prompt.selected_candidates:
            reference = (reference_masks or {}).get((str(ref.sample_id), ref.candidate_id))
            if reference is None:
                raise Sam3InferenceError("Selected candidate mask is unavailable")
            matches = []
            for sam_id in eligible:
                _, mask = _response_mask(grounded, frame_count, {sam_id})
                if mask.shape != reference.shape:
                    raise Sam3InferenceError("Candidate source dimensions changed")
                union = np.count_nonzero(mask | reference)
                iou = np.count_nonzero(mask & reference) / union if union else 0
                if iou >= 0.5:
                    matches.append(sam_id)
            if len(matches) != 1 or matches[0] in members.values():
                raise Sam3PromptMatchError(prompt.object_id, len(matches))
            members[ref.candidate_id] = matches[0]
        if prompt.points or prompt.box is not None:
            member = _selected_member(members, prompt.member_candidate_id)
            _prime_semantic_track(predictor, session_id, prompt.frame_index, frame_count, check_lease)
            predictor.handle_request(_instance_prompt_request(session_id, prompt, member))
        return {**grounded, "grounding_masks": grounding_masks, "original_detection_scores": scores, "member_ids": members, "refined_ids": [member] if (prompt.points or prompt.box is not None) else []}, set(members.values())
    # A labeling box is SAM instance geometry, not a semantic detector gate.
    # Encode its corners with labels 2/3 alongside +/- points in one request.
    if mixed_spatial and not prompt.text:
        request = _instance_prompt_request(session_id, prompt, 1)
        request["clear_old_points"] = True
        return predictor.handle_request(request), {1}
    request = _prompt_request(session_id, prompt)
    if not (prompt.points or (prompt.text and prompt.box is not None)) or (not prompt.text and prompt.box is None):
        explicit_ids = {int(request["obj_id"])} if prompt.points else None
        response = predictor.handle_request(request)
        if prompt.text and prompt.confidence_threshold is not None:
            response = {**response, "grounding_masks": _grounding_masks(response, frame_count)}
        return response, explicit_ids
    # The official multiplex API forbids points together with text/boxes.
    # Ground concepts first, resolve the user's geometry in THAT session, then
    # add an instance correction without discarding the grounded track state.
    request.update(points=None, point_labels=None)
    if prompt.text:
        request.update(bounding_boxes=None, bounding_box_labels=None)
    grounded = predictor.handle_request(request)
    grounding_masks = _grounding_masks(grounded, frame_count) if prompt.text and prompt.confidence_threshold is not None else {}
    object_ids = _prompt_object_ids(grounded)
    original_scores = _detection_scores(grounded) if prompt.confidence_threshold is not None else None
    if original_scores is not None:
        object_ids &= {key for key, value in original_scores.items() if value >= prompt.confidence_threshold}
        if not object_ids:
            from datasetui.segmentation.errors import SegmentationGuidanceError
            raise SegmentationGuidanceError("최소 SAM 탐지 점수 이상인 후보가 없습니다.")
    _prime_semantic_track(
        predictor, session_id, prompt.frame_index, frame_count, check_lease
    )
    if len(object_ids) != 1:
        object_ids = {
            _resolve_correction_object(
                predictor, session_id, prompt, prompt, frame_count, check_lease
            )
        }
    refined = predictor.handle_request(
        _instance_prompt_request(session_id, prompt, next(iter(object_ids)))
    )
    if original_scores is not None:
        refined = {**refined, "original_detection_scores": original_scores}
    return {**refined, "grounding_masks": grounding_masks}, object_ids


def _prime_semantic_track(predictor, session_id, frame_index, frame_count, check_lease):
    # Multiplex interactive propagation merges SAM2 predictions with the full
    # semantic VG cache. Follow the official notebook: propagate the grounded
    # concept before refining its instance, even if only one object was found.
    seen = set()
    for response in predictor.handle_stream_request(
        {
            "type": "propagate_in_video",
            "session_id": session_id,
            "start_frame_index": frame_index,
            "max_frame_num_to_track": frame_count,
            "propagation_direction": "both",
        }
    ):
        check_lease()
        seen.add(int(_field(response, "frame_index")))
    if seen != set(range(frame_count)):
        raise Sam3InferenceError(
            "SAM 3.1 semantic initialization did not cover all frames"
        )


def _load_candidate_masks(settings, prompts):
    from datasetui.segmentation.frames import safe_snapshot_path
    masks = {}
    for prompt in prompts:
        for ref in prompt.selected_candidates:
            root = settings.jobs_root / "segmentation-samples" / str(ref.sample_id)
            grounding = root / f"grounding-{ref.candidate_id}.png"
            path = safe_snapshot_path(root, grounding if grounding.exists() else root / f"candidate-{ref.candidate_id}.png")
            with Image.open(path) as image:
                masks[(str(ref.sample_id), ref.candidate_id)] = np.asarray(image.convert("L")) > 0
    return masks


def _detection_scores(response):
    """Read semantic scores only; never infer confidence from masks or area."""
    if isinstance(response, dict) and "original_detection_scores" in response:
        return response["original_detection_scores"]
    outputs = _field(response, "outputs")
    ids = _to_numpy(_field(outputs, "out_obj_ids")).reshape(-1)
    scores = _to_numpy(_field(outputs, "out_probs")).reshape(-1)
    if len(ids) != len(scores) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise Sam3InferenceError("SAM returned invalid detection scores")
    return {int(key): float(value) for key, value in zip(ids, scores, strict=True)}


def _prompt_object_ids(response: Any) -> set[int]:
    outputs = _field(response, "outputs")
    object_ids = _to_numpy(_field(outputs, "out_obj_ids")).reshape(-1)
    try:
        return {int(value) for value in object_ids.tolist()}
    except (TypeError, ValueError) as exc:
        raise Sam3InferenceError("SAM 3.1 returned invalid object identifiers") from exc


def _response_mask(
    response: Any, frame_count: int, matching_object_ids: set[int] | None
) -> tuple[int, np.ndarray]:
    frame_index = _field(response, "frame_index")
    if (
        isinstance(frame_index, bool)
        or not isinstance(frame_index, (int, np.integer))
        or not 0 <= int(frame_index) < frame_count
    ):
        raise Sam3InferenceError("SAM 3.1 returned an invalid frame index")
    outputs = _field(response, "outputs")
    object_ids = _to_numpy(_field(outputs, "out_obj_ids")).reshape(-1)
    masks = _to_numpy(_field(outputs, "out_binary_masks"))
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    elif masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0, :, :]
    if masks.ndim != 3 or masks.shape[0] != object_ids.shape[0]:
        raise Sam3InferenceError("SAM 3.1 returned malformed mask output")
    if masks.shape[1] < 1 or masks.shape[2] < 1:
        raise Sam3InferenceError("SAM 3.1 returned invalid mask dimensions")
    matches = (
        np.ones(object_ids.shape, dtype=bool)
        if matching_object_ids is None
        else np.isin(object_ids, list(matching_object_ids))
    )
    if np.any(matches):
        union = np.any(masks[matches].astype(bool), axis=0)
    else:
        union = np.zeros(masks.shape[1:], dtype=bool)
    return int(frame_index), union


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        if name not in value:
            raise Sam3InferenceError(f"SAM 3.1 output is missing {name}")
        return value[name]
    try:
        return getattr(value, name)
    except AttributeError as exc:
        raise Sam3InferenceError(f"SAM 3.1 output is missing {name}") from exc


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "to"):
        value = value.to("cpu")
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        return np.asarray(value)
    except Exception as exc:
        raise Sam3InferenceError("SAM 3.1 returned unreadable output") from exc


def _write_mask(path: Path, mask: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
            temporary, format="PNG"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _selected_member(members, candidate_id):
    """An unassigned correction is unambiguous only for a singleton group."""
    if candidate_id is None and len(members) == 1:
        return next(iter(members.values()))
    if candidate_id in members:
        return members[candidate_id]
    from datasetui.segmentation.errors import SegmentationGuidanceError
    raise SegmentationGuidanceError("보정할 그룹 내 후보를 선택하세요.")


def _grounding_masks(response, frame_count):
    # Keep immutable semantic identity separate from the corrected display mask.
    return {sam_id: _response_mask(response, frame_count, {sam_id})[1].copy()
            for sam_id in _prompt_object_ids(response)}
