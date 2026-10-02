"""Single-frame samples (optionally tracked from a detection frame)."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from io import BytesIO
from pathlib import Path
from uuid import UUID

import av
import numpy as np
from PIL import Image
from pydantic import Field

from datasetui.segmentation.contract import (
    SegmentationSpec,
    StrictModel,
    decode_background,
    validate_initial_guidance,
)
from datasetui.segmentation.errors import SegmentationGuidanceError
from datasetui.segmentation.selection import (
    brush_hints,
    has_keep_objects,
    retained_mask,
    validate_detections,
)


class SampleSpec(SegmentationSpec):
    fingerprint: None = None
    source_preview_id: None = None
    frame_token: UUID
    frame_index: int = Field(ge=0, le=999_999)
    selected_candidate_ids: list[str] = Field(default_factory=list, max_length=0)


class SampleCreate(StrictModel):
    profile_id: UUID
    idempotency_key: str = Field(min_length=1, max_length=160)
    spec: SampleSpec


MAX_TRACK_FRAMES = 24


def tracking_frames(parsed: SampleSpec) -> list[int]:
    """Source frames supplied to SAM for one sample.

    Selected Instruction candidates are bound to the frame they were detected
    on. To show such an object on another frame, SAM tracks it through a short,
    evenly spaced clip between the detection frame(s) and the viewed frame.
    """
    anchors = {parsed.frame_index} | {
        p.frame_index for p in parsed.prompts if p.selected_candidates
    }
    if len(anchors) == 1:
        return [parsed.frame_index]
    low, high = min(anchors), max(anchors)
    stride = max(1, -(-(high - low) // (MAX_TRACK_FRAMES - 1)))
    return sorted(set(range(low, high + 1, stride)) | anchors)


def frame_guidance(parsed: SampleSpec, clip_index: dict[int, int] | None = None):
    """Retain object text, but never leak spatial hints from another frame."""
    clip_index = clip_index or {parsed.frame_index: 0}
    current = clip_index[parsed.frame_index]
    if any(p.selected_candidates for p in parsed.prompts):
        # Preserve member-specific hints instead of unioning them into one prompt.
        # Selected candidates stay on their own detection frame in the clip.
        selected = [
            p.model_copy(update={"frame_index": clip_index[p.frame_index] if p.selected_candidates else current})
            for p in parsed.prompts
            if p.text or p.frame_index == parsed.frame_index
        ]
        corrections = [c.model_copy(update={"frame_index": current}) for c in parsed.corrections
                       if c.frame_index == parsed.frame_index]
        return parsed.model_copy(update={"prompts": selected, "corrections": corrections})
    objects = {}
    for prompt in parsed.prompts:
        key = (prompt.object_id, prompt.target)
        entry = objects.setdefault(
            key,
            {
                "object_id": prompt.object_id,
                "target": prompt.target,
                "frame_index": current,
                "text": "",
                "confidence_threshold": prompt.confidence_threshold,
                "selected_candidates": [r.model_dump(mode="json") for r in prompt.selected_candidates],
                "member_candidate_id": prompt.member_candidate_id,
                "points": [],
                "box": None,
            },
        )
        if prompt.text:
            entry["text"] = prompt.text
            entry["confidence_threshold"] = prompt.confidence_threshold
            entry["selected_candidates"] = [r.model_dump(mode="json") for r in prompt.selected_candidates]
        if prompt.frame_index == parsed.frame_index:
            entry["member_candidate_id"] = prompt.member_candidate_id
            entry["points"].extend(point.model_dump() for point in prompt.points)
            entry["box"] = prompt.box or entry["box"]
    prompts = [
        item
        for item in objects.values()
        if item["text"] or item["points"] or item["box"]
    ]
    corrections = [
        item.model_copy(update={"frame_index": current})
        for item in parsed.corrections
        if item.frame_index == parsed.frame_index
    ]
    regions = [
        item.model_copy(update={"frame_index": current})
        for item in parsed.manual_regions
        if item.frame_index is None or item.frame_index == parsed.frame_index
    ]
    from datasetui.segmentation.contract import RegionPrompt, seed_brush_objects

    parsed_prompts = [RegionPrompt.model_validate(item) for item in prompts]
    if parsed.mode == "object_selection":
        parsed_prompts = seed_brush_objects(parsed_prompts, corrections)
    guidance = parsed.model_copy(
        update={
            "prompts": parsed_prompts,
            "corrections": corrections,
            "manual_regions": regions,
        }
    )

    validate_initial_guidance(guidance)
    return guidance


def create_sample(database, settings, *, job_id, worker_id, spec, engine=None):
    from datasetui.job_progress import JobProgressReporter
    from datasetui.segmentation.engine import default_engine
    from datasetui.segmentation.frames import read_snapshot_frame
    from datasetui.segmentation.selection import (
        apply_legacy_corrections,
        apply_object_corrections,
        apply_protected_regions,
        apply_selection,
        read_mask,
    )

    progress = JobProgressReporter(database, job_id=job_id, worker_id=worker_id)

    def report(stage, completed, total, message):
        progress(
            {
                "stage": stage,
                "completed": completed,
                "total": total,
                "unit": "frames",
                "current_item": message,
            }
        )

    report("preparing", 0, 0, "선택한 프레임 원본 확인")
    parsed = SampleSpec.model_validate(spec)
    from datasetui.segmentation.candidates import validate_candidate_references
    if any(p.selected_candidates for p in parsed.prompts):
        validate_candidate_references(database, settings, parsed, database.get_job(job_id)["profile_id"])
    clip_frames = tracking_frames(parsed)
    clip_index = {frame: index for index, frame in enumerate(clip_frames)}
    current = clip_index[parsed.frame_index]
    guidance = frame_guidance(parsed, clip_index)
    if not guidance.prompts and not guidance.manual_regions:
        raise SegmentationGuidanceError("현재 프레임에 적용할 텍스트 또는 라벨이 필요합니다.")
    from datasetui.job_progress import WeightedProgress
    from datasetui.segmentation.engine import engine_progress, estimated_sam_passes

    count = len(clip_frames)
    progress = WeightedProgress(
        progress,
        {
            "preparing": 1,
            "read": count,
            "segment": count * 4 * estimated_sam_passes(guidance) if guidance.prompts else 0,
            "write": 1,
        },
    )

    def lease():
        database.assert_job_lease(job_id, worker_id=worker_id)

    lease()
    raw = read_snapshot_frame(
        database,
        settings,
        str(parsed.dataset_id),
        str(parsed.frame_token),
        parsed.episode_index,
        parsed.video_key,
        parsed.frame_index,
    )
    frame = Image.open(BytesIO(raw)).convert("RGB")
    width, height = frame.size
    clip = []
    for source_frame in clip_frames:
        if source_frame == parsed.frame_index:
            clip.append(frame)
            continue
        other = Image.open(BytesIO(read_snapshot_frame(
            database, settings, str(parsed.dataset_id), str(parsed.frame_token),
            parsed.episode_index, parsed.video_key, source_frame,
        ))).convert("RGB")
        if other.size != frame.size:
            raise ValueError("Tracked sample frames changed dimensions")
        clip.append(other)
        report("read", len(clip), count, "추적 클립 프레임 읽기")
        lease()
    parent = settings.jobs_root / "segmentation-samples"
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / job_id
    if parent.is_symlink() or final.exists() or final.is_symlink():
        raise ValueError("Sample output is unavailable")
    staging = Path(tempfile.mkdtemp(prefix=f".{job_id}-", dir=parent))
    try:
        frame.save(staging / "original.png")
        # The viewed frame (plus a short tracking clip only when a selected
        # candidate was detected on another frame) is supplied to SAM.
        with av.open(str(staging / "input.mp4"), mode="w") as output:
            stream = output.add_stream("libx264", rate=1)
            stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
            for index, image in enumerate(clip):
                video_frame = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
                video_frame.pts = index
                for packet in stream.encode(video_frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        prompts = [item.model_dump(mode="json") for item in guidance.prompts]
        for correction in guidance.corrections:
            if correction.object_id is not None:
                seeds = [
                    correction.points[int(index)]
                    for index in np.linspace(
                        0,
                        len(correction.points) - 1,
                        min(64, len(correction.points)),
                        dtype=int,
                    )
                ]
                prompts.append(
                    {
                        "object_id": correction.object_id,
                        "member_candidate_id": correction.member_candidate_id,
                        "target": correction.target,
                        "frame_index": current,
                        "points": brush_hints(correction, width, height)
                        if parsed.mode == "object_selection"
                        else [
                            {
                                "x": p.x,
                                "y": p.y,
                                "label": int(correction.operation == "add"),
                            }
                            for p in seeds
                        ],
                    }
                )
        report("read", count, count, "프레임 준비 완료")
        report("segment", 0, 0, "SAM3.1 현재 프레임 추론")
        if prompts:
            segmenter = engine or default_engine(
                settings, mixed_spatial=parsed.mode == "object_selection"
            )
            provenance = segmenter.propagate(
                video_path=staging / "input.mp4",
                prompts=prompts,
                frame_count=len(clip),
                output_dir=staging,
                check_lease=lease,
                **engine_progress(segmenter, progress),
            )
        else:
            provenance = {"engine": "manual-protection", "candidates": []}
            for target in ("protect", "replace"):
                (staging / target).mkdir()
                Image.new("L", frame.size).save(staging / target / "000000.png")
        validate_detections(guidance, provenance)
        if len(clip) > 1:
            # Detection identity was checked across the clip; the sample shows
            # only the viewed frame (an object may legitimately be off-camera).
            _keep_single_frame(staging, current, len(clip), provenance, parsed.frame_index)
            guidance = guidance.model_copy(update={
                "prompts": [p.model_copy(update={"frame_index": 0}) for p in guidance.prompts],
                "corrections": [c.model_copy(update={"frame_index": 0}) for c in guidance.corrections],
            })
        report("write", 0, 1, "샘플 이미지 합성")
        apply_object_corrections(staging, guidance, provenance, 1, width, height)
        apply_selection(staging, guidance, provenance, 1, width, height)
        apply_legacy_corrections(
            staging,
            guidance,
            1,
            width,
            height,
            include_object_corrections="candidates" not in provenance,
        )
        apply_protected_regions(staging, guidance, 1, width, height)
        mask = retained_mask(
            parsed.mode,
            read_mask(staging, "protect", 0, width, height),
            read_mask(staging, "replace", 0, width, height),
            has_keep=has_keep_objects(guidance),
            margin_px=parsed.edge_margin_px,
        )
        mask_image = Image.fromarray(mask.astype("uint8") * 255)
        mask_image.save(staging / "mask.png")
        background = (
            decode_background(parsed.background_base64).resize(frame.size)
            if parsed.render_mode == "image"
            else Image.new("RGB", frame.size)
        )
        Image.composite(frame, background, mask_image).save(staging / "composite.png")
        # Recheck only immutable selected source and metadata dependencies.
        read_snapshot_frame(
            database,
            settings,
            str(parsed.dataset_id),
            str(parsed.frame_token),
            parsed.episode_index,
            parsed.video_key,
            parsed.frame_index,
            verify_only=True,
        )
        lease()
        artifacts = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in staging.glob("*.png")
            if not p.is_symlink()
        }
        report("complete", 1, 1, "샘플 이미지 준비 완료")
        staging.rename(final)
        return {
            "sample_id": job_id,
            "result_type": "segmentation.sample",
            "frame_count": 1,
            **({"tracked_clip": {"source_frames": clip_frames}} if len(clip) > 1 else {}),
            "source_frame_index": parsed.frame_index,
            "candidates": provenance.get("candidates", []),
            "model_provenance": provenance,
            "artifacts": artifacts,
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _keep_single_frame(staging: Path, index: int, count: int, provenance: dict, source_frame: int) -> None:
    """Reduce a tracking clip's masks to the viewed frame (stored as frame 0)."""
    from datasetui.segmentation.engine import _write_mask

    directories = [staging / "protect", staging / "replace"]
    instances = staging / "instances"
    if instances.is_dir():
        directories += sorted(path for path in instances.iterdir() if path.is_dir())
    for directory in directories:
        for frame in range(count):
            path = directory / f"{frame:06d}.png"
            if frame != index:
                path.unlink(missing_ok=True)
        kept = directory / f"{index:06d}.png"
        if index and kept.exists():
            kept.replace(directory / "000000.png")
    for candidate in provenance.get("candidates", []):
        path = instances / candidate["candidate_id"] / "000000.png"
        with Image.open(path) as image:
            mask = np.asarray(image) > 0
        candidate["area_pixels"] = int(mask.sum())
        candidate["frame_index"] = source_frame
        candidate["visible_ranges"] = [[0, 0]] if mask.any() else []
        _write_mask(staging / candidate["artifact_name"], mask)
