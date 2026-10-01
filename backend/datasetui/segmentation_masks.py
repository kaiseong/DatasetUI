"""Background-independent mask identity, explicit selection and protected regions."""

from __future__ import annotations

import shutil
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from datasetui.segmentation_contract import SegmentationSpec


def mask_inputs(spec: dict) -> dict:
    return {
        key: value
        for key, value in spec.items()
        if key
        not in {
            "render_mode",
            "background_base64",
            "background_sha256",
            "source_preview_id",
            "selected_candidate_ids",
        }
    }


def reuse_masks(
    database, settings, parsed, staging: Path, *, owner_profile_id: str
) -> dict:
    from datasetui.segmentation import (
        SegmentationError,
        _read_manifest,
        _safe_regular_path,
    )
    from datasetui.segmentation_api import verified_preview

    job, result = verified_preview(database, settings, str(parsed.source_preview_id))
    if job["profile_id"] != owner_profile_id:
        raise SegmentationError("Mask reuse requires the original preview profile")
    previous = SegmentationSpec.model_validate(job["payload"]["spec"])
    if mask_inputs(previous.model_dump(mode="json")) != mask_inputs(
        parsed.model_dump(mode="json")
    ):
        raise SegmentationError(
            "Mask inputs changed; generate a new segmentation preview"
        )
    root = settings.jobs_root / "segmentation" / str(parsed.source_preview_id)
    manifest = _read_manifest(root)
    if parsed.mode == "object_selection":
        from datasetui.sam3_engine import MIXED_HINT_POLICY

        if manifest.get("model", {}).get("hint_policy") != MIXED_HINT_POLICY:
            raise SegmentationError(
                "혼합 힌트 처리 방식이 변경되었습니다. 기존 마스크 재사용 대신 새로 분할하세요."
            )
    for name in (
        "original.mp4",
        *[item["artifact_name"] for item in result.get("candidates", [])],
    ):
        shutil.copyfile(_safe_regular_path(root, root / name), staging / name)
    for name in ("replace", "protect", "instances"):
        source = root / name
        if source.is_dir():
            # verified_preview fingerprint rejects symlinks before this copy.
            shutil.copytree(source, staging / name)
    return {**manifest["model"], "mask_cache_source": str(parsed.source_preview_id)}


def apply_selection(
    root: Path, parsed, provenance: dict, count: int, width: int, height: int
) -> bool:
    from datasetui.segmentation import SegmentationError, _mask_image

    candidates = provenance.get("candidates", [])
    known = {item["candidate_id"] for item in candidates}
    selected = set(parsed.selected_candidate_ids)
    if not selected <= known:
        raise SegmentationError(
            "Selected candidates are not part of this immutable mask result"
        )
    pending: dict[int, list[dict]] = {}
    for item in candidates:
        if item.get("requires_selection"):
            pending.setdefault(item["object_id"], []).append(item)
    text_unselected = "candidates" in provenance and any(
        prompt.text and not prompt.selected_candidates for prompt in parsed.prompts
    )
    # One physical object per text group: a single proposal, or proposals that
    # never appear together (it left and SAM re-detected it), is chosen
    # automatically and flagged for review. Explicit choices always win.
    auto: set[str] = set()
    unresolved = False
    for items in pending.values():
        for item in items:
            item.pop("auto_selected", None)
        if any(item["candidate_id"] in selected for item in items):
            continue
        if _one_object_over_time(items):
            for item in items:
                item["auto_selected"] = True
                auto.add(item["candidate_id"])
        else:
            unresolved = True
    if selected:
        selection_required = False
    else:
        selection_required = unresolved or (text_unselected and not pending and not candidates)
    if any(
        item["candidate_id"] in selected and item["area_pixels"] <= 0
        for item in candidates
    ):
        raise SegmentationError(
            "The selected object has no visible pixels; correct the prompt"
        )
    # Unselected text proposals may be visualized, but cannot be approved/exported.
    active = [
        item
        for item in candidates
        if not item.get("requires_selection")
        or item["candidate_id"] in selected
        or item["candidate_id"] in auto
        or selection_required
    ]
    if candidates:
        for target in ("replace", "protect"):
            (root / target).mkdir(exist_ok=True)
            for index in range(count):
                mask = np.zeros((height, width), dtype=bool)
                for candidate in active:
                    if candidate["target"] == target:
                        path = (
                            root
                            / "instances"
                            / candidate["candidate_id"]
                            / f"{index:06d}.png"
                        )
                        mask |= np.asarray(_mask_image(path, width, height)) > 0
                Image.fromarray(mask.astype(np.uint8) * 255).save(
                    root / target / f"{index:06d}.png"
                )
    return selection_required


def _one_object_over_time(items: list[dict]) -> bool:
    """True when the proposals can only be one object seen at different times."""
    if len(items) == 1:
        return True
    frames: set[int] = set()
    for item in items:
        ranges = item.get("visible_ranges")
        if not ranges:
            return False
        for start, end in ranges:
            span = set(range(int(start), int(end) + 1))
            if frames & span:
                return False
            frames |= span
    return True


def selection_review_signals(provenance: dict) -> list[dict]:
    """Automatic choices are never silent: point review at where they start."""
    signals = []
    for item in provenance.get("candidates", []):
        if item.get("reentry") or (item.get("auto_selected") and item.get("late_track")):
            signals.append(
                {
                    "frame_index": int(item.get("first_visible_frame", item.get("frame_index", 0))),
                    "reason": "reentry",
                    "candidate_id": item["candidate_id"],
                }
            )
        elif item.get("auto_selected"):
            signals.append(
                {
                    "frame_index": int(item.get("frame_index", 0)),
                    "reason": "auto_selected",
                    "candidate_id": item["candidate_id"],
                }
            )
    return signals


def object_coverage(provenance: dict, selected_ids, frame_count: int) -> list[dict]:
    """Frames on which each kept/removed object is actually present.

    Only candidates that take part in the result count (explicitly chosen,
    auto-selected, or not subject to selection). Objects whose candidates
    predate per-frame visibility tracking are omitted rather than guessed.
    """
    selected = set(selected_ids or [])
    objects: dict[int, dict] = {}
    for item in provenance.get("candidates", []):
        if item.get("requires_selection") and not (
            item.get("auto_selected") or item["candidate_id"] in selected
        ):
            continue
        entry = objects.setdefault(
            int(item["object_id"]),
            {"target": item.get("target"), "frames": set(), "known": True},
        )
        ranges = item.get("visible_ranges")
        if ranges is None:
            entry["known"] = False
            continue
        for start, end in ranges:
            entry["frames"].update(range(int(start), int(end) + 1))
    result = []
    for object_id, entry in sorted(objects.items()):
        if not entry["known"]:
            continue
        missing: list[list[int]] = []
        for frame in range(frame_count):
            if frame in entry["frames"]:
                continue
            if missing and missing[-1][1] == frame - 1:
                missing[-1][1] = frame
            else:
                missing.append([frame, frame])
        result.append(
            {
                "object_id": object_id,
                "target": entry["target"],
                "visible_frames": len(entry["frames"]),
                "frame_count": frame_count,
                "missing_ranges": missing,
            }
        )
    return result


def coverage_signals(coverage: list[dict]) -> list[dict]:
    """One review signal per object that is missing on some frames."""
    return [
        {
            "frame_index": item["missing_ranges"][0][0],
            "reason": "object_gap",
            "object_id": item["object_id"],
            "missing_frames": item["frame_count"] - item["visible_frames"],
        }
        for item in coverage
        if item["missing_ranges"]
    ]


def apply_object_corrections(root: Path, parsed, provenance, count, width, height):
    """Rasterize exact keyframes BEFORE union so erasure never cuts another track."""
    from datasetui.segmentation import SegmentationError, _mask_image

    if parsed.mode == "object_selection":
        return
    candidates = provenance.get("candidates", [])
    touched = set()
    for correction in parsed.corrections:
        if correction.object_id is None:
            continue
        matching = [
            item
            for item in candidates
            if item["object_id"] == correction.object_id
            and item["target"] == correction.target
        ]
        if not matching:
            if "candidates" in provenance:
                raise SegmentationError(
                    "The corrected object is absent from the tracked masks"
                )
            continue  # Legacy injected engines supply aggregate masks only.
        if len(matching) != 1:
            raise SegmentationError(
                "An object correction must resolve exactly one tracked candidate"
            )
        candidate = matching[0]
        directory = root / "instances" / candidate["candidate_id"]
        path = directory / f"{correction.frame_index:06d}.png"
        mask = _mask_image(path, width, height)
        draw_correction(mask, correction, width, height)
        mask.save(path)
        touched.add(candidate["candidate_id"])
    for candidate in candidates:
        if candidate["candidate_id"] not in touched:
            continue
        candidate["area_pixels"] = -1
        for index in range(count):
            mask = _mask_image(
                root / "instances" / candidate["candidate_id"] / f"{index:06d}.png",
                width,
                height,
            )
            area = int((np.asarray(mask) > 0).sum())
            if area > candidate["area_pixels"]:
                candidate["area_pixels"] = area
                candidate["frame_index"] = index
                mask.save(root / candidate["artifact_name"])


def draw_correction(mask, correction, width, height):
    draw = ImageDraw.Draw(mask)
    radius = correction.radius * width
    fill = 255 if correction.operation == "add" else 0
    for point in correction.points:
        x, y = point.x * (width - 1), point.y * (height - 1)
        draw.ellipse(
            (
                math.floor(x - radius),
                math.floor(y - radius),
                math.ceil(x + radius),
                math.ceil(y + radius),
            ),
            fill=fill,
        )


def apply_protected_regions(
    root: Path, parsed, count: int, width: int, height: int
) -> None:
    from datasetui.segmentation import SegmentationError, _mask_image

    for region in parsed.manual_regions:
        if region.frame_index is not None and region.frame_index >= count:
            raise SegmentationError("Manual protection frame is outside the episode")
    for index in range(count):
        path = root / "protect" / f"{index:06d}.png"
        mask = _mask_image(path, width, height)
        draw = ImageDraw.Draw(mask)
        for region in parsed.manual_regions:
            if parsed.camera_mode == "wrist" and region.frame_index != index:
                continue
            if region.box:
                x, y, w, h = region.box
                draw.rectangle(
                    (
                        x * (width - 1),
                        y * (height - 1),
                        (x + w) * (width - 1),
                        (y + h) * (height - 1),
                    ),
                    fill=255,
                )
            else:
                draw.polygon(
                    [
                        (point.x * (width - 1), point.y * (height - 1))
                        for point in region.points
                    ],
                    fill=255,
                )
        mask.save(path)


def review_signals(
    root: Path, parsed, count: int, width: int, height: int
) -> list[dict]:
    from datasetui.segmentation import _read_mask

    signals = []
    previous = None
    for index in range(count):
        target = "protect" if parsed.mode == "protect_foreground" else "replace"
        mask = _read_mask(root, target, index, width, height)
        if parsed.mode == "object_selection":
            from datasetui.segmentation_selection import retained_mask, has_keep_objects

            has_keep = has_keep_objects(parsed)
            mask = retained_mask(
                parsed.mode,
                _read_mask(root, "protect", index, width, height),
                _read_mask(root, "replace", index, width, height),
                has_keep=has_keep,
            )
            # Removal-only reviews measure the removed region, not retained background.
            if not has_keep:
                mask = ~mask
        area = int(mask.sum())
        if area == 0:
            signals.append({"frame_index": index, "reason": "empty_mask"})
        elif area == width * height:
            signals.append({"frame_index": index, "reason": "full_frame_mask"})
        elif previous is not None and abs(area - previous) > max(area, previous) * 0.5:
            signals.append({"frame_index": index, "reason": "abrupt_area_change"})
        previous = area
    return signals
