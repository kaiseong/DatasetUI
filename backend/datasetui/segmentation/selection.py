"""Pure mask rules: keep/remove composition, candidate selection, coverage."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from datasetui.job_progress import report_progress
from datasetui.segmentation.contract import SegmentationSpec, OBJECT_SELECTION
from datasetui.segmentation.errors import SegmentationError, SegmentationGuidanceError


MASK_TARGETS = ("replace", "protect")


def apply_legacy_corrections(
    root: Path,
    spec: SegmentationSpec,
    count: int,
    width: int,
    height: int,
    *,
    include_object_corrections: bool = True,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    if spec.mode == "object_selection":
        return
    by_frame: dict[tuple[str, int], list[Any]] = {}
    for correction in spec.corrections:
        if correction.object_id is not None and not include_object_corrections:
            continue
        by_frame.setdefault((correction.target, correction.frame_index), []).append(
            correction
        )
    total = count * len(MASK_TARGETS)
    completed = 0
    report_progress(
        on_progress,
        stage="write",
        completed=0,
        total=total,
        unit="items",
        current_item="마스크 보정 반영",
        force=True,
    )
    for target in MASK_TARGETS:
        directory = root / target
        if directory.is_symlink() or not directory.is_dir():
            raise SegmentationError("Segmentation engine omitted a mask directory")
        for frame_index in range(count):
            path = directory / f"{frame_index:06d}.png"
            mask = _mask_image(path, width, height)
            draw = ImageDraw.Draw(mask)
            for correction in by_frame.get((target, frame_index), []):
                radius = correction.radius * width
                fill = 255 if correction.operation == "add" else 0
                for point in correction.points:
                    x = point.x * (width - 1)
                    y = point.y * (height - 1)
                    draw.ellipse(
                        (x - radius, y - radius, x + radius, y + radius), fill=fill
                    )
            mask.save(path, format="PNG", compress_level=6)
            completed += 1
            report_progress(
                on_progress,
                stage="write",
                completed=completed,
                total=total,
                unit="items",
                current_item=f"{target} 마스크 · 프레임 {frame_index}",
            )


def read_mask(
    root: Path, target: str, index: int, width: int, height: int
) -> np.ndarray:
    return (
        np.asarray(
            _mask_image(root / target / f"{index:06d}.png", width, height),
            dtype=np.uint8,
        )
        >= 128
    )


def _mask_image(path: Path, width: int, height: int) -> Image.Image:
    if path.is_symlink() or not path.is_file():
        raise SegmentationError("Segmentation engine omitted a frame mask")
    try:
        with Image.open(path) as image:
            image.load()
            if image.size != (width, height):
                raise SegmentationError(
                    "Segmentation mask dimensions do not match video"
                )
            binary = np.where(np.asarray(image.convert("L")) >= 128, 255, 0).astype(
                np.uint8
            )
            return Image.fromarray(binary, mode="L")
    except OSError as exc:
        raise SegmentationError("Segmentation mask could not be read") from exc


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


def apply_selection(
    root: Path, parsed, provenance: dict, count: int, width: int, height: int
) -> bool:
    from datasetui.segmentation.errors import SegmentationError

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
    from datasetui.segmentation.errors import SegmentationError

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
    from datasetui.segmentation.errors import SegmentationError

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

    signals = []
    previous = None
    for index in range(count):
        target = "protect" if parsed.mode == "protect_foreground" else "replace"
        mask = read_mask(root, target, index, width, height)
        if parsed.mode == "object_selection":

            has_keep = has_keep_objects(parsed)
            mask = retained_mask(
                parsed.mode,
                read_mask(root, "protect", index, width, height),
                read_mask(root, "replace", index, width, height),
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




def retained_mask(mode, protect, remove, *, has_keep=True):
    if mode == OBJECT_SELECTION:
        return (protect if has_keep else np.ones_like(protect, dtype=bool)) & ~remove
    if mode == "protect_foreground":
        return protect
    if mode == "replace_background":
        return protect | ~remove
    raise ValueError("Unknown segmentation mode")


def has_keep_objects(spec):
    prompts = spec.get("prompts", []) if isinstance(spec, dict) else spec.prompts
    return any(
        (p.get("target") if isinstance(p, dict) else p.target) == "protect"
        for p in prompts
    )


def brush_hints(correction, width, height):
    """Bounded deterministic coverage of stroke centerline and radius in pixels."""
    points = correction.points
    centers = [
        points[int(i)]
        for i in np.linspace(0, len(points) - 1, min(7, len(points)), dtype=int)
    ]
    offsets = [(0.0, 0.0)] + [
        (math.cos(i * math.pi / 4), math.sin(i * math.pi / 4)) for i in range(8)
    ]
    result = []
    seen = set()
    for center in centers:
        for dx, dy in offsets:
            x = round(min(1.0, max(0.0, center.x + dx * correction.radius)), 6)
            y = round(
                min(1.0, max(0.0, center.y + dy * correction.radius * width / height)),
                6,
            )
            if (x, y) not in seen:
                result.append(
                    {"x": x, "y": y, "label": int(correction.operation == "add")}
                )
                seen.add((x, y))
    return result






def validate_detections(parsed, provenance):
    if parsed.mode != OBJECT_SELECTION:
        return
    candidates = provenance.get("candidates", [])
    for prompt in parsed.prompts:
        if not any(
            c.get("target") == prompt.target
            and (prompt.object_id is None or c.get("object_id") == prompt.object_id)
            and c.get("area_pixels", 0) > 0
            for c in candidates
        ):
            label = "남길" if prompt.target == "protect" else "제거할"
            raise SegmentationGuidanceError(
                f'SAM이 {label} 객체 {prompt.object_id or ""}를 찾지 못했습니다. 포함 힌트·Box·프롬프트를 보정하세요.'
            )
