"""Versioned object-selection semantics shared by sample, preview and export."""

from __future__ import annotations
import math
import numpy as np

from datasetui.transform_errors import CurationTransformError

OBJECT_SELECTION = "object_selection"


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


def seed_brush_objects(prompts, corrections):
    from datasetui.segmentation_contract import RegionPrompt

    result = list(prompts)
    objects = {p.object_id for p in result}
    for correction in sorted(corrections, key=lambda c: c.frame_index):
        if correction.object_id is None:
            raise ValueError("브러시 힌트에 객체 번호가 필요합니다.")
        if correction.object_id not in objects and correction.operation == "add":
            point = correction.points[0]
            result.append(
                RegionPrompt(
                    object_id=correction.object_id,
                    frame_index=correction.frame_index,
                    target=correction.target,
                    points=[{"x": point.x, "y": point.y, "label": 1}],
                )
            )
            objects.add(correction.object_id)
    if len(result) > 32:
        raise ValueError("객체 지시는 최대 32개까지 지원합니다.")
    return result


class SegmentationGuidanceError(CurationTransformError, ValueError):
    """Only actionable, safe guidance messages may reach the job UI."""


def validate_initial_guidance(parsed):
    if parsed.mode != OBJECT_SELECTION:
        return
    seen = set()
    for index, prompt in enumerate(parsed.prompts):
        key = prompt.object_id if prompt.object_id is not None else -(index + 1)
        if key in seen:
            continue
        seen.add(key)
        if (
            not prompt.text
            and prompt.box is None
            and not any(p.label == 1 for p in prompt.points)
        ):
            raise SegmentationGuidanceError(
                f"객체 {prompt.object_id or index + 1}에 제외 힌트만 있습니다. "
                "포함점·Box·텍스트로 먼저 객체를 지정하세요. "
                "다른 객체의 경계를 보정하려면 해당 객체 번호에서 제외점을 찍으세요."
            )


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
