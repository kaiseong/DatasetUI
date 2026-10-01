"""Bounded, immutable inputs shared by the API and optional GPU worker."""

from __future__ import annotations

import base64
import binascii
from io import BytesIO
from typing import Annotated, Literal
from uuid import UUID

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

Coordinate = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
FrameIndex = Annotated[int, Field(ge=0, le=999_999, strict=True)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Point(StrictModel):
    x: Coordinate
    y: Coordinate


class PromptPoint(Point):
    label: Literal[0, 1] = 1


class CandidateReference(StrictModel):
    sample_id: UUID
    candidate_id: str = Field(pattern=r"^[0-9]+-[0-9]+$")


class RegionPrompt(StrictModel):
    object_id: int | None = Field(default=None, ge=1, le=32)
    frame_index: FrameIndex
    target: Literal["replace", "protect"]
    text: str = Field(default="", max_length=256)
    confidence_threshold: Coordinate | None = None
    selected_candidates: list[CandidateReference] = Field(default_factory=list, max_length=32)
    member_candidate_id: str | None = Field(default=None, pattern=r"^[0-9]+-[0-9]+$")
    points: list[PromptPoint] = Field(default_factory=list, max_length=64)
    box: tuple[Coordinate, Coordinate, Coordinate, Coordinate] | None = None

    @model_validator(mode="after")
    def valid_prompt(self):
        self.text = self.text.strip()
        if not self.text and not self.points and self.box is None:
            raise ValueError("각 영역에 텍스트, 점 또는 박스를 지정하세요.")
        if self.box:
            x, y, width, height = self.box
            if (
                width <= 0
                or height <= 0
                or x + width > 1.000001
                or y + height > 1.000001
            ):
                raise ValueError("박스가 영상 범위를 벗어났습니다.")
        return self


class Correction(StrictModel):
    member_candidate_id: str | None = Field(default=None, pattern=r"^[0-9]+-[0-9]+$")
    object_id: int | None = Field(default=None, ge=1, le=32)
    frame_index: FrameIndex
    target: Literal["replace", "protect"]
    operation: Literal["add", "erase"] = "add"
    radius: float = Field(gt=0, le=0.25, allow_inf_nan=False)
    points: list[Point] = Field(min_length=1, max_length=1000)


class ManualRegion(StrictModel):
    frame_index: FrameIndex | None = None
    box: tuple[Coordinate, Coordinate, Coordinate, Coordinate] | None = None
    points: list[Point] = Field(default_factory=list, max_length=1000)

    @model_validator(mode="after")
    def valid_region(self):
        if (self.box is None) == (not self.points):
            raise ValueError("보호 영역은 박스 또는 다각형 하나로 지정하세요.")
        if self.box:
            x, y, width, height = self.box
            if width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
                raise ValueError("보호 영역이 영상 범위를 벗어났습니다.")
        if self.points and len(self.points) < 3:
            raise ValueError("다각형 보호 영역은 점이 3개 이상 필요합니다.")
        return self


class SegmentationSpec(StrictModel):
    dataset_id: UUID
    fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    episode_index: FrameIndex
    video_key: str = Field(min_length=1, max_length=240, pattern=r"^[A-Za-z0-9_.-]+$")
    mode: Literal["replace_background", "protect_foreground", "object_selection"] = (
        "protect_foreground"
    )
    render_mode: Literal["black", "image"] | None = None
    background_base64: str | None = Field(
        default=None, min_length=4, max_length=14_000_000
    )
    camera_mode: Literal["fixed", "wrist"] = "fixed"
    manual_regions: list[ManualRegion] = Field(default_factory=list, max_length=100)
    source_preview_id: UUID | None = None
    selected_candidate_ids: list[str] = Field(default_factory=list, max_length=256)
    prompts: list[RegionPrompt] = Field(default_factory=list, max_length=32)
    corrections: list[Correction] = Field(default_factory=list, max_length=100)

    @model_validator(mode="before")
    @classmethod
    def normalize_background(cls, value):
        if isinstance(value, dict) and (
            value.get("render_mode") == "black"
            or (
                value.get("background_base64") == ""
                and value.get("render_mode") != "image"
            )
        ):
            return {**value, "background_base64": None}
        return value

    @model_validator(mode="after")
    def primary_prompt(self):
        if self.render_mode is None:
            self.render_mode = "image" if self.background_base64 else "black"
        if self.render_mode == "image" and not self.background_base64:
            raise ValueError("이미지 배경 모드에는 배경 이미지가 필요합니다.")
        if self.render_mode == "black":
            self.background_base64 = None
        if self.camera_mode == "wrist" and any(
            region.frame_index is None for region in self.manual_regions
        ):
            raise ValueError("손목 카메라의 수동 보호 영역은 프레임을 지정하세요.")
        if len(self.selected_candidate_ids) != len(set(self.selected_candidate_ids)):
            raise ValueError("후보 선택이 중복되었습니다.")
        import re

        if any(
            not re.fullmatch(r"[0-9]+-[0-9]+", item)
            for item in self.selected_candidate_ids
        ):
            raise ValueError("객체 후보 ID가 올바르지 않습니다.")
        if self.mode == "object_selection":
            from datasetui.segmentation_selection import (
                seed_brush_objects,
                validate_initial_guidance,
            )

            if self.manual_regions:
                raise ValueError(
                    "객체 선택 방식에서는 수동 보호 영역을 사용할 수 없습니다."
                )
            self.prompts = seed_brush_objects(self.prompts, self.corrections)
            validate_initial_guidance(self)
        objects = {
            item.object_id: item.target
            for item in self.prompts
            if item.object_id is not None
        }
        seen_objects = set()
        for prompt in self.prompts:
            if (
                prompt.object_id is not None
                and objects[prompt.object_id] != prompt.target
            ):
                raise ValueError("동일 객체는 같은 보존 대상을 사용해야 합니다.")
            if prompt.object_id is not None:
                if (
                    prompt.object_id in seen_objects
                    and not prompt.points
                    and prompt.box is None
                ):
                    raise ValueError(
                        "동일 객체의 키프레임 보정에는 점 또는 박스를 지정하세요."
                    )
                seen_objects.add(prompt.object_id)
        for correction in self.corrections:
            if (
                correction.object_id is not None
                and objects.get(correction.object_id) != correction.target
            ):
                raise ValueError("보정할 객체를 먼저 지정하세요.")
        primary = "replace" if self.mode == "replace_background" else "protect"
        if not any(
            self.mode == "object_selection" or prompt.target == primary
            for prompt in self.prompts
        ) and not (
            self.mode != "object_selection"
            and primary == "protect"
            and self.manual_regions
        ):
            raise ValueError("선택한 모드에 맞는 영역 프롬프트가 필요합니다.")
        return self


def decode_background(encoded: str) -> Image.Image:
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > 10 * 1024 * 1024:
            raise ValueError("배경 이미지는 10 MiB 이하만 지원합니다.")
        with Image.open(BytesIO(raw)) as source:
            if source.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("PNG, JPEG, WebP 이미지만 지원합니다.")
            if source.width * source.height > 16_000_000:
                raise ValueError("배경 이미지가 1600만 화소를 초과합니다.")
            source.load()
            return source.convert("RGB")
    except (binascii.Error, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("배경 이미지가 올바르지 않습니다.") from exc


class PendingPreviewSpec(SegmentationSpec):
    fingerprint: None = None
    frame_token: UUID


class PreviewCreate(StrictModel):
    profile_id: UUID
    idempotency_key: str = Field(min_length=1, max_length=160)
    spec: SegmentationSpec | PendingPreviewSpec


class PreviewApprove(StrictModel):
    profile_id: UUID
    recipe_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class SegmentationExport(StrictModel):
    profile_id: UUID
    idempotency_key: str = Field(min_length=1, max_length=160)
    preview_id: UUID | None = None
    approval_token: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    previews: list["ApprovedPreview"] = Field(default_factory=list, max_length=1000)
    recompute_statistics: bool = False
    output_name: str = Field(
        min_length=1,
        max_length=96,
        pattern=r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$",
    )

    @model_validator(mode="after")
    def safe_output(self):
        if bool(self.previews) == bool(self.preview_id):
            raise ValueError("하나의 미리보기 또는 승인된 미리보기 목록을 지정하세요.")
        if self.preview_id and not self.approval_token:
            raise ValueError("미리보기 승인 토큰이 필요합니다.")
        if len({item.preview_id for item in self.previews}) != len(self.previews):
            raise ValueError("미리보기 목록에 중복이 있습니다.")
        if ".." in self.output_name:
            raise ValueError("출력 이름에 ..을 사용할 수 없습니다.")
        return self


class ApprovedPreview(StrictModel):
    preview_id: UUID
    approval_token: str = Field(pattern=r"^[a-f0-9]{64}$")


SegmentationExport.model_rebuild()
