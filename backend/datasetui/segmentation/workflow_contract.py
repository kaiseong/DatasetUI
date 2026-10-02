"""Request models for camera templates and episode x camera batches."""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, model_serializer, model_validator

from datasetui.segmentation.contract import (
    Correction,
    EdgeMargin,
    FrameIndex,
    ManualRegion,
    RegionPrompt,
    StrictModel,
    omit_zero_edge_margin,
)

Fingerprint = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


VideoKey = Annotated[
    str, Field(min_length=1, max_length=240, pattern=r"^[A-Za-z0-9_.-]+$")
]


MAX_BATCH_ITEMS = 256


MAX_TEMPLATE_BYTES = 1024 * 1024


class ProfileRequest(StrictModel):
    profile_id: UUID


class CameraTemplate(StrictModel):
    reuse_policy: Literal["legacy", "text_by_default"] = "legacy"
    source_episode_index: FrameIndex | None = None
    video_key: VideoKey
    camera_mode: Literal["fixed", "wrist"]
    mode: Literal["protect_foreground", "replace_background", "object_selection"] = (
        "protect_foreground"
    )
    prompts: list[RegionPrompt] = Field(default_factory=list, max_length=32)
    corrections: list[Correction] = Field(default_factory=list, max_length=100)
    manual_regions: list[ManualRegion] = Field(default_factory=list, max_length=32)
    edge_margin_px: EdgeMargin = 0

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        return omit_zero_edge_margin(handler(self))

    @model_validator(mode="after")
    def reusable_prompts(self):
        if self.mode == "object_selection":
            from datasetui.segmentation.contract import (
                seed_brush_objects,
                validate_initial_guidance,
            )

            if self.manual_regions:
                raise ValueError(
                    "객체 선택 방식에서는 수동 보호 영역을 사용할 수 없습니다."
                )
            self.prompts = seed_brush_objects(self.prompts, self.corrections)
            validate_initial_guidance(self)
        primary = "protect" if self.mode == "protect_foreground" else "replace"
        if not any(
            self.mode == "object_selection" or prompt.target == primary
            for prompt in self.prompts
        ) and not (self.mode == "protect_foreground" and self.manual_regions):
            raise ValueError("선택한 모드에 맞는 영역 프롬프트가 필요합니다.")
        if self.camera_mode == "wrist" and not any(
            (self.mode == "object_selection" or prompt.target == primary)
            and prompt.text.strip()
            for prompt in self.prompts
        ):
            raise ValueError(
                "손목 카메라 템플릿에는 재탐지할 대상의 텍스트가 필요합니다."
            )
        return self


class TemplateSave(ProfileRequest):
    name: str = Field(min_length=1, max_length=120)
    dataset_id: UUID
    fingerprint: Fingerprint | None = None
    metadata_revision: Fingerprint | None = None
    cameras: list[CameraTemplate] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def unique_cameras(self):
        self.name = self.name.strip()
        if not self.name:
            raise ValueError("템플릿 이름을 입력하세요.")
        keys = [camera.video_key for camera in self.cameras]
        if len(set(keys)) != len(keys):
            raise ValueError("카메라별 템플릿을 하나씩 지정하세요.")
        if len(self.model_dump_json().encode()) > MAX_TEMPLATE_BYTES:
            raise ValueError("템플릿은 1 MiB 이하로 줄여 주세요.")
        return self


class BatchCreate(ProfileRequest):
    idempotency_key: str = Field(min_length=1, max_length=160)
    template_id: UUID
    dataset_id: UUID
    fingerprint: Fingerprint
    episode_indices: list[FrameIndex] = Field(min_length=1, max_length=MAX_BATCH_ITEMS)
    video_keys: list[VideoKey] = Field(min_length=1, max_length=16)
    same_camera_setup_confirmed: bool = False

    @model_validator(mode="after")
    def bounded_selection(self):
        if len(set(self.episode_indices)) != len(self.episode_indices):
            raise ValueError("에피소드를 중복 선택할 수 없습니다.")
        if len(set(self.video_keys)) != len(self.video_keys):
            raise ValueError("카메라를 중복 선택할 수 없습니다.")
        if len(self.episode_indices) * len(self.video_keys) > MAX_BATCH_ITEMS:
            raise ValueError(
                f"한 번에 {MAX_BATCH_ITEMS}개 에피소드·카메라까지 선택하세요."
            )
        return self


class BatchApprove(ProfileRequest):
    recipe_hash: Fingerprint


class BatchPreviewBind(ProfileRequest):
    preview_id: UUID


class BatchExportCreate(ProfileRequest):
    idempotency_key: str = Field(min_length=1, max_length=160)
    output_name: str = Field(
        min_length=1,
        max_length=96,
        pattern=r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$",
    )
    recompute_statistics: bool = False

    @model_validator(mode="after")
    def safe_name(self):
        if ".." in self.output_name:
            raise ValueError("출력 이름에 ..을 사용할 수 없습니다.")
        return self


class BatchExportPayload(ProfileRequest):
    batch_id: UUID
    output_name: str = Field(
        min_length=1,
        max_length=96,
        pattern=r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$",
    )
    recompute_statistics: bool = False
    preview_ids: list[UUID] = Field(min_length=1, max_length=MAX_BATCH_ITEMS)
    approval_hashes: list[Fingerprint] = Field(min_length=1, max_length=MAX_BATCH_ITEMS)

    @model_validator(mode="after")
    def valid_selection(self):
        if ".." in self.output_name:
            raise ValueError("출력 이름에 ..을 사용할 수 없습니다.")
        if len(self.preview_ids) != len(self.approval_hashes):
            raise ValueError("전체 선택에 대한 승인이 필요합니다.")
        if len(set(self.preview_ids)) != len(self.preview_ids):
            raise ValueError("미리보기를 중복 선택할 수 없습니다.")
        return self


class BatchPrepare(BatchCreate):
    fingerprint: None = None
    metadata_revision: Fingerprint
