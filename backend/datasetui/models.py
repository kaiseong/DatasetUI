from __future__ import annotations

import json
import math
import re
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PROFILE_NAME_PATTERN = re.compile(r"^[^\x00-\x1f\x7f]+$")
JobStatus = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "interrupted",
]
StorageArea = Literal["raw", "derived"]
DatasetReadiness = Literal["ready", "incomplete", "unsupported", "invalid"]
HuggingFaceDatasetStatus = Literal[
    "not_downloaded",
    "queued",
    "downloading",
    "ready",
    "update_available",
    "incomplete",
    "validation_failed",
]


def normalize_profile_name(value: str) -> str:
    name = " ".join(unicodedata.normalize("NFKC", value).strip().split())
    if not name:
        raise ValueError("profile name cannot be empty")
    if len(name) > 80:
        raise ValueError("profile name must be at most 80 characters")
    if not PROFILE_NAME_PATTERN.fullmatch(name):
        raise ValueError("profile name cannot contain control characters")
    return name


def normalize_recipe_name(value: str) -> str:
    name = " ".join(unicodedata.normalize("NFKC", value).strip().split())
    if not name:
        raise ValueError("recipe name cannot be empty")
    if len(name) > 100:
        raise ValueError("recipe name must be at most 100 characters")
    if not PROFILE_NAME_PATTERN.fullmatch(name):
        raise ValueError("recipe name cannot contain control characters")
    return name


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProfileCreate(StrictModel):
    name: str

    _normalize_name = field_validator("name")(normalize_profile_name)


class ProfileUpdate(StrictModel):
    name: str | None = None
    archived: bool | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        return normalize_profile_name(value) if value is not None else None

    @model_validator(mode="after")
    def require_change(self) -> "ProfileUpdate":
        if self.name is None and self.archived is None:
            raise ValueError("name or archived is required")
        return self


class Profile(StrictModel):
    id: str
    name: str
    created_at: str
    updated_at: str
    archived_at: str | None


class JobCreate(StrictModel):
    kind: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    profile_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = Field(min_length=1, max_length=120)


class Job(StrictModel):
    id: str
    kind: str
    queue_name: str
    status: JobStatus
    profile_id: str
    payload: dict[str, Any]
    result: dict[str, Any] | None
    error_code: str | None
    error_message: str | None
    idempotency_key: str
    rq_job_id: str | None
    created_at: str
    enqueued_at: str | None
    started_at: str | None
    finished_at: str | None


class JobEvent(StrictModel):
    sequence: int
    event_type: str
    payload: dict[str, Any]
    created_at: str


class SystemHealth(StrictModel):
    ok: bool
    service: str
    database: Literal["ok", "error"]
    queue: Literal["ok", "error"]
    schema_versions: list[int]


class Dataset(StrictModel):
    id: str
    storage_area: StorageArea
    relative_path: str
    name: str
    codebase_version: str | None
    readiness: DatasetReadiness
    robot_type: str | None
    total_episodes: int | None
    total_frames: int | None
    total_tasks: int | None
    fps: float | None
    fingerprint: str
    scan_error: str | None
    first_seen_at: str
    last_seen_at: str
    available: bool


class EpisodeFlagChange(StrictModel):
    episode_index: int = Field(ge=0)
    flagged: bool


class EpisodeFlagPatch(StrictModel):
    profile_id: str
    expected_revision: int = Field(ge=0)
    changes: list[EpisodeFlagChange] = Field(min_length=1, max_length=500)

    @field_validator("changes")
    @classmethod
    def require_unique_episodes(
        cls, value: list[EpisodeFlagChange]
    ) -> list[EpisodeFlagChange]:
        indices = [change.episode_index for change in value]
        if len(indices) != len(set(indices)):
            raise ValueError("each episode may appear only once")
        return value


class EpisodeFlags(StrictModel):
    dataset_id: str
    dataset_fingerprint: str
    profile_id: str
    revision: int
    episode_indices: list[int]
    updated_at: str | None


AnnotationRole = Literal["user", "assistant", "system", "tool"]
AnnotationStyle = Literal[
    "task_aug", "subtask", "plan", "memory", "interjection", "vqa"
]


class SayToolArguments(StrictModel):
    text: str = Field(min_length=1, max_length=2000)


class AnnotationToolFunction(StrictModel):
    name: Literal["say"]
    arguments: SayToolArguments


class AnnotationToolCall(StrictModel):
    type: Literal["function"]
    function: AnnotationToolFunction


def _require_text(value: Any, field: str, *, maximum: int = 500) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"VQA {field} must be non-empty text")
    return value


def _require_unit_coordinates(value: Any, field: str, length: int) -> None:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"VQA {field} has an invalid shape")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        or float(item) < 0
        or float(item) > 1
        for item in value
    ):
        raise ValueError(f"VQA {field} coordinates must be between 0 and 1")


def validate_vqa_answer(content: str) -> None:
    try:
        answer = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("VQA assistant content must be valid JSON") from exc
    if not isinstance(answer, dict):
        raise ValueError("VQA assistant content must be a JSON object")
    keys = set(answer)
    if "detections" in answer:
        if keys != {"detections"} or not isinstance(answer["detections"], list):
            raise ValueError("VQA detections answer is invalid")
        if not 1 <= len(answer["detections"]) <= 100:
            raise ValueError("VQA detections must contain 1 to 100 items")
        for detection in answer["detections"]:
            if not isinstance(detection, dict) or not set(detection) <= {
                "label",
                "bbox_format",
                "bbox",
                "camera",
            }:
                raise ValueError("VQA detection is invalid")
            _require_text(detection.get("label"), "label")
            if detection.get("bbox_format") not in {"xyxy", "xywh"}:
                raise ValueError("VQA bbox_format must be xyxy or xywh")
            _require_unit_coordinates(detection.get("bbox"), "bbox", 4)
        return
    if "point" in answer:
        if not {"label", "point_format", "point"} <= keys or not keys <= {
            "label",
            "point_format",
            "point",
            "camera",
        }:
            raise ValueError("VQA keypoint answer is invalid")
        _require_text(answer.get("label"), "label")
        if answer.get("point_format") != "xy":
            raise ValueError("VQA point_format must be xy")
        _require_unit_coordinates(answer.get("point"), "point", 2)
        return
    if "count" in answer:
        if not {"label", "count"} <= keys or not keys <= {"label", "count", "note"}:
            raise ValueError("VQA count answer is invalid")
        _require_text(answer.get("label"), "label")
        count = answer.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("VQA count must be a non-negative integer")
        if "note" in answer:
            _require_text(answer["note"], "note", maximum=2000)
        return
    if "attribute" in answer:
        if keys != {"label", "attribute", "value"}:
            raise ValueError("VQA attribute answer is invalid")
        _require_text(answer.get("label"), "label")
        _require_text(answer.get("attribute"), "attribute")
        _require_text(answer.get("value"), "value", maximum=2000)
        return
    if "relation" in answer:
        if keys != {"subject", "relation", "object"}:
            raise ValueError("VQA spatial answer is invalid")
        _require_text(answer.get("subject"), "subject")
        _require_text(answer.get("relation"), "relation")
        _require_text(answer.get("object"), "object")
        return
    raise ValueError("VQA assistant answer has an unsupported shape")


class LanguageAnnotationAtom(StrictModel):
    role: AnnotationRole
    content: str | None = Field(default=None, max_length=16_000)
    style: AnnotationStyle | None
    timestamp: float = Field(ge=0, le=1_000_000_000, allow_inf_nan=False)
    camera: str | None = Field(default=None, min_length=1, max_length=200)
    tool_calls: list[AnnotationToolCall] | None = Field(default=None, max_length=4)

    @field_validator("content")
    @classmethod
    def reject_control_content(cls, value: str | None) -> str | None:
        if value is not None and ("\x00" in value or "\r" in value):
            raise ValueError("annotation content contains unsupported characters")
        return value

    @field_validator("camera")
    @classmethod
    def validate_camera(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if (
            not value.startswith("observation.")
            or "/" in value
            or "\\" in value
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise ValueError("annotation camera is invalid")
        return value

    @model_validator(mode="after")
    def validate_atom(self) -> "LanguageAnnotationAtom":
        has_content = self.content is not None and bool(self.content.strip())
        has_tools = bool(self.tool_calls)
        if not has_content and not has_tools:
            raise ValueError("annotation requires content or a tool call")
        if self.style is None:
            if self.role != "assistant" or not has_tools or has_content:
                raise ValueError(
                    "speech annotations require an assistant say tool call"
                )
        elif has_tools:
            raise ValueError("tool calls are allowed only for speech annotations")
        if self.style == "vqa":
            if self.camera is None or self.role not in {"user", "assistant"}:
                raise ValueError(
                    "VQA annotations require a camera and user/assistant role"
                )
            if self.role == "assistant":
                validate_vqa_answer(self.content or "")
        elif self.camera is not None:
            raise ValueError("only VQA annotations may select a camera")
        return self


class EpisodeAnnotationsPut(StrictModel):
    profile_id: str
    expected_revision: int = Field(ge=0)
    task_override: str | None = Field(default=None, max_length=1000)
    atoms: list[LanguageAnnotationAtom] = Field(default_factory=list, max_length=1000)

    @field_validator("task_override")
    @classmethod
    def normalize_task_override(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if "\x00" in normalized or "\r" in normalized:
            raise ValueError("task override contains unsupported characters")
        return normalized

    @field_validator("atoms")
    @classmethod
    def require_unique_atoms(
        cls, value: list[LanguageAnnotationAtom]
    ) -> list[LanguageAnnotationAtom]:
        encoded = [atom.model_dump_json() for atom in value]
        if len(encoded) != len(set(encoded)):
            raise ValueError("duplicate annotations are not allowed")
        return value


class EpisodeAnnotations(StrictModel):
    dataset_id: str
    dataset_fingerprint: str
    profile_id: str
    episode_index: int
    revision: int
    task_override: str | None
    atoms: list[LanguageAnnotationAtom]
    updated_at: str | None


CurationSelectionMode = Literal["all", "flagged", "unflagged"]
CurationOperation = Literal["subset", "delete_flagged", "train_eval_split"]


class TrimEpisodeOverride(StrictModel):
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "TrimEpisodeOverride":
        if self.end_frame <= self.start_frame:
            raise ValueError("trim end frame must be after start frame")
        return self


class TrimConfig(StrictModel):
    enabled: bool = False
    threshold: float = Field(default=0.02, ge=0, le=10)
    hold_time_s: float = Field(default=0.5, gt=0, le=30)
    margin_s: float = Field(default=1.0, ge=0, le=60)
    dimensions: list[str] = Field(default_factory=list, max_length=256)
    episode_overrides: dict[int, TrimEpisodeOverride] = Field(
        default_factory=dict, max_length=500
    )

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item or len(item) > 160 for item in cleaned):
            raise ValueError("trim dimensions must be non-empty names")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("trim dimensions cannot contain duplicates")
        return cleaned


class RelativeActionConfig(StrictModel):
    enabled: bool = False
    dimensions: list[str] = Field(default_factory=list, max_length=256)

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item or len(item) > 160 for item in cleaned):
            raise ValueError("relative action dimensions must be non-empty names")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("relative action dimensions cannot contain duplicates")
        return cleaned

    @model_validator(mode="after")
    def require_dimensions(self) -> "RelativeActionConfig":
        if self.enabled and not self.dimensions:
            raise ValueError("relative action requires at least one dimension")
        return self


class CurationRecipeCreate(StrictModel):
    profile_id: str
    name: str
    selection_mode: CurationSelectionMode
    operation: CurationOperation = "subset"
    trim_config: TrimConfig = Field(default_factory=TrimConfig)
    include_annotations: bool = False
    relative_action: RelativeActionConfig = Field(default_factory=RelativeActionConfig)

    _normalize_name = field_validator("name")(normalize_recipe_name)


class CurationRecipeUpdate(StrictModel):
    profile_id: str
    name: str | None = None
    selection_mode: CurationSelectionMode | None = None
    operation: CurationOperation | None = None
    trim_config: TrimConfig | None = None
    include_annotations: bool | None = None
    relative_action: RelativeActionConfig | None = None
    archived: bool | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        return normalize_recipe_name(value) if value is not None else None

    @model_validator(mode="after")
    def require_change(self) -> "CurationRecipeUpdate":
        if all(
            value is None
            for value in (
                self.name,
                self.selection_mode,
                self.operation,
                self.trim_config,
                self.include_annotations,
                self.relative_action,
                self.archived,
            )
        ):
            raise ValueError("at least one recipe change is required")
        return self


class CurationRecipe(StrictModel):
    id: str
    dataset_id: str
    dataset_fingerprint: str
    profile_id: str
    name: str
    selection_mode: CurationSelectionMode
    operation: CurationOperation
    trim_config: TrimConfig
    include_annotations: bool
    relative_action: RelativeActionConfig
    created_at: str
    updated_at: str
    archived_at: str | None


class CurationRecipeSnapshotCreate(StrictModel):
    profile_id: str


class CurationRecipeSnapshot(StrictModel):
    id: str
    recipe_id: str
    dataset_id: str
    dataset_fingerprint: str
    profile_id: str
    recipe_name: str
    selection_mode: CurationSelectionMode
    operation: CurationOperation
    trim_config: TrimConfig
    include_annotations: bool
    relative_action: RelativeActionConfig
    annotation_episode_indices: list[int]
    flag_revision: int
    flagged_episode_indices: list[int]
    selected_episode_indices: list[int]
    created_at: str


class CurationRunCreate(StrictModel):
    profile_id: str
    output_name: str = Field(
        min_length=1,
        max_length=96,
        pattern=r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$",
    )
    idempotency_key: str = Field(min_length=1, max_length=120)

    @field_validator("output_name")
    @classmethod
    def validate_output_name(cls, value: str) -> str:
        if ".." in value:
            raise ValueError("invalid output name")
        return value


class HuggingFaceDataset(StrictModel):
    repo_id: str
    name: str
    private: bool
    gated: bool
    downloads: int
    likes: int
    last_modified: str | None
    latest_commit_sha: str
    current_commit_sha: str | None
    tags: list[str]
    status: HuggingFaceDatasetStatus


class HuggingFaceRevision(StrictModel):
    name: str
    kind: Literal["branch", "tag"]
    commit_sha: str


class HuggingFaceImportCreate(StrictModel):
    profile_id: str
    dataset_name: str = Field(
        min_length=1,
        max_length=96,
        pattern=r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$",
    )
    requested_revision: str = Field(min_length=1, max_length=200)
    commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    idempotency_key: str = Field(min_length=1, max_length=120)

    @field_validator("dataset_name")
    @classmethod
    def reject_ambiguous_dataset_name(cls, value: str) -> str:
        if ".." in value:
            raise ValueError("invalid Hugging Face dataset name")
        return value

    @field_validator("requested_revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        if (
            value.startswith("/")
            or "\\" in value
            or any(part in {"", ".", ".."} for part in value.split("/"))
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise ValueError("invalid Hugging Face revision")
        return value


class DatasetMergeCreate(StrictModel):
    profile_id: str
    dataset_ids: list[str] = Field(min_length=2, max_length=50)
    output_name: str = Field(
        min_length=1,
        max_length=96,
        pattern=r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$",
    )
    robot_type: str = Field(min_length=1, max_length=120)
    idempotency_key: str = Field(min_length=1, max_length=120)

    @field_validator("dataset_ids")
    @classmethod
    def unique_sources(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("merge sources cannot contain duplicates")
        return value

    @field_validator("output_name")
    @classmethod
    def safe_output_name(cls, value: str) -> str:
        if ".." in value:
            raise ValueError("invalid output name")
        return value

    @field_validator("robot_type")
    @classmethod
    def safe_robot_type(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or any(ord(character) < 32 for character in normalized):
            raise ValueError("invalid robot type")
        return normalized


ValidationMode = Literal["quick", "full", "export_gate"]


class DatasetValidationCreate(StrictModel):
    profile_id: str
    mode: ValidationMode = "quick"
    idempotency_key: str = Field(min_length=1, max_length=120)


class DatasetConversionCreate(StrictModel):
    profile_id: str
    output_name: str = Field(
        min_length=1,
        max_length=96,
        pattern=r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$",
    )
    idempotency_key: str = Field(min_length=1, max_length=120)

    @field_validator("output_name")
    @classmethod
    def safe_output_name(cls, value: str) -> str:
        if ".." in value:
            raise ValueError("invalid output name")
        return value


class ValidationRun(StrictModel):
    job_id: str
    dataset_id: str
    dataset_fingerprint: str
    mode: ValidationMode
    created_at: str
    status: JobStatus
    result_json: str | None
    result: dict[str, Any] | None
    error_code: str | None
    error_message: str | None
    finished_at: str | None
