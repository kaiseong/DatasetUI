from __future__ import annotations

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


def normalize_profile_name(value: str) -> str:
    name = " ".join(unicodedata.normalize("NFKC", value).strip().split())
    if not name:
        raise ValueError("profile name cannot be empty")
    if len(name) > 80:
        raise ValueError("profile name must be at most 80 characters")
    if not PROFILE_NAME_PATTERN.fullmatch(name):
        raise ValueError("profile name cannot contain control characters")
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
