"""Data models for the project registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Project:
    """A registered dataset project."""

    id: str
    name: str
    source_path: str
    target_format: Literal["v2.1", "v3"]
    runtime_mode: Literal["embedded", "external"]
    output_path: str | None = None
    selected_revision: str | None = None
    python_path: str | None = None
    ffmpeg_path: str | None = None
    created_at: str = ""
    updated_at: str = ""
    last_opened_at: str = ""


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime configuration for a project."""

    mode: Literal["embedded", "external"]
    python_path: str | None = None
    ffmpeg_path: str | None = None


@dataclass(frozen=True)
class RegistryMeta:
    """Schema metadata."""

    schema_version: int
    created_at: str
    updated_at: str
