"""Frozen dataclass models for the version-neutral DatasetDocument."""

from __future__ import annotations

from dataclasses import dataclass

# Re-export DatasetVersion for convenience
from .version import DatasetVersion


@dataclass(frozen=True)
class FeatureSpec:
    """A single feature column specification."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    names: tuple[str, ...] | None


@dataclass(frozen=True)
class EpisodeRef:
    """Reference to a single episode with frame range metadata."""

    index: int
    length: int  # frame count
    chunk_index: int
    file_index: int  # v3: file within chunk; v2.1: episode_index
    tasks: tuple[str, ...]
    frame_start: int  # inclusive global frame index
    frame_end: int  # exclusive global frame index


@dataclass(frozen=True)
class MediaRef:
    """Reference to a video file associated with episodes."""

    video_key: str
    path_template: str | None
    episodes_with_video: tuple[int, ...]


@dataclass(frozen=True)
class ValidationResult:
    """Result of structural validation checks."""

    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class DatasetDocument:
    """Immutable, version-neutral representation of a LeRobot dataset.

    Contains only metadata — no frame data is loaded.
    """

    source_path: str
    version: DatasetVersion
    features: tuple[FeatureSpec, ...]
    episodes: tuple[EpisodeRef, ...]
    total_frames: int
    fps: int
    media: tuple[MediaRef, ...]
    tasks: tuple[str, ...]
    has_annotations: bool
    annotation_styles: tuple[str, ...] | None
    provenance: str | None
    validation: ValidationResult
    indexed_at: str  # ISO timestamp
