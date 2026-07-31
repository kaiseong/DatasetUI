"""LeRobot version detection with legacy read support and future fail-close."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetVersion:
    family: str
    version: str
    raw: str


class UnsupportedVersionError(Exception):
    def __init__(self, version: str, reason: str = "") -> None:
        detail = f": {reason}" if reason else ""
        super().__init__(f"Unsupported dataset version '{version}'{detail}")
        self.version = version


_EXACT_READABLE = frozenset({"v2.0", "v2.1", "v3.0"})
_LEGACY_V1 = re.compile(r"^v1\.[0-9]+$")


def _has_language_features(info: dict[str, Any]) -> bool:
    features = info.get("features")
    if not isinstance(features, dict):
        return False
    required = {"language_persistent", "language_events"}
    return required <= features.keys() or any(
        isinstance(spec, dict) and spec.get("dtype") == "language"
        for spec in features.values()
    )


def detect_version(info: dict[str, Any]) -> DatasetVersion:
    """Detect supported read layouts; reject unknown/future core versions."""
    raw = info.get("codebase_version")
    if raw is None:
        raise UnsupportedVersionError("(missing)", reason="missing codebase_version field")
    if not isinstance(raw, str):
        raise UnsupportedVersionError(str(raw), reason="codebase_version must be a string")
    if raw not in _EXACT_READABLE and _LEGACY_V1.fullmatch(raw) is None:
        raise UnsupportedVersionError(raw)
    if raw == "v3.0" and _has_language_features(info):
        return DatasetVersion(family="lerobot", version="v3.1", raw=raw)
    return DatasetVersion(family="lerobot", version=raw, raw=raw)


def uses_legacy_episode_layout(version: DatasetVersion) -> bool:
    """Return whether metadata is expected in JSONL/per-episode form."""
    return version.version.startswith("v1.") or version.version in {"v2.0", "v2.1"}
