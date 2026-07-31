"""Tests for version detection — fail-closed on unknown versions."""

from __future__ import annotations

import pytest

from lerobot_dataset_editor.dataset.version import (
    DatasetVersion,
    UnsupportedVersionError,
    detect_version,
)


class TestDetectVersion:
    """Version detection from info.json metadata."""

    def test_detects_v21(self) -> None:
        info = {"codebase_version": "v2.1"}
        result = detect_version(info)
        assert result == DatasetVersion(family="lerobot", version="v2.1", raw="v2.1")

    def test_detects_v30(self) -> None:
        info = {"codebase_version": "v3.0"}
        result = detect_version(info)
        assert result == DatasetVersion(family="lerobot", version="v3.0", raw="v3.0")

    def test_detects_v31_from_language_features(self) -> None:
        info = {
            "codebase_version": "v3.0",
            "features": {
                "language_persistent": {"dtype": "language", "shape": [1], "names": None},
                "action": {"dtype": "float32", "shape": [4], "names": None},
            },
        }
        result = detect_version(info)
        assert result == DatasetVersion(family="lerobot", version="v3.1", raw="v3.0")

    def test_rejects_unknown_v99(self) -> None:
        info = {"codebase_version": "v99.0"}
        with pytest.raises(UnsupportedVersionError, match="v99.0"):
            detect_version(info)

    def test_detects_legacy_v1x_for_read_only_document(self) -> None:
        info = {"codebase_version": "v1.6"}
        result = detect_version(info)
        assert result == DatasetVersion(family="lerobot", version="v1.6", raw="v1.6")

    def test_detects_v20_for_read_only_document(self) -> None:
        info = {"codebase_version": "v2.0"}
        result = detect_version(info)
        assert result == DatasetVersion(family="lerobot", version="v2.0", raw="v2.0")

    def test_rejects_missing_codebase_version(self) -> None:
        info = {"fps": 10}
        with pytest.raises(UnsupportedVersionError, match="missing"):
            detect_version(info)

    def test_version_is_frozen(self) -> None:
        v = DatasetVersion(family="lerobot", version="v2.1", raw="v2.1")
        with pytest.raises(AttributeError):
            v.version = "v3.0"  # type: ignore[misc]
