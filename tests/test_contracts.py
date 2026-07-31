"""Tests for schema contracts and validation.

These tests verify that:
1. Contract JSON schemas are well-formed and loadable
2. Generated fixtures validate against their respective schemas
3. Corrupt fixtures are correctly rejected
4. Space parity features are covered
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lerobot_dataset_editor.schemas import (
    language_v31_schema,
    rpc_schema,
    v21_schema,
    v30_schema,
    validate_info_v21,
    validate_info_v30,
    validate_language_row,
)


class TestSchemaLoading:
    """Test that all contract schemas load without error."""

    def test_v21_schema_loads(self):
        schema = v21_schema()
        assert schema["$id"] == "urn:lerobot:dataset-editor:lerobot-v21"
        assert "properties" in schema

    def test_v30_schema_loads(self):
        schema = v30_schema()
        assert schema["$id"] == "urn:lerobot:dataset-editor:lerobot-v30"
        assert "properties" in schema

    def test_language_v31_schema_loads(self):
        schema = language_v31_schema()
        assert schema["$id"] == "urn:lerobot:dataset-editor:language-v31"
        assert "$defs" in schema

    def test_rpc_schema_loads(self):
        schema = rpc_schema()
        assert schema["$id"] == "urn:lerobot:dataset-editor:rpc"
        assert "properties" in schema


class TestV21Validation:
    """Test v2.1 schema validation."""

    def test_v21_fixture_info_schema(self, v21_fixture):
        """SP-14: v2.1 backward compat display."""
        info = json.loads((v21_fixture / "meta" / "info.json").read_text())
        errors = validate_info_v21(info)
        assert errors == [], f"v2.1 fixture failed validation: {errors}"

    def test_v21_rejects_missing_features(self):
        info = {"codebase_version": "2.1.0", "fps": 10}
        errors = validate_info_v21(info)
        assert len(errors) > 0

    def test_v21_accepts_official_image_only_null_video_path(self, v21_fixture):
        """Official lerobot==0.3.3 emits null when total_videos is zero."""
        info = json.loads((v21_fixture / "meta" / "info.json").read_text())
        info["video_path"] = None
        assert validate_info_v21(info) == []

    def test_v21_accepts_official_video_template(self, v21_fixture):
        """Video-backed v2.1 datasets retain the canonical template."""
        info = json.loads((v21_fixture / "meta" / "info.json").read_text())
        info["video_path"] = (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        )
        assert validate_info_v21(info) == []

    def test_v21_rejects_invalid_fps(self):
        info = {
            "codebase_version": "2.1.0",
            "fps": 0,
            "features": {},
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": 0,
            "splits": {},
        }
        errors = validate_info_v21(info)
        assert any("minimum" in e for e in errors)

    def test_v21_rejects_wrong_version(self):
        info = {
            "codebase_version": "3.0.0",
            "fps": 10,
            "features": {},
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": 0,
            "splits": {},
        }
        errors = validate_info_v21(info)
        assert len(errors) > 0
        assert any("v2.1" in e or "3.0.0" in e for e in errors)


class TestV30Validation:
    """Test v3.0 schema validation."""

    def test_v30_fixture_info_schema(self, v30_fixture):
        """SP-01: info.json schema conformance."""
        info = json.loads((v30_fixture / "meta" / "info.json").read_text())
        errors = validate_info_v30(info)
        assert errors == [], f"v3.0 fixture failed validation: {errors}"

    def test_v30_rejects_missing_data_path(self):
        info = {
            "codebase_version": "3.0.0",
            "fps": 10,
            "features": {},
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": 0,
            "chunks_size": 1000,
            "splits": {},
        }
        errors = validate_info_v30(info)
        assert any("data_path" in e for e in errors)

    def test_v30_accepts_official_video_template(self, v30_fixture):
        info = json.loads((v30_fixture / "meta" / "info.json").read_text())
        info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        assert validate_info_v30(info) == []

    def test_v30_multi_camera_features(self, v30_fixture):
        """SP-06: multi-camera feature keys."""
        info = json.loads((v30_fixture / "meta" / "info.json").read_text())
        camera_features = [
            k for k, v in info["features"].items() if v["dtype"] in {"image", "video"}
        ]
        assert len(camera_features) >= 2, "v3.0 fixture should have at least 2 camera features"
        assert "observation.images.front" in camera_features
        assert "observation.images.wrist" in camera_features


class TestLanguageValidation:
    """Test language annotation schema validation."""

    def test_valid_persistent_row(self):
        row = {
            "role": "assistant",
            "content": "Pick up the red cube",
            "style": "subtask",
            "timestamp": 1.5,
            "camera": None,
            "tool_calls": None,
        }
        errors = validate_language_row(row, "persistent")
        assert errors == []

    def test_valid_event_row(self):
        row = {
            "role": "assistant",
            "content": "Grasping now",
            "style": "interjection",
            "camera": None,
            "tool_calls": None,
        }
        errors = validate_language_row(row, "event")
        assert errors == []

    def test_invalid_style_rejected(self):
        row = {
            "role": "assistant",
            "content": "test",
            "style": "invalid_style",
        }
        errors = validate_language_row(row, "persistent")
        assert len(errors) > 0

    def test_missing_content_rejected(self):
        row = {"role": "assistant", "style": "subtask"}
        errors = validate_language_row(row, "persistent")
        assert len(errors) > 0


class TestCorruptFixture:
    """Test that corrupt fixtures are properly detected."""

    def test_corrupt_fixture_detection(self, corrupt_fixture):
        """SP-15: corrupt dataset error handling."""
        info = json.loads((corrupt_fixture / "meta" / "info.json").read_text())
        # Should fail both v2.1 and v3.0 validation
        v21_errors = validate_info_v21(info)
        v30_errors = validate_info_v30(info)
        assert len(v21_errors) > 0, "Corrupt fixture should fail v2.1 validation"
        assert len(v30_errors) > 0, "Corrupt fixture should fail v3.0 validation"

    def test_corrupt_parquet_is_not_valid(self, corrupt_fixture):
        """The corrupt parquet file should not be loadable."""
        import pyarrow.parquet as pq

        corrupt_pq = corrupt_fixture / "data" / "chunk-000" / "file-000.parquet"
        assert corrupt_pq.exists()
        with pytest.raises(Exception):
            pq.read_table(corrupt_pq)
