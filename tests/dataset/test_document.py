"""Tests for DatasetDocument frozen model and loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from lerobot_dataset_editor.dataset.document import (
    DatasetDocument,
    DatasetVersion,
    EpisodeRef,
    FeatureSpec,
    MediaRef,
    ValidationResult,
)
from lerobot_dataset_editor.dataset.loader import load_document


class TestDatasetDocumentModel:
    """DatasetDocument is a frozen, immutable dataclass."""

    def test_document_is_frozen(self) -> None:
        doc = DatasetDocument(
            source_path="/tmp/test",
            version=DatasetVersion(family="lerobot", version="v2.1", raw="v2.1"),
            features=(FeatureSpec(name="action", dtype="float32", shape=(4,), names=("a0", "a1", "a2", "a3")),),
            episodes=(EpisodeRef(index=0, length=10, chunk_index=0, file_index=0, tasks=("pick",), frame_start=0, frame_end=10),),
            total_frames=10,
            fps=10,
            media=(),
            tasks=("pick",),
            has_annotations=False,
            annotation_styles=None,
            provenance=None,
            validation=ValidationResult(valid=True, errors=(), warnings=()),
            indexed_at="2024-01-01T00:00:00Z",
        )
        with pytest.raises(AttributeError):
            doc.fps = 30  # type: ignore[misc]

    def test_version_is_frozen(self) -> None:
        v = DatasetVersion(family="lerobot", version="v3.0", raw="v3.0")
        with pytest.raises(AttributeError):
            v.family = "other"  # type: ignore[misc]

    def test_episode_ref_is_frozen(self) -> None:
        ep = EpisodeRef(index=0, length=10, chunk_index=0, file_index=0, tasks=("pick",), frame_start=0, frame_end=10)
        with pytest.raises(AttributeError):
            ep.index = 1  # type: ignore[misc]


class TestLoadDocument:
    """Test loading DatasetDocument from fixture directories."""

    def test_document_from_v21_fixture(self, v21_fixture: Path) -> None:
        doc = load_document(v21_fixture)
        assert doc.version.version == "v2.1"
        assert doc.version.family == "lerobot"
        assert doc.total_frames == 20
        assert doc.fps == 10
        assert len(doc.episodes) == 2
        assert doc.episodes[0].index == 0
        assert doc.episodes[0].length == 10
        assert doc.episodes[1].index == 1
        assert doc.episodes[1].length == 10
        assert doc.has_annotations is False
        assert doc.annotation_styles is None

    def test_document_from_v30_fixture(self, v30_fixture: Path) -> None:
        doc = load_document(v30_fixture)
        assert doc.version.version == "v3.0"
        assert doc.total_frames == 30
        assert doc.fps == 10
        assert len(doc.episodes) == 3
        assert doc.episodes[0].length == 10
        assert doc.episodes[1].length == 10
        assert doc.episodes[2].length == 10

    def test_document_from_v30_annotated_fixture(self, v30_annotated_fixture: Path) -> None:
        doc = load_document(v30_annotated_fixture)
        assert doc.version.version == "v3.1"
        assert doc.has_annotations is True
        assert doc.annotation_styles is not None
        assert "language_persistent" in doc.annotation_styles
        assert "language_events" in doc.annotation_styles

    def test_document_features_match_info_json(self, v21_fixture: Path) -> None:
        doc = load_document(v21_fixture)
        feature_names = {f.name for f in doc.features}
        # Must include core structural features from info.json
        assert "action" in feature_names
        assert "observation.state" in feature_names
        assert "timestamp" in feature_names

    def test_document_episodes_from_v21_jsonl(self, v21_fixture: Path) -> None:
        doc = load_document(v21_fixture)
        # v2.1 episodes come from episodes.jsonl
        assert len(doc.episodes) == 2
        for ep in doc.episodes:
            assert isinstance(ep, EpisodeRef)
            assert ep.tasks == ("pick up object",)

    def test_document_episodes_from_v30_parquet(self, v30_fixture: Path) -> None:
        doc = load_document(v30_fixture)
        # v3.0 episodes come from meta/episodes Parquet
        assert len(doc.episodes) == 3
        for ep in doc.episodes:
            assert isinstance(ep, EpisodeRef)
            assert ep.length == 10

    def test_document_media_refs_null_video(self, v21_fixture: Path) -> None:
        doc = load_document(v21_fixture)
        # video_path is None in the v21_valid fixture
        assert doc.media == ()

    def test_document_provenance_from_file(self, v21_fixture: Path) -> None:
        doc = load_document(v21_fixture)
        assert doc.provenance is not None
        assert "CONTRACT-DERIVED" in doc.provenance

    def test_document_source_path_is_canonical(self, v21_fixture: Path) -> None:
        doc = load_document(v21_fixture)
        assert doc.source_path == str(v21_fixture.resolve())

    def test_document_indexed_at_is_iso(self, v21_fixture: Path) -> None:
        doc = load_document(v21_fixture)
        # Just verify it's a non-empty string looking like ISO
        assert "T" in doc.indexed_at

    def test_document_exposes_lazy_global_frame_ranges(self, v21_fixture: Path) -> None:
        doc = load_document(v21_fixture)
        assert [(episode.frame_start, episode.frame_end) for episode in doc.episodes] == [
            (0, 10),
            (10, 20),
        ]
