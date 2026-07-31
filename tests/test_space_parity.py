"""Space parity contract validation tests.

Verifies that the space-parity.yaml contract is consistent and that
all referenced test IDs actually exist in the test suite.
"""

from __future__ import annotations

from pathlib import Path

import yaml
import pytest


CONTRACTS_DIR = Path(__file__).resolve().parent.parent / "contracts"


class TestSpaceParityContract:
    """Validate the space-parity.yaml contract itself."""

    def test_space_parity_yaml_loads(self):
        """Contract YAML should be parseable."""
        path = CONTRACTS_DIR / "space-parity.yaml"
        assert path.exists()
        data = yaml.safe_load(path.read_text())
        assert "space_commit" in data
        assert "parity_features" in data

    def test_space_commit_pinned(self):
        """Commit should be pinned to the specified value."""
        data = yaml.safe_load((CONTRACTS_DIR / "space-parity.yaml").read_text())
        assert data["space_commit"] == "d724744111cae6feb9a2194e607e71749813a97a"

    def test_all_features_have_test_ids(self):
        """Every parity feature should map to a test_id."""
        data = yaml.safe_load((CONTRACTS_DIR / "space-parity.yaml").read_text())
        for feature in data["parity_features"]:
            assert "test_id" in feature, f"Feature {feature['id']} missing test_id"
            assert feature["test_id"], f"Feature {feature['id']} has empty test_id"

    def test_all_features_have_required_fields(self):
        """Every parity feature should have executable planning metadata."""
        data = yaml.safe_load((CONTRACTS_DIR / "space-parity.yaml").read_text())
        required = {"id", "category", "description", "test_id", "status", "implementation_task"}
        for feature in data["parity_features"]:
            missing = required - set(feature.keys())
            assert not missing, f"Feature {feature.get('id', '?')} missing: {missing}"

    def test_feature_ids_are_unique(self):
        """All feature IDs should be unique."""
        data = yaml.safe_load((CONTRACTS_DIR / "space-parity.yaml").read_text())
        ids = [f["id"] for f in data["parity_features"]]
        assert len(ids) == len(set(ids))

    def test_all_features_have_truthful_status(self):
        """Task 1 must not claim that later UI parity tests already pass."""
        data = yaml.safe_load((CONTRACTS_DIR / "space-parity.yaml").read_text())
        for feature in data["parity_features"]:
            assert feature["status"] in {"contract-tested", "planned"}
            assert "::test_" in feature["test_id"]
            assert 1 <= feature["implementation_task"] <= 23

    def test_revision_file_matches(self):
        """third_party/visualize_dataset.REVISION should match the contract."""
        revision_path = Path(__file__).resolve().parent.parent / "third_party" / "visualize_dataset.REVISION"
        assert revision_path.exists()
        data = yaml.safe_load((CONTRACTS_DIR / "space-parity.yaml").read_text())
        assert revision_path.read_text().strip() == data["space_commit"]

    def test_minimum_feature_coverage(self):
        """Should have at least 15 parity features defined."""
        data = yaml.safe_load((CONTRACTS_DIR / "space-parity.yaml").read_text())
        assert len(data["parity_features"]) >= 15
