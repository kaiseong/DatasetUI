import copy
from types import SimpleNamespace

import pytest

from datasetui.merge.job import MergeCompatibilityError, _compatible_info
from datasetui.official.operations import provenance, write_official_merge
from datasetui.transform_errors import CurationTransformError




def test_incompatible_merge_fails_before_any_private_copy(tmp_path, monkeypatch):
    from datasetui.official import sources as processing_sources
    from datasetui.merge import normalization as merge_normalization

    root = tmp_path / "original"
    root.mkdir()

    def reject(*args, **kwargs):
        raise CurationTransformError("action: expected length 16, got 15")

    def forbidden(*args, **kwargs):
        pytest.fail("private copy started before compatibility check")

    monkeypatch.setattr(merge_normalization, "plan_merge_normalization", reject)
    monkeypatch.setattr(processing_sources, "private_sources", forbidden)
    with pytest.raises(CurationTransformError, match="expected length 16"):
        write_official_merge(
            sources=[SimpleNamespace(root=root)],
            destination=tmp_path / "out",
            robot_type="rby1",
        )


def test_timestamp_metadata_widening_only_for_official_merge():
    a = {
        "codebase_version": "v3.0",
        "fps": 30,
        "features": {
            "timestamp": {"dtype": "float32", "shape": [1]},
            "action": {"dtype": "float32", "shape": [16]},
        },
    }
    b = copy.deepcopy(a)
    b["features"]["timestamp"]["dtype"] = "float64"
    with pytest.raises(MergeCompatibilityError):
        _compatible_info([a, b])
    _compatible_info([a, b], normalize_timestamp=True)
    assert a["features"]["timestamp"]["dtype"] == "float32"
    b["features"]["action"]["dtype"] = "float64"
    with pytest.raises(MergeCompatibilityError):
        _compatible_info([a, b], normalize_timestamp=True)


def test_merge_cache_policy_changes_without_affecting_subset():
    assert (
        provenance("merge_datasets")["schema_policy"]
        == "lossless-merge-list-f32-timestamp-f64-v2"
    )
    assert provenance("split_dataset")["schema_policy"] == "source-arrow-types-v1"
