import json
from types import SimpleNamespace

import pandas as pd
import pytest

from datasetui.transform_errors import CurationTransformError
from datasetui.transforms import (
    _curation_processing,
    _remap_tasks,
    CURATION_PROCESSING_POLICY,
    reuse_published_outputs,
)



@pytest.mark.parametrize("change", ["engine", "path", "snapshot", "duplicate"])
def test_cache_never_accepts_wrong_engine_path_or_snapshot(tmp_path, change):
    settings = SimpleNamespace(nas_root=tmp_path)
    path = tmp_path / "manifests/curation/job.json"
    path.parent.mkdir(parents=True)
    processing = {"engine": "verified"}
    snapshot = {"id": "snapshot", "dataset_id": "source", "dataset_fingerprint": "hash"}
    result = {
        "video_codec_policy": "source",
        "processing_policy": CURATION_PROCESSING_POLICY,
        "snapshot_id": "snapshot",
        "source_dataset_id": "source",
        "source_fingerprint": "hash",
        "outputs": [{"name": "out", "relative_path": "out", "processing": processing}],
    }
    if change == "engine":
        result["outputs"][0]["processing"] = {"engine": "legacy"}
    elif change == "path":
        result["outputs"][0]["relative_path"] = "../original"
    elif change == "snapshot":
        result["snapshot_id"] = "other"
    else:
        result["outputs"] *= 2
    path.write_text(json.dumps(result))
    with pytest.raises(CurationTransformError):
        reuse_published_outputs(
            settings=settings,
            job_id="job",
            outputs=[{"name": "out"}],
            processing=processing,
            snapshot=snapshot,
        )


def test_routing_identity_distinguishes_official_subset_and_custom_trim(monkeypatch):
    monkeypatch.setenv("DATASETUI_PROCESSOR_ENGINE", "lerobot-v3")
    official = _curation_processing("v3.0", {}, {}, {})
    custom = _curation_processing("v3.0", {"enabled": True}, {}, {})
    assert official["official_function"] == "split_dataset"
    assert "official_function" not in custom
    assert official != custom
    assert official["statistics_policy"] == "lerobot-official-aggregate-v1"
    assert custom["statistics_policy"] == "exact-global-numeric-sampled-rgb-v1"


def test_missing_task_is_not_replaced_by_invented_text():
    with pytest.raises(CurationTransformError, match="7"):
        _remap_tasks([(pd.DataFrame({"task_index": [7]}), {}, 0, 1)], {})
