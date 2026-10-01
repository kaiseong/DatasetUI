from __future__ import annotations

from pathlib import Path

import pytest
import pandas as pd

from datasetui.validation import validate_dataset_root
from test_transforms import _write_v21
from test_validation_conversion import _add_shared_v3_video, _write_v3


def _checks_by_id(result):
    return {check["id"]: check for check in result["checks"]}


def test_official_aggregate_issues_belong_to_statistics_not_metadata():
    from datasetui.validation_reporting import ValidationReporter

    reporter = ValidationReporter(None)
    reporter.issue("WARN", "official_visual_stats_difference", "sampled images")
    reporter.issue("WARN", "official_bookkeeping_stats_difference", "reindexing")
    reporter.issue("FAIL", "official_statistics_provenance_invalid", "changed file")
    checks = _checks_by_id(reporter.snapshot())
    assert checks["statistics"]["warnings"] == 2
    assert checks["statistics"]["failures"] == 1
    assert checks["metadata"]["warnings"] == checks["metadata"]["failures"] == 0


def test_quick_validation_reports_sampled_checks_and_progress(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_v21(root)
    updates = []

    result = validate_dataset_root(root, mode="quick", on_progress=updates.append)

    checks = _checks_by_id(result)
    assert checks["metadata"]["status"] == "passed"
    assert checks["indices"]["status"] == "passed"
    assert checks["timestamps"]["status"] == "passed"
    assert checks["features"]["status"] == "passed"
    assert checks["videos"] == {
        "id": "videos",
        "label": "Video decoding and frame coverage",
        "status": "skipped",
        "failures": 0,
        "warnings": 0,
        "detail": "빠른 검사에서는 영상을 검사하지 않습니다",
    }
    assert checks["statistics"]["status"] == "skipped"
    assert updates[-1]["stage"] == "complete"
    assert updates[-1]["completed"] == updates[-1]["total"] == 2


def test_full_validation_reports_actual_decoded_frames(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_v3(root)
    _add_shared_v3_video(root)
    updates = []

    result = validate_dataset_root(root, mode="full", on_progress=updates.append)

    checks = _checks_by_id(result)
    assert result["passed"] is True
    assert checks["videos"]["status"] == "passed"
    assert checks["statistics"]["status"] == "passed"
    assert updates[-1]["decoded_frames"] == 8
    assert any(update["stage"] == "video" for update in updates)
    assert updates[-1]["stage"] == "complete"


def test_read_failure_keeps_unfinished_checks_from_appearing_green(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    _write_v21(root)
    (root / "data/chunk-000/episode_000001.parquet").unlink()
    updates = []

    result = validate_dataset_root(root, mode="quick", on_progress=updates.append)

    checks = _checks_by_id(result)
    assert result["passed"] is False
    assert checks["indices"]["status"] == "failed"
    assert checks["indices"]["failures"] == 1
    assert checks["timestamps"]["status"] == "skipped"
    assert checks["features"]["status"] == "skipped"
    assert updates[-1]["completed"] == updates[-1]["total"] == 2


def test_empty_episodes_do_not_mark_unexecuted_checks_as_passed(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_v21(root)
    for path in (root / "data/chunk-000").glob("*.parquet"):
        pd.read_parquet(path).iloc[0:0].to_parquet(path, index=False)

    result = validate_dataset_root(root, mode="quick")

    checks = _checks_by_id(result)
    assert checks["indices"]["status"] == "failed"
    assert checks["timestamps"]["status"] == "skipped"
    assert checks["features"]["status"] == "skipped"


def test_progress_callback_failure_propagates_without_becoming_a_video_issue(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    _write_v3(root)
    _add_shared_v3_video(root)

    def broken_callback(progress):
        if progress["stage"] == "video" and progress["decoded_frames"] > 0:
            raise RuntimeError("UI reporting is unavailable")

    with pytest.raises(RuntimeError, match="UI reporting is unavailable"):
        validate_dataset_root(root, mode="full", on_progress=broken_callback)
