from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from datasetui.config import Settings
from datasetui.database import Database, RecipeRevisionMismatchError
from datasetui.transform_errors import CurationTransformError
from datasetui.transforms import (
    MAX_INFO_BYTES,
    _DatasetSource,
    _read_regular_bytes,
    _safe_dataset_root,
)
from datasetui.validation_reporting import ProgressCallback, ValidationReporter
from datasetui.validation_structure import (
    validate_info,
    validate_metadata,
    validate_episode_structure,
    validate_features,
)
from datasetui.validation_statistics import NumericStatisticsValidator
from datasetui.validation_video import VideoValidator
from datasetui.validation_integrity import VALIDATOR_POLICY, validation_content_manifest

from datasetui.content_integrity import dataset_content_fingerprint


class DatasetValidationError(CurationTransformError):
    pass


def validate_registered_dataset(
    *,
    database: Database,
    settings: Settings,
    payload: dict[str, Any],
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    record = database.get_dataset(payload["dataset_id"])
    if (
        not record["available"]
        or record["readiness"] != "ready"
        or record["fingerprint"] != payload["fingerprint"]
        or record["storage_area"] != payload["storage_area"]
        or record["relative_path"] != payload["relative_path"]
    ):
        raise RecipeRevisionMismatchError(payload["dataset_id"])
    root = _safe_dataset_root(
        settings.nas_root, record["storage_area"], record["relative_path"]
    )
    return validate_dataset_root(
        root,
        mode=payload["mode"],
        fingerprint=payload["fingerprint"],
        on_progress=on_progress,
    )


def validate_dataset_root(
    root: Path,
    *,
    mode: str,
    fingerprint: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    if mode not in {"quick", "full", "export_gate"}:
        raise DatasetValidationError("Unknown validation mode")
    issues: list[dict[str, Any]] = []
    reporter = ValidationReporter(on_progress)
    total_episodes = 0
    total_frames = 0
    indices: list[int] = []
    content_manifest = None
    statistics_deferred = False

    def issue(
        severity: str, code: str, message: str, episode: int | None = None
    ) -> None:
        item: dict[str, Any] = {"severity": severity, "code": code, "message": message}
        if episode is not None:
            item["episode_index"] = episode
        issues.append(item)
        reporter.issue(severity, code, message)

    actual_fingerprint = fingerprint or ""

    def result() -> dict[str, Any]:
        failures = sum(item["severity"] == "FAIL" for item in issues)
        return {
            "mode": mode,
            "passed": failures == 0,
            "fingerprint": actual_fingerprint,
            "checked_episodes": reporter.completed,
            "total_episodes": total_episodes,
            "checked_frames": total_frames,
            "failures": failures,
            "warnings": sum(item["severity"] == "WARN" for item in issues),
            "issues": issues,
            "checks": reporter.complete(),
            "loader_probe": "datasetui-parquet-video-reader",
            "validator_policy": VALIDATOR_POLICY,
            "content_manifest": content_manifest,
            "statistics_deferred": statistics_deferred,
        }

    try:
        raw = _read_regular_bytes(root / "meta/info.json", max_bytes=MAX_INFO_BYTES)
        actual_fingerprint = dataset_content_fingerprint(root)
    except (OSError, ValueError, CurationTransformError):
        issue(
            "FAIL",
            "metadata_load_failed",
            "Dataset metadata/content could not be read safely",
        )
        reporter.skip_pending("Metadata validation did not complete")
        return result()
    if fingerprint is not None and actual_fingerprint != fingerprint:
        issue("FAIL", "metadata_changed", "Dataset content changed before validation")
        reporter.skip_pending("Dataset revision changed")
        reporter.complete()
        raise RecipeRevisionMismatchError(actual_fingerprint)
    try:
        info = json.loads(raw)
    except (ValueError, UnicodeError):
        info = None
    if not validate_info(info, issue):
        reporter.skip_pending("Metadata schema must be repaired first")
        return result()
    total_episodes = info["total_episodes"]
    try:
        source = _DatasetSource(root, info)
        validate_metadata(source, issue)
    except (OSError, ValueError, TypeError, KeyError, CurationTransformError):
        issue(
            "FAIL", "metadata_load_failed", "Task/episode metadata could not be loaded"
        )
        reporter.skip_pending("Metadata loading did not complete")
        return result()
    try:
        from datasetui.deferred_statistics import read_deferred_statistics

        statistics_deferred = read_deferred_statistics(root)
    except (OSError, ValueError, CurationTransformError):
        issue("FAIL", "stats_deferred_invalid", "분포 통계 미재계산 표시가 메타데이터와 일치하지 않습니다")
        reporter.skip_pending("통계 상태 메타데이터를 확인할 수 없습니다")
        return result()
    if mode == "export_gate":
        try:
            content_manifest = validation_content_manifest(root)
        except (OSError, ValueError, CurationTransformError):
            issue(
                "FAIL",
                "content_changed",
                "Could not capture dataset contents before validation",
            )
            reporter.skip_pending("Content snapshot unavailable")
            return result()

    indices = (
        [0, total_episodes - 1]
        if mode == "quick" and total_episodes > 2
        else list(range(total_episodes))
    )
    sample_detail = f"검사 대상 {len(indices)} / 전체 {total_episodes} 에피소드"
    reporter.finish("metadata")
    reporter.configure_episodes(len(indices))
    for check in ("indices", "timestamps", "features"):
        reporter.start(check, detail=sample_detail)
    if mode == "quick":
        reporter.skip("videos", "빠른 검사에서는 영상을 검사하지 않습니다")
        reporter.skip("statistics", "빠른 검사에서는 통계를 검사하지 않습니다")
    elif source.video_keys:
        reporter.start("videos")
    else:
        reporter.skip("videos", "영상 feature가 없는 데이터셋입니다")

    statistics = NumericStatisticsValidator(info, issue)
    videos = VideoValidator(source, info, issue, reporter)
    lengths = []
    successful_episodes = 0
    timestamp_episodes = 0
    for episode_index in indices:
        reporter.set_stage("data")
        try:
            data, metadata = source.episode(episode_index)
        except (OSError, ValueError, KeyError, TypeError, CurationTransformError):
            issue(
                "FAIL",
                "episode_read_failed",
                "Episode data could not be read",
                episode_index,
            )
            reporter.episode_complete()
            continue
        length = len(data)
        if length < 1:
            issue("FAIL", "empty_episode", "Episode contains no frames", episode_index)
            reporter.episode_complete()
            continue
        successful_episodes += 1
        lengths.append(length)
        timestamps = validate_episode_structure(
            source,
            data,
            metadata,
            episode_index,
            total_frames,
            full=mode != "quick",
            issue=issue,
        )
        timestamp_episodes += timestamps is not None
        validate_features(info, data, episode_index, issue)
        total_frames += length
        if mode != "quick":
            statistics.add_episode(data, episode_index=episode_index)
            if source.video_keys:
                reporter.set_stage("video")
                videos.validate_episode(
                    episode_index, metadata, length, timestamps=timestamps
                )
        reporter.episode_complete()

    all_read = successful_episodes == len(indices)
    for check in ("indices", "features"):
        if all_read:
            reporter.finish(check, detail=sample_detail)
        else:
            reporter.skip(
                check, "읽지 못한 에피소드가 있어 전체 확인을 완료하지 못했습니다"
            )
    if all_read and timestamp_episodes == len(indices):
        reporter.finish("timestamps", detail=sample_detail)
    else:
        reporter.skip("timestamps", "일부 에피소드의 시간을 확인하지 못했습니다")
    if mode != "quick" and source.video_keys:
        if all_read:
            reporter.finish("videos")
        else:
            reporter.skip("videos", "일부 에피소드의 영상을 확인하지 못했습니다")

    if mode != "quick":
        if total_frames != info["total_frames"]:
            issue(
                "FAIL",
                "frame_total_mismatch",
                "Frame total differs from meta/info.json",
            )
        if lengths and max(lengths) > max(2, int(np.median(lengths) * 5)):
            issue(
                "WARN",
                "abnormal_episode_length",
                "An episode is much longer than the median",
            )
        if statistics_deferred:
            issue("WARN", "stats_deferred", "분포 통계 미재계산: 기존 stats는 현재 출력의 통계가 아닙니다. 학습 전 최종 데이터로 norm_stats를 계산하세요.")
            reporter.skip("statistics", "사용자 선택으로 분포 통계 비교 생략 · 구조·영상 검사는 수행")
        else:
            reporter.set_stage("statistics")
            reporter.start("statistics")
            summary = statistics.validate(
                root,
                complete_dataset=all_read,
                visual_statistics=videos.visual_statistics,
                on_progress=reporter.statistics_progress,
            )
            if mode == "export_gate" and summary.unverified_visual_features:
                issue(
                    "FAIL",
                    "stats_unverified_visual",
                    "Visual statistics were not reproducibly verified; export gate is blocked",
                )
            reporter.finish(
                "statistics",
                detail=f"수치 통계 비교: {len(summary.validated_features)} features",
            )
    if mode == "export_gate" and content_manifest is not None:
        reporter.set_stage("metadata")
        try:
            if validation_content_manifest(root) != content_manifest:
                issue(
                    "FAIL",
                    "content_changed",
                    "Dataset contents changed during validation",
                )
                content_manifest = None
        except (OSError, ValueError, CurationTransformError):
            issue(
                "FAIL",
                "content_changed",
                "Dataset contents could not be verified after validation",
            )
            content_manifest = None
    return result()
