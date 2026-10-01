from __future__ import annotations

import time
from collections.abc import Callable
from copy import deepcopy
from typing import Any


ProgressCallback = Callable[[dict[str, Any]], None]


CHECK_DEFINITIONS = (
    ("metadata", "Metadata and dataset structure"),
    ("indices", "Episode and frame indices"),
    ("timestamps", "Timestamps and FPS consistency"),
    ("features", "Declared numeric features"),
    ("videos", "Video decoding and frame coverage"),
    ("statistics", "Dataset statistics"),
)


ISSUE_CHECKS = {
    "invalid_episode_total": "metadata",
    "episode_read_failed": "indices",
    "empty_episode": "indices",
    "missing_required_column": "indices",
    "episode_index_mismatch": "indices",
    "frame_index_mismatch": "indices",
    "global_index_mismatch": "indices",
    "numeric_data_invalid": "indices",
    "invalid_episode_offsets": "indices",
    "frame_total_mismatch": "indices",
    "episode_metadata_mismatch": "indices",
    "abnormal_episode_length": "indices",
    "timestamp_non_finite": "timestamps",
    "timestamp_regression": "timestamps",
    "estimated_missing_frames": "timestamps",
    "fps_jitter": "timestamps",
    "feature_shape_mismatch": "features",
    "feature_non_finite": "features",
    "constant_sensor": "features",
    "action_jump": "features",
    "video_decoder_unavailable": "videos",
    "video_frame_mismatch": "videos",
    "video_decode_failed": "videos",
    "stats_load_failed": "statistics",
    "official_statistics_provenance_invalid": "statistics",
    "official_visual_stats_difference": "statistics",
    "official_bookkeeping_stats_difference": "statistics",
}


class ValidationReporter:
    def __init__(self, callback: ProgressCallback | None) -> None:
        self._callback = callback
        self._started = time.monotonic()
        self._last_video_emit = 0.0
        self._last_statistics_emit = 0.0
        self.stage = "metadata"
        self.completed = 0
        self.total = 0
        self.decoded_frames = 0
        self.checks = [
            {
                "id": check_id,
                "label": label,
                "status": "pending",
                "failures": 0,
                "warnings": 0,
            }
            for check_id, label in CHECK_DEFINITIONS
        ]
        self._by_id = {check["id"]: check for check in self.checks}
        self.start("metadata")

    def snapshot(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "completed": self.completed,
            "total": self.total,
            "decoded_frames": self.decoded_frames,
            "elapsed_seconds": round(time.monotonic() - self._started, 3),
            "checks": deepcopy(self.checks),
        }

    def emit(self) -> None:
        if self._callback is None:
            return
        self._callback(self.snapshot())

    def configure_episodes(self, total: int) -> None:
        self.total = total
        self.emit()

    def set_stage(self, stage: str) -> None:
        self.stage = stage
        self.emit()

    def start(self, check_id: str, *, detail: str | None = None) -> None:
        check = self._by_id[check_id]
        if check["status"] == "pending":
            check["status"] = "running"
        if detail:
            check["detail"] = detail
        self.emit()

    def issue(self, severity: str, code: str, message: str) -> None:
        category = ISSUE_CHECKS.get(code)
        if category is None:
            if code.startswith(("stats_", "visual_stats_", "relative_")):
                category = "statistics"
            elif code.startswith("video_"):
                category = "videos"
            elif code.startswith("feature_"):
                category = "features"
            elif code.startswith("timestamp_"):
                category = "timestamps"
            elif code == "task_reference_invalid":
                category = "indices"
            else:
                category = "metadata"
        check = self._by_id[category]
        if severity == "FAIL":
            check["failures"] += 1
            check["status"] = "failed"
        else:
            check["warnings"] += 1
            if check["status"] != "failed":
                check["status"] = "warning"
        check["detail"] = message

    def finish(self, check_id: str, *, detail: str | None = None) -> None:
        check = self._by_id[check_id]
        if check["status"] in {"pending", "running"}:
            check["status"] = "passed"
        if detail:
            check["detail"] = detail

    def skip(self, check_id: str, detail: str) -> None:
        check = self._by_id[check_id]
        if check["status"] in {"pending", "running"}:
            check["status"] = "skipped"
            check["detail"] = detail

    def skip_pending(self, detail: str) -> None:
        for check in self.checks:
            if check["status"] == "pending":
                check["status"] = "skipped"
                check["detail"] = detail

    def episode_complete(self) -> None:
        self.completed += 1
        self.stage = "data"
        self.emit()

    def frame_decoded(self) -> None:
        self.decoded_frames += 1
        now = time.monotonic()
        if now - self._last_video_emit >= 1.0:
            self._last_video_emit = now
            self.stage = "video"
            self.emit()

    def statistics_progress(self, event: dict[str, Any]) -> None:
        now = time.monotonic()
        if (
            not event.get("_force")
            and now - self._last_statistics_emit < 1.0
            and event.get("completed") != event.get("total")
        ):
            return
        self._last_statistics_emit = now
        self.stage = "statistics"
        self.start(
            "statistics",
            detail=(
                f"{event.get('current_item', '통계')} · "
                f"{event.get('completed', 0):,} / {event.get('total', 0):,} "
                f"{event.get('unit', 'items')}"
            ),
        )

    def complete(self) -> list[dict[str, Any]]:
        self.stage = "complete"
        self.emit()
        return deepcopy(self.checks)
