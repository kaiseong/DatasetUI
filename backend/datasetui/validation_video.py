from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from datasetui.validation_statistics import RunningMoments


IssueCallback = Callable[[str, str, str, int | None], None]


class _Source(Protocol):
    fps: float
    version: str
    video_keys: list[str]
    episode_metadata: dict[int, dict[str, Any]]

    def video_source(
        self, episode_index: int, video_key: str, metadata: dict[str, Any]
    ) -> tuple[Path, int]: ...


class _Reporter(Protocol):
    def frame_decoded(self) -> None: ...


class _DecodeFailure(Exception):
    pass


@dataclass(frozen=True)
class DecodedVideo:
    frame_count: int
    width: int
    height: int
    channels: int
    dimensions_consistent: bool
    stream_fps: float | None
    pts_seconds: tuple[float, ...]


class VideoValidator:
    def __init__(
        self,
        source: _Source,
        info: dict[str, Any],
        issue: IssueCallback,
        reporter: _Reporter,
    ) -> None:
        self._source = source
        self._features = info.get("features", {})
        self._issue = issue
        self._reporter = reporter
        self._cache: dict[Path, DecodedVideo | None] = {}
        self._visual_running: dict[str, RunningMoments] = {}
        self._visual_frame_counts: dict[str, int] = {}

    @property
    def decoded_paths(self) -> tuple[Path, ...]:
        return tuple(self._cache)

    @property
    def visual_statistics(self) -> dict[str, dict[str, np.ndarray]]:
        result: dict[str, dict[str, np.ndarray]] = {}
        for key, running in self._visual_running.items():
            values = running.values()
            values["_legacy_pixel_count"] = np.asarray([running.count], dtype=np.int64)
            values["count"] = np.asarray(
                [self._visual_frame_counts[key]], dtype=np.int64
            )
            result[key] = values
        return result

    def validate_episode(
        self,
        episode_index: int,
        metadata: dict[str, Any],
        expected_frames: int,
        *,
        timestamps: np.ndarray | None = None,
    ) -> None:
        for key in self._source.video_keys:
            try:
                path, start = self._source.video_source(episode_index, key, metadata)
                if start < 0 or path.is_symlink() or not path.is_file():
                    raise OSError
            except Exception:
                self._issue(
                    "FAIL",
                    "video_source_invalid",
                    f"Video source is missing or unsafe: {key}",
                    episode_index,
                )
                continue
            if path not in self._cache:
                selected = self._selected_frames(
                    key, path, expected_frames=expected_frames
                )
                self._cache[path] = self._decode(path, key, selected)
            decoded = self._cache[path]
            if decoded is None:
                self._issue(
                    "FAIL",
                    "video_decode_failed",
                    f"Video could not be decoded: {key}",
                    episode_index,
                )
                continue
            self._validate_format(key, decoded, episode_index)
            self._validate_fps(key, decoded, episode_index)
            if self._source.version == "v3.0":
                self._validate_v3_segment(
                    key,
                    decoded,
                    metadata,
                    expected_frames,
                    start,
                    episode_index,
                    timestamps,
                )
            elif decoded.frame_count != expected_frames:
                self._issue(
                    "FAIL",
                    "video_frame_mismatch",
                    f"Video frame count differs from episode data: {key}",
                    episode_index,
                )

    def _decode(
        self, path: Path, key: str, selected_frames: set[int] | None
    ) -> DecodedVideo | None:
        try:
            import av

            container = av.open(str(path))
        except Exception:
            return None
        try:
            stream = container.streams.video[0]
            rate = _positive_float(stream.average_rate)
            time_base = _positive_float(stream.time_base)
            frames = iter(container.decode(video=0))
            count = 0
            width = height = channels = 0
            dimensions_consistent = True
            pts: list[float] = []
            while True:
                try:
                    frame = next(frames)
                except StopIteration:
                    break
                except Exception as exc:
                    raise _DecodeFailure from exc
                if count == 0:
                    try:
                        rgb = frame.to_ndarray(format="rgb24")
                    except Exception as exc:
                        raise _DecodeFailure from exc
                    height, width = rgb.shape[:2]
                    channels = rgb.shape[2] if rgb.ndim == 3 else 1
                elif frame.width != width or frame.height != height:
                    dimensions_consistent = False
                elif selected_frames is not None and count in selected_frames:
                    try:
                        rgb = frame.to_ndarray(format="rgb24")
                    except Exception as exc:
                        raise _DecodeFailure from exc
                if selected_frames is not None and count in selected_frames:
                    self._add_visual_frame(key, rgb)
                if frame.pts is not None and time_base is not None:
                    pts.append(float(frame.pts) * time_base)
                count += 1
                # Lease/progress exceptions intentionally propagate unchanged.
                self._reporter.frame_decoded()
            if count == 0:
                raise _DecodeFailure
            return DecodedVideo(
                count,
                width,
                height,
                channels,
                dimensions_consistent,
                rate,
                tuple(pts),
            )
        except _DecodeFailure:
            return None
        finally:
            try:
                container.close()
            except Exception:
                pass

    def _selected_frames(
        self, key: str, path: Path, *, expected_frames: int
    ) -> set[int] | None:
        if self._source.version != "v3.0":
            return set(_sample_indices(expected_frames))
        selected: set[int] = set()
        try:
            metadata_items = self._source.episode_metadata.items()
        except AttributeError:
            return None
        try:
            for episode_index, metadata in metadata_items:
                candidate, start = self._source.video_source(
                    episode_index, key, metadata
                )
                if candidate != path:
                    continue
                length = int(metadata["dataset_to_index"]) - int(
                    metadata["dataset_from_index"]
                )
                if length < 1 or start < 0:
                    return None
                selected.update(start + index for index in _sample_indices(length))
        except (KeyError, TypeError, ValueError, OSError):
            return None
        return selected or None

    def _add_visual_frame(self, key: str, rgb: np.ndarray) -> None:
        descriptor = self._features.get(key, {})
        depth_map = bool(
            isinstance(descriptor, dict)
            and (
                descriptor.get("video.is_depth_map")
                or (
                    isinstance(descriptor.get("info"), dict)
                    and (
                        descriptor["info"].get("is_depth_map")
                        or descriptor["info"].get("video.is_depth_map")
                    )
                )
                or (
                    isinstance(descriptor.get("video_info"), dict)
                    and descriptor["video_info"].get("video.is_depth_map")
                )
            )
        )
        if depth_map:
            # rgb24 decoding does not preserve a depth map's native units.
            return
        height, width = rgb.shape[:2]
        if max(height, width) >= 300:
            factor = int(max(height, width) / 150)
            rgb = rgb[::factor, ::factor]
        values = rgb.reshape(-1, rgb.shape[2]).astype(np.float64)
        values /= 255.0
        self._visual_running.setdefault(key, RunningMoments()).update(values)
        self._visual_frame_counts[key] = self._visual_frame_counts.get(key, 0) + 1

    def _validate_format(
        self, key: str, decoded: DecodedVideo, episode_index: int
    ) -> None:
        descriptor = self._features.get(key, {})
        shape = descriptor.get("shape") if isinstance(descriptor, dict) else None
        if not (
            isinstance(shape, list)
            and len(shape) == 3
            and all(
                isinstance(item, int) and not isinstance(item, bool) for item in shape
            )
        ):
            self._issue(
                "FAIL",
                "video_shape_metadata_invalid",
                f"Video metadata shape is invalid: {key}",
                episode_index,
            )
            return
        expected = tuple(shape)
        actual = (decoded.height, decoded.width, decoded.channels)
        if expected != actual or not decoded.dimensions_consistent:
            self._issue(
                "FAIL",
                "video_shape_mismatch",
                f"Decoded video shape {actual} differs from metadata {expected}: {key}",
                episode_index,
            )

    def _validate_fps(
        self, key: str, decoded: DecodedVideo, episode_index: int
    ) -> None:
        fps = self._source.fps
        if not math.isfinite(fps) or fps <= 0:
            return
        if decoded.stream_fps is None or not math.isclose(
            decoded.stream_fps, fps, rel_tol=0.01, abs_tol=0.01
        ):
            self._issue(
                "FAIL",
                "video_fps_mismatch",
                f"Video stream FPS differs from dataset FPS: {key}",
                episode_index,
            )
        pts = np.asarray(decoded.pts_seconds, dtype=np.float64)
        if len(pts) != decoded.frame_count:
            self._issue(
                "FAIL",
                "video_pts_missing",
                f"Decoded video frames do not all have timestamps: {key}",
                episode_index,
            )
            return
        deltas = np.diff(pts)
        expected = 1.0 / fps
        if (
            not math.isclose(pts[0], 0.0, abs_tol=0.5 / fps)
            or np.any(deltas <= 0)
            or not np.allclose(deltas, expected, rtol=0.10, atol=0.001)
        ):
            self._issue(
                "FAIL",
                "video_pts_mismatch",
                f"Video timestamps are non-monotonic or inconsistent with FPS: {key}",
                episode_index,
            )

    def _validate_v3_segment(
        self,
        key: str,
        decoded: DecodedVideo,
        metadata: dict[str, Any],
        expected_frames: int,
        start: int,
        episode_index: int,
        timestamps: np.ndarray | None,
    ) -> None:
        prefix = f"videos/{key}"
        try:
            from_timestamp = float(metadata[f"{prefix}/from_timestamp"])
            to_timestamp = float(metadata[f"{prefix}/to_timestamp"])
        except (KeyError, TypeError, ValueError):
            self._issue(
                "FAIL",
                "video_segment_metadata_invalid",
                f"Video segment timestamps are missing or invalid: {key}",
                episode_index,
            )
            return
        fps = self._source.fps
        duration = expected_frames / fps
        if (
            not math.isfinite(from_timestamp)
            or not math.isfinite(to_timestamp)
            or from_timestamp < 0
            or to_timestamp <= from_timestamp
            or not math.isclose(
                to_timestamp - from_timestamp, duration, abs_tol=0.5 / fps
            )
            or not math.isclose(from_timestamp * fps, start, abs_tol=0.5)
        ):
            self._issue(
                "FAIL",
                "video_segment_metadata_invalid",
                f"Video segment boundaries do not match episode length: {key}",
                episode_index,
            )
        end = start + expected_frames
        if end > decoded.frame_count:
            self._issue(
                "FAIL",
                "video_frame_mismatch",
                f"Video frames do not cover episode data: {key}",
                episode_index,
            )
        if timestamps is not None and len(timestamps) == expected_frames:
            expected_timestamps = np.arange(expected_frames, dtype=np.float64) / fps
            if not np.allclose(
                timestamps, expected_timestamps, rtol=0.0, atol=0.5 / fps
            ):
                self._issue(
                    "FAIL",
                    "video_data_timestamp_mismatch",
                    f"Data timestamps do not align with the video segment: {key}",
                    episode_index,
                )
        pts = np.asarray(decoded.pts_seconds, dtype=np.float64)
        if end <= len(pts) and (
            not math.isclose(pts[start], from_timestamp, abs_tol=0.5 / fps)
            or not math.isclose(
                pts[end - 1], to_timestamp - 1.0 / fps, abs_tol=0.5 / fps
            )
        ):
            self._issue(
                "FAIL",
                "video_segment_pts_mismatch",
                f"Decoded video PTS do not align with segment metadata: {key}",
                episode_index,
            )


def _positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _sample_indices(length: int) -> list[int]:
    if length < 1:
        return []
    minimum = min(length, 100)
    count = max(minimum, min(int(length**0.75), 10_000))
    return np.round(np.linspace(0, length - 1, count)).astype(int).tolist()
