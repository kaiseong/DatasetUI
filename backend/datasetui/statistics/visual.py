"""RGB sampling/official moments plus exact sampled-pixel quantiles.

This writer is independent of VideoValidator. RGB quantiles use a 256-bin exact
histogram, not the approximate episode-envelope aggregation. Native depth is
not RGB and is deliberately rejected until a units-preserving path is verified.
"""

from __future__ import annotations

import os
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Iterator

import av
import numpy as np

from datasetui.transform_errors import CurationTransformError


_DECODE_PROGRESS_INTERVAL = 64
_PARALLEL_TICK_SECONDS = 1.0


def _decode_workers() -> int:
    """Video files decoded at once; PyAV decodes and converts without the GIL."""
    configured = os.environ.get("DATASETUI_STATS_DECODE_WORKERS")
    if configured:
        return max(1, int(configured))
    return max(1, min(8, (os.cpu_count() or 1) // 4))


def _fallback_sample_indices(length):
    # v2.1 converter Python 3.10 cannot import the pinned v3 Python >=3.12
    # package. Keep the same verified sampling definition in this extension.
    count = max(min(length, 100), min(int(length**0.75), 10_000))
    return np.round(np.linspace(0, length - 1, count)).astype(int).tolist()


def _fallback_downsample(image):
    size = max(image.shape[1:])
    factor = int(size / 150) if size >= 300 else 1
    return image[:, ::factor, ::factor]


def _measure_video(
    path: str, indices: tuple[int, ...], use_official: bool, tick=None, stop=None
) -> list[tuple[int, "_MeasuredFrame"]]:
    """Decode one video and measure its sampled frames, in decode order.

    ``tick(sampled)`` runs on every sampled frame and every 64 skipped frames;
    progress callbacks double as cancellation checks during long decodes.
    """
    if use_official:
        from lerobot.datasets.compute_stats import (
            auto_downsample_height_width,
            get_feature_stats,
        )
    else:
        auto_downsample_height_width = _fallback_downsample
        get_feature_stats = None
    measure = _FrameMeasure(use_official, get_feature_stats)
    wanted = set(indices)
    measured = []
    with av.open(path) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if stop is not None and stop.is_set():
                return measured  # abandoned; the caller already failed
            if index not in wanted:
                if tick is not None and index % _DECODE_PROGRESS_INTERVAL == 0:
                    tick(len(measured))
                continue
            rgb = frame.to_ndarray(format="rgb24")
            chw = auto_downsample_height_width(rgb.transpose(2, 0, 1))
            measured.append((index, measure(chw)))
            if tick is not None:
                tick(len(measured))
            if len(measured) == len(wanted):
                break
    if len(measured) != len(wanted):
        raise CurationTransformError(
            "Video is missing expected statistics sample frames"
        )
    return measured


def _measured_videos(
    targets: dict, use_official: bool, tick
) -> Iterator[tuple[Path, list[tuple[int, "_MeasuredFrame"]]]]:
    """Measured sampled frames per video, yielded in ``targets`` order.

    Files decode in parallel threads, but results are consumed strictly in
    the original file order, so accumulation (including the order-sensitive
    official float aggregation) is identical to a sequential scan. ``tick``
    (sampled frames of the current file) keeps running while workers decode.
    """
    jobs = []
    for path, indices in targets.items():
        if path.is_symlink() or not path.is_file():
            raise CurationTransformError("Invalid output video for statistics")
        jobs.append((path, tuple(sorted(indices))))
    workers = min(_decode_workers(), len(jobs))
    if workers <= 1:
        for path, indices in jobs:
            yield path, _measure_video(str(path), indices, use_official, tick)
        return
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rgb-stats")
    stop = threading.Event()
    finished = False
    try:
        pending: deque = deque()
        remaining = iter(jobs)

        def submit() -> None:
            job = next(remaining, None)
            if job is not None:
                path, indices = job
                pending.append(
                    (
                        path,
                        pool.submit(
                            _measure_video, str(path), indices, use_official, None, stop
                        ),
                    )
                )

        # A bounded window keeps finished-but-unconsumed results small.
        for _ in range(workers * 2):
            submit()
        while pending:
            path, future = pending.popleft()
            while True:
                try:
                    measured = future.result(timeout=_PARALLEL_TICK_SECONDS)
                    break
                except FuturesTimeout:
                    tick(0)
            submit()
            yield path, measured
        finished = True
    finally:
        if not finished:
            stop.set()  # cancellation or error: running decodes stop early
        pool.shutdown(wait=True, cancel_futures=True)


def histogram_statistics(histogram: np.ndarray, frame_count: int) -> dict:
    if histogram.shape != (3, 256) or np.any(histogram < 0):
        raise CurationTransformError("Invalid sampled RGB histogram")
    counts = histogram.sum(axis=1)
    if (
        histogram.shape != (3, 256)
        or frame_count < 1
        or np.any(counts <= 0)
        or not np.all(counts == counts[0])
    ):
        raise CurationTransformError("Invalid sampled RGB histogram")
    levels = np.arange(256, dtype=np.float64)
    mean = (histogram * levels).sum(axis=1) / counts
    variance = (histogram * (levels - mean[:, None]) ** 2).sum(axis=1) / counts
    active = histogram > 0
    values = {
        "min": active.argmax(axis=1).astype(np.float64),
        "max": (255 - active[:, ::-1].argmax(axis=1)).astype(np.float64),
        "mean": mean,
        "std": np.sqrt(variance),
        "count": [frame_count],
    }
    cumulative = histogram.cumsum(axis=1)
    for percentile in (1, 10, 50, 90, 99):
        position = (int(counts[0]) - 1) * percentile / 100
        lower, upper = int(np.floor(position)), int(np.ceil(position))
        lows = np.asarray(
            [np.searchsorted(row, lower, side="right") for row in cumulative]
        )
        highs = np.asarray(
            [np.searchsorted(row, upper, side="right") for row in cumulative]
        )
        values[f"q{percentile:02d}"] = lows + (highs - lows) * (position - lower)
    return {
        key: (value if key == "count" else (value / 255).reshape(3, 1, 1).tolist())
        for key, value in values.items()
    }


def recompute_visual_statistics(
    root: Path, *, on_progress=None, episode_indices=None
) -> dict:
    from datasetui.dataset_io.files import read_json
    from datasetui.dataset_io.source import DatasetSource
    from datasetui.official.operations import enabled

    info = read_json(root / "meta/info.json")
    if not any(
        feature.get("dtype") == "video" for feature in info["features"].values()
    ):
        return {}
    source = DatasetSource(root, info)
    if episode_indices is not None and not set(episode_indices).issubset(
        source.episode_metadata
    ):
        raise CurationTransformError("Unknown episode in visual statistics selection")
    use_official = enabled()
    if use_official:
        from datasetui.official.runtime import require_runtime

        require_runtime()
        from lerobot.datasets.compute_stats import aggregate_stats, sample_indices
    else:
        aggregate_stats = None
        sample_indices = _fallback_sample_indices

    result = {}
    for key in source.video_keys:
        feature = info["features"][key]
        video_info = feature.get("info", {})
        if (
            feature.get("video.is_depth_map")
            or feature.get("is_depth_map")
            or video_info.get("video.is_depth_map")
            or video_info.get("is_depth_map")
        ):
            raise CurationTransformError(
                "Depth statistics require a native-unit decoder; RGB conversion is forbidden"
            )
        selected = defaultdict(set)
        for episode, metadata in source.episode_metadata.items():
            if episode_indices is not None and episode not in episode_indices:
                continue
            length = int(metadata["length"])
            if length < 1:
                raise CurationTransformError(
                    "Cannot calculate visual stats for an empty episode"
                )
            path, offset = source.video_source(episode, key, metadata)
            selected[path].update(offset + index for index in sample_indices(length))
        total = sum(len(indices) for indices in selected.values())
        completed = 0
        accumulator = _VisualAccumulator(
            key=key,
            use_official=use_official,
            aggregate_stats=aggregate_stats,
        )
        def tick(sampled: int) -> None:
            if on_progress:
                _report_visual_progress(on_progress, key, completed + sampled, total)

        for path, measured in _measured_videos(selected, use_official, tick):
            for _, frame in measured:
                accumulator.add(frame)
                completed += 1
            tick(0)
        result[key] = accumulator.finish()
    return result


def recompute_visual_statistics_with_episodes(
    root: Path, *, on_progress=None
) -> tuple[dict, dict[int, dict]]:
    """Compute global and per-episode RGB statistics in one video decode pass.

    Global samples retain the existing union semantics for shared video shards:
    an overlapping physical sample contributes once globally and once to every
    episode that selected it.
    """
    from datasetui.dataset_io.files import read_json
    from datasetui.dataset_io.source import DatasetSource
    from datasetui.official.operations import enabled

    info = read_json(root / "meta/info.json")
    source = DatasetSource(root, info)
    episode_indices = sorted(source.episode_metadata)
    per_episode: dict[int, dict] = {index: {} for index in episode_indices}
    if not source.video_keys:
        return {}, per_episode

    use_official = enabled()
    if use_official:
        from datasetui.official.runtime import require_runtime

        require_runtime()
        from lerobot.datasets.compute_stats import aggregate_stats, sample_indices
    else:
        aggregate_stats = None
        sample_indices = _fallback_sample_indices

    result = {}
    for key in source.video_keys:
        _reject_depth_feature(info["features"][key])
        # path -> physical frame -> episodes selecting that frame. Dict order
        # follows metadata order, matching the legacy global scan order.
        targets: dict[Path, dict[int, set[int]]] = {}
        for episode, metadata in source.episode_metadata.items():
            length = int(metadata["length"])
            if length < 1:
                raise CurationTransformError(
                    "Cannot calculate visual stats for an empty episode"
                )
            path, offset = source.video_source(episode, key, metadata)
            path_targets = targets.setdefault(path, defaultdict(set))
            for index in set(sample_indices(length)):
                path_targets[offset + index].add(episode)

        total = sum(len(indices) for indices in targets.values())
        completed = 0
        global_accumulator = _VisualAccumulator(
            key=key,
            use_official=use_official,
            aggregate_stats=aggregate_stats,
        )
        episode_accumulators = {
            episode: _VisualAccumulator(
                key=key,
                use_official=use_official,
                aggregate_stats=aggregate_stats,
            )
            for episode in episode_indices
        }
        def tick(sampled: int) -> None:
            if on_progress:
                _report_visual_progress(on_progress, key, completed + sampled, total)

        for path, measured in _measured_videos(targets, use_official, tick):
            indices = targets[path]
            for index, frame in measured:
                global_accumulator.add(frame)
                for episode in indices[index]:
                    episode_accumulators[episode].add(frame)
                completed += 1
            tick(0)
        result[key] = global_accumulator.finish()
        for episode, accumulator in episode_accumulators.items():
            per_episode[episode][key] = accumulator.finish()
    return result, per_episode


class _MeasuredFrame:
    """One sampled frame's histogram counts and official per-frame moments."""

    __slots__ = ("shape", "counts", "batch")

    def __init__(self, shape, counts, batch) -> None:
        self.shape = shape
        self.counts = counts
        self.batch = batch


class _FrameMeasure:
    """Measure a sampled frame once; global and episode stats share the result.

    The official per-frame moments are lerobot ``get_feature_stats`` without
    its quantile histograms, which this module discards anyway. The first
    frame is checked against the full lerobot call; any difference falls back
    to that call for the rest of the camera.
    """

    def __init__(self, use_official, get_feature_stats) -> None:
        self.use_official = use_official
        self.get_feature_stats = get_feature_stats
        self.verified = False
        self.fast = True

    def __call__(self, chw: np.ndarray) -> _MeasuredFrame:
        counts = np.stack(
            [np.bincount(chw[channel].ravel(), minlength=256) for channel in range(3)]
        ).astype(np.int64, copy=False)
        batch = None
        if self.use_official:
            array = chw[None].astype(np.float64)
            if not self.verified:
                official = self._official(array)
                fast = _image_moments(array)
                self.fast = fast is not None and _same_moments(fast, official)
                self.verified = True
                raw = official
            elif self.fast:
                raw = _image_moments(array)
            else:
                raw = self._official(array)
            batch = {
                name: value if name == "count" else np.squeeze(value, axis=0) / 255
                for name, value in raw.items()
            }
        return _MeasuredFrame(chw.shape, counts, batch)

    def _official(self, array: np.ndarray) -> dict:
        # Pixel values are unchanged. float64 avoids the observed float32
        # accumulation error on real RGB.
        stats = self.get_feature_stats(array, axis=(0, 2, 3), keepdims=True)
        return {name: value for name, value in stats.items() if not name.startswith("q")}


def _image_moments(array: np.ndarray) -> dict | None:
    """lerobot get_feature_stats(array, axis=(0, 2, 3), keepdims=True) moments.

    Same reshape and reductions as RunningQuantileStats on a single update:
    its second fold of the same batch adds exactly zero for finite pixels.
    """
    batch_size, channels = array.shape[:2]
    reshaped = array.transpose(0, 2, 3, 1).reshape(-1, channels)
    if reshaped.shape[0] < 2:
        return None  # lerobot uses a different (basic) path here
    mean = np.mean(reshaped, axis=0)
    mean_of_squares = np.mean(reshaped**2, axis=0)
    shape = (1, channels, 1, 1)
    return {
        "min": np.min(reshaped, axis=0).reshape(shape),
        "max": np.max(reshaped, axis=0).reshape(shape),
        "mean": mean.reshape(shape),
        "std": np.sqrt(np.maximum(0, mean_of_squares - mean**2)).reshape(shape),
        "count": np.array([batch_size]),
    }


def _same_moments(left: dict, right: dict) -> bool:
    return left.keys() == right.keys() and all(
        left[name].shape == right[name].shape
        and left[name].dtype == right[name].dtype
        and np.array_equal(left[name], right[name])
        for name in left
    )


class _VisualAccumulator:
    def __init__(
        self,
        *,
        key,
        use_official,
        aggregate_stats,
    ) -> None:
        self.key = key
        self.use_official = use_official
        self.aggregate_stats = aggregate_stats
        self.histogram = np.zeros((3, 256), dtype=np.int64)
        self.running = None
        self.sampled_shape = None
        self.frame_count = 0

    def add(self, measured: "_MeasuredFrame") -> None:
        if self.sampled_shape is None:
            self.sampled_shape = measured.shape
        if measured.shape != self.sampled_shape:
            raise CurationTransformError("Sampled RGB shapes differ within a camera")
        self.histogram += measured.counts
        if self.use_official:
            self.running = (
                measured.batch
                if self.running is None
                else self.aggregate_stats(
                    [{self.key: self.running}, {self.key: measured.batch}]
                )[self.key]
            )
        self.frame_count += 1

    def finish(self) -> dict:
        stats = histogram_statistics(self.histogram, self.frame_count)
        if self.use_official:
            for name in ("min", "max", "mean", "std", "count"):
                matches = (
                    np.array_equal(self.running[name], stats[name])
                    if name == "count"
                    else np.allclose(
                        self.running[name], stats[name], rtol=1e-5, atol=1e-7
                    )
                )
                if not matches:
                    raise CurationTransformError(
                        "Official RGB statistic disagrees with exact histogram: "
                        f"{self.key}/{name}"
                    )
                stats[name] = self.running[name].tolist()
        return stats


def _reject_depth_feature(feature: dict) -> None:
    video_info = feature.get("info", {})
    if (
        feature.get("video.is_depth_map")
        or feature.get("is_depth_map")
        or video_info.get("video.is_depth_map")
        or video_info.get("is_depth_map")
    ):
        raise CurationTransformError(
            "Depth statistics require a native-unit decoder; RGB conversion is forbidden"
        )


def _report_visual_progress(on_progress, key: str, completed: int, total: int) -> None:
    on_progress(
        {
            "stage": "statistics",
            "completed": completed,
            "total": total,
            "unit": "frames",
            "current_item": f"{key} · RGB 표본 통계",
        }
    )
