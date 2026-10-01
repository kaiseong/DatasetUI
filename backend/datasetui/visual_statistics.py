"""RGB sampling/official moments plus exact sampled-pixel quantiles.

This writer is independent of VideoValidator. RGB quantiles use a 256-bin exact
histogram, not the approximate episode-envelope aggregation. Native depth is
not RGB and is deliberately rejected until a units-preserving path is verified.
"""

from collections import defaultdict
from pathlib import Path

import av
import numpy as np

from datasetui.transform_errors import CurationTransformError


_DECODE_PROGRESS_INTERVAL = 64


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
    from datasetui.transforms import _DatasetSource, _read_json
    from datasetui.official_operations import enabled

    info = _read_json(root / "meta/info.json")
    if not any(
        feature.get("dtype") == "video" for feature in info["features"].values()
    ):
        return {}
    source = _DatasetSource(root, info)
    if episode_indices is not None and not set(episode_indices).issubset(
        source.episode_metadata
    ):
        raise CurationTransformError("Unknown episode in visual statistics selection")
    use_official = enabled()
    if use_official:
        from datasetui.lerobot_runtime import require_runtime

        require_runtime()
        from lerobot.datasets.compute_stats import (
            sample_indices,
            auto_downsample_height_width,
            get_feature_stats,
            aggregate_stats,
        )
    else:
        # v2.1 converter Python 3.10 cannot import the pinned v3 Python >=3.12
        # package. Keep the same verified sampling definition in this extension.
        def sample_indices(length):
            count = max(min(length, 100), min(int(length**0.75), 10_000))
            return np.round(np.linspace(0, length - 1, count)).astype(int).tolist()

        def auto_downsample_height_width(image):
            size = max(image.shape[1:])
            factor = int(size / 150) if size >= 300 else 1
            return image[:, ::factor, ::factor]

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
        histogram = np.zeros((3, 256), dtype=np.int64)
        running = None
        sampled_shape = None
        for path, indices in selected.items():
            found = set()
            if path.is_symlink() or not path.is_file():
                raise CurationTransformError("Invalid output video for statistics")
            with av.open(str(path)) as container:
                for index, frame in enumerate(container.decode(video=0)):
                    if index not in indices:
                        continue
                    rgb = frame.to_ndarray(format="rgb24")
                    chw = auto_downsample_height_width(rgb.transpose(2, 0, 1))
                    if sampled_shape is None:
                        sampled_shape = chw.shape
                    if chw.shape != sampled_shape:
                        raise CurationTransformError(
                            "Sampled RGB shapes differ within a camera"
                        )
                    for channel in range(3):
                        histogram[channel] += np.bincount(
                            chw[channel].ravel(), minlength=256
                        )
                    if use_official:
                        batch = get_feature_stats(
                            # Pixel values are unchanged. float64 avoids the
                            # observed float32 accumulation error on real RGB.
                            chw[None].astype(np.float64),
                            axis=(0, 2, 3),
                            keepdims=True,
                        )
                        batch = {
                            name: value
                            if name == "count"
                            else np.squeeze(value, axis=0) / 255
                            for name, value in batch.items()
                        }
                        batch = {
                            name: value
                            for name, value in batch.items()
                            if not name.startswith("q")
                        }
                        running = (
                            batch
                            if running is None
                            else aggregate_stats([{key: running}, {key: batch}])[key]
                        )
                    found.add(index)
                    completed += 1
                    if on_progress:
                        on_progress(
                            {
                                "stage": "statistics",
                                "completed": completed,
                                "total": total,
                                "unit": "frames",
                                "current_item": f"{key} · RGB 표본 통계",
                            }
                        )
                    if found == indices:
                        break
            if found != indices:
                raise CurationTransformError(
                    "Video is missing expected statistics sample frames"
                )
        stats = histogram_statistics(histogram, completed)
        if use_official:
            for name in ("min", "max", "mean", "std", "count"):
                matches = (
                    np.array_equal(running[name], stats[name])
                    if name == "count"
                    else np.allclose(running[name], stats[name], rtol=1e-5, atol=1e-7)
                )
                if not matches:
                    raise CurationTransformError(
                        f"Official RGB statistic disagrees with exact histogram: {key}/{name}"
                    )
                stats[name] = running[name].tolist()
        result[key] = stats
    return result


def recompute_visual_statistics_with_episodes(
    root: Path, *, on_progress=None
) -> tuple[dict, dict[int, dict]]:
    """Compute global and per-episode RGB statistics in one video decode pass.

    Global samples retain the existing union semantics for shared video shards:
    an overlapping physical sample contributes once globally and once to every
    episode that selected it.
    """
    from datasetui.transforms import _DatasetSource, _read_json
    from datasetui.official_operations import enabled

    info = _read_json(root / "meta/info.json")
    source = _DatasetSource(root, info)
    episode_indices = sorted(source.episode_metadata)
    per_episode: dict[int, dict] = {index: {} for index in episode_indices}
    if not source.video_keys:
        return {}, per_episode

    use_official = enabled()
    if use_official:
        from datasetui.lerobot_runtime import require_runtime

        require_runtime()
        from lerobot.datasets.compute_stats import (
            aggregate_stats,
            auto_downsample_height_width,
            get_feature_stats,
            sample_indices,
        )
    else:
        aggregate_stats = None
        get_feature_stats = None

        def sample_indices(length):
            count = max(min(length, 100), min(int(length**0.75), 10_000))
            return np.round(np.linspace(0, length - 1, count)).astype(int).tolist()

        def auto_downsample_height_width(image):
            size = max(image.shape[1:])
            factor = int(size / 150) if size >= 300 else 1
            return image[:, ::factor, ::factor]

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
            get_feature_stats=get_feature_stats,
            aggregate_stats=aggregate_stats,
        )
        episode_accumulators = {
            episode: _VisualAccumulator(
                key=key,
                use_official=use_official,
                get_feature_stats=get_feature_stats,
                aggregate_stats=aggregate_stats,
            )
            for episode in episode_indices
        }
        for path, indices in targets.items():
            found = set()
            if path.is_symlink() or not path.is_file():
                raise CurationTransformError("Invalid output video for statistics")
            with av.open(str(path)) as container:
                for index, frame in enumerate(container.decode(video=0)):
                    episodes = indices.get(index)
                    if episodes is None:
                        if on_progress and index % _DECODE_PROGRESS_INTERVAL == 0:
                            _report_visual_progress(on_progress, key, completed, total)
                        continue
                    rgb = frame.to_ndarray(format="rgb24")
                    chw = auto_downsample_height_width(rgb.transpose(2, 0, 1))
                    global_accumulator.update(chw)
                    for episode in episodes:
                        episode_accumulators[episode].update(chw)
                    found.add(index)
                    completed += 1
                    if on_progress:
                        _report_visual_progress(on_progress, key, completed, total)
                    if len(found) == len(indices):
                        break
            if len(found) != len(indices):
                raise CurationTransformError(
                    "Video is missing expected statistics sample frames"
                )
        result[key] = global_accumulator.finish()
        for episode, accumulator in episode_accumulators.items():
            per_episode[episode][key] = accumulator.finish()
    return result, per_episode


class _VisualAccumulator:
    def __init__(
        self,
        *,
        key,
        use_official,
        get_feature_stats,
        aggregate_stats,
    ) -> None:
        self.key = key
        self.use_official = use_official
        self.get_feature_stats = get_feature_stats
        self.aggregate_stats = aggregate_stats
        self.histogram = np.zeros((3, 256), dtype=np.int64)
        self.running = None
        self.sampled_shape = None
        self.frame_count = 0

    def update(self, chw: np.ndarray) -> None:
        if self.sampled_shape is None:
            self.sampled_shape = chw.shape
        if chw.shape != self.sampled_shape:
            raise CurationTransformError("Sampled RGB shapes differ within a camera")
        for channel in range(3):
            self.histogram[channel] += np.bincount(
                chw[channel].ravel(), minlength=256
            )
        if self.use_official:
            batch = self.get_feature_stats(
                chw[None].astype(np.float64), axis=(0, 2, 3), keepdims=True
            )
            batch = {
                name: value if name == "count" else np.squeeze(value, axis=0) / 255
                for name, value in batch.items()
                if not name.startswith("q")
            }
            self.running = (
                batch
                if self.running is None
                else self.aggregate_stats(
                    [{self.key: self.running}, {self.key: batch}]
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
