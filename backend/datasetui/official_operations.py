"""Thin official v3 adapter. Call only on a private staging destination.

Registry authorization, lease assertions and atomic publication remain in the
existing job boundary. Unsupported input is rejected, never silently converted.
"""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from types import SimpleNamespace

import av
import numpy as np
import pandas as pd

from datasetui.lerobot_runtime import ENGINE_POLICY, require_runtime, UPSTREAM_COMMIT
from datasetui.transform_errors import CurationTransformError



MERGE_STATISTICS_POLICY = "lerobot-official-aggregate-v1"
OFFICIAL_STATISTICS_MARKER = "datasetui-official-statistics-v1"


def _episode_metadata_digest(root: Path) -> str:
    digest = hashlib.sha256()
    episodes = root / "meta/episodes"
    legacy = root / "meta/episodes_stats.jsonl"
    if episodes.is_symlink() or legacy.is_symlink():
        raise CurationTransformError("Official output episode metadata is unsafe")
    paths = sorted(episodes.rglob("*.parquet")) if episodes.is_dir() else []
    if not paths and legacy.is_file():
        paths = [legacy]
    if not paths:
        raise CurationTransformError("Official output episode metadata is missing")
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise CurationTransformError("Official output episode metadata is unsafe")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_digest(path)))
    return digest.hexdigest()


def _video_digest(root: Path) -> str:
    digest = hashlib.sha256()
    videos = root / "videos"
    if videos.is_symlink():
        raise CurationTransformError("Official output video metadata is unsafe")
    paths = sorted(videos.rglob("*.mp4")) if videos.is_dir() else []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise CurationTransformError("Official output video file is unsafe")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_digest(path)))
    return digest.hexdigest()


def _data_digest(root: Path) -> str:
    digest = hashlib.sha256()
    data = root / "data"
    if data.is_symlink() or not data.is_dir():
        raise CurationTransformError("Official output data is missing or unsafe")
    paths = sorted(data.rglob("*.parquet"))
    if not paths:
        raise CurationTransformError("Official output data is missing")
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise CurationTransformError("Official output data file is unsafe")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_digest(path)))
    return digest.hexdigest()


def _bind_official_statistics(root: Path, *, operation: str) -> None:
    info, stats = root / "meta/info.json", root / "meta/stats.json"
    if (
        operation not in {"merge_datasets", "split_dataset", "aggregate_stats"}
        or info.is_symlink()
        or stats.is_symlink()
        or not info.is_file()
        or not stats.is_file()
    ):
        raise CurationTransformError("Official output statistics are missing")
    marker = {
        "schema": OFFICIAL_STATISTICS_MARKER,
        "statistics_policy": MERGE_STATISTICS_POLICY,
        "engine": ENGINE_POLICY,
        "upstream_commit": UPSTREAM_COMMIT,
        "official_function": operation,
        "info_sha256": _digest(info),
        "stats_sha256": _digest(stats),
        "episodes_sha256": _episode_metadata_digest(root),
        "videos_sha256": _video_digest(root),
        "data_sha256": _data_digest(root),
    }
    from datasetui.dataset_io.files import write_json_atomic

    write_json_atomic(root / "meta/datasetui_provenance.json", marker)


def write_legacy_aggregated_statistics(
    destination: Path,
    episodes: list[tuple[Path, int, pd.DataFrame]],
) -> bool:
    """Reuse v2 per-episode stats and aggregate them with pinned LeRobot."""
    if not episodes:
        raise CurationTransformError("Legacy statistics require output episodes")
    by_root: dict[Path, dict[int, dict]] = {}
    for root, _, _ in episodes:
        if root in by_root:
            continue
        path = root / "meta/episodes_stats.jsonl"
        if not path.exists() and not path.is_symlink():
            return False
        if path.is_symlink() or not path.is_file():
            raise CurationTransformError("Legacy episode statistics are unsafe")
        from datasetui.dataset_io.files import MAX_METADATA_BYTES, read_regular_bytes

        try:
            rows = [
                json.loads(line)
                for line in read_regular_bytes(
                    path, max_bytes=MAX_METADATA_BYTES
                )
                .decode("utf-8")
                .splitlines()
                if line.strip()
            ]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CurationTransformError(
                "Legacy episode statistics are invalid"
            ) from exc
        parsed: dict[int, dict] = {}
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"episode_index", "stats"}:
                raise CurationTransformError("Legacy episode statistics are invalid")
            index = row["episode_index"]
            if (
                type(index) is not int
                or index < 0
                or not isinstance(row["stats"], dict)
            ):
                raise CurationTransformError("Legacy episode statistics are invalid")
            if index in parsed:
                raise CurationTransformError(
                    "Legacy episode statistics contain duplicate episodes"
                )
            parsed[index] = row["stats"]
        by_root[root] = parsed

    info = json.loads((destination / "meta/info.json").read_text(encoding="utf-8"))
    required = {
        name
        for name, descriptor in info.get("features", {}).items()
        if isinstance(descriptor, dict)
        and descriptor.get("dtype") not in {"string", "language"}
    }
    output_stats: list[dict[str, dict[str, np.ndarray]]] = []
    for root, source_index, data in episodes:
        source_stats = by_root[root].get(source_index)
        if source_stats is None or not required.issubset(source_stats):
            return False
        converted = _statistics_to_arrays(source_stats)
        for name in ("index", "episode_index", "task_index"):
            if name in required:
                if name not in data:
                    raise CurationTransformError(
                        f"Legacy bookkeeping feature is missing: {name}"
                    )
                converted[name] = _statistics_from_values(data[name])
        output_stats.append(converted)

    require_runtime()
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.io_utils import write_stats

    write_stats(aggregate_stats(output_stats), destination)
    from datasetui.dataset_io.files import write_json_lines

    write_json_lines(
        destination / "meta/episodes_stats.jsonl",
        [
            {"episode_index": index, "stats": _statistics_to_json(stats)}
            for index, stats in enumerate(output_stats)
        ],
    )
    _bind_official_statistics(destination, operation="aggregate_stats")
    return True


def _statistics_to_arrays(stats: dict) -> dict[str, dict[str, np.ndarray]]:
    converted: dict[str, dict[str, np.ndarray]] = {}
    for feature, values in stats.items():
        if not isinstance(feature, str) or not isinstance(values, dict):
            raise CurationTransformError("Legacy episode statistics are invalid")
        converted[feature] = {}
        for statistic, value in values.items():
            array = np.asarray(value)
            if (
                not isinstance(statistic, str)
                or array.dtype.kind not in "iuf"
                or array.ndim == 0
                or array.size == 0
                or not np.isfinite(array).all()
            ):
                raise CurationTransformError("Legacy episode statistics are invalid")
            converted[feature][statistic] = array
    return converted


def _statistics_from_values(values: pd.Series) -> dict[str, np.ndarray]:
    rows = [np.asarray(value).reshape(-1) for value in values]
    if not rows or any(row.dtype.kind not in "iuf" for row in rows):
        raise CurationTransformError("Legacy bookkeeping values are invalid")
    matrix = np.stack(rows).astype(np.float64, copy=False)
    if not np.isfinite(matrix).all():
        raise CurationTransformError("Legacy bookkeeping values are invalid")
    return {
        "min": np.min(matrix, axis=0),
        "max": np.max(matrix, axis=0),
        "mean": np.mean(matrix, axis=0),
        "std": np.std(matrix, axis=0),
        "q01": np.quantile(matrix, 0.01, axis=0),
        "q10": np.quantile(matrix, 0.10, axis=0),
        "q50": np.quantile(matrix, 0.50, axis=0),
        "q90": np.quantile(matrix, 0.90, axis=0),
        "q99": np.quantile(matrix, 0.99, axis=0),
        "count": np.asarray([len(matrix)], dtype=np.int64),
    }


def _statistics_to_json(stats: dict[str, dict[str, np.ndarray]]) -> dict:
    return {
        feature: {name: value.tolist() for name, value in values.items()}
        for feature, values in stats.items()
    }


def enabled() -> bool:
    engine = os.environ.get("DATASETUI_PROCESSOR_ENGINE", "legacy")
    if engine not in {"legacy", "lerobot-v3"}:
        raise CurationTransformError("Unknown dataset processing engine")
    return engine == "lerobot-v3"


def provenance(operation: str) -> dict:
    from datasetui.output_statistics import STATISTICS_POLICY

    return {
        "engine": ENGINE_POLICY,
        "upstream_commit": UPSTREAM_COMMIT,
        "video_policy": "source",
        "official_function": operation,
        "statistics_policy": (
            MERGE_STATISTICS_POLICY
            if operation in {"merge_datasets", "split_dataset"} else STATISTICS_POLICY
        ),
        "schema_policy": (
            "lossless-merge-list-f32-timestamp-f64-v2"
            if operation == "merge_datasets" else "source-arrow-types-v1"
        ),
    }


def _digest(path: Path) -> str:
    from datasetui.content_integrity import _hash_regular_file

    return _hash_regular_file(path)[1]


def _inventory(root: Path) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise CurationTransformError("Unsafe official operation source")
    result = {}

    def failed(error):
        raise CurationTransformError(
            "Source directory could not be inspected"
        ) from error

    for current, directories, files in os.walk(root, followlinks=False, onerror=failed):
        for name in directories + files:
            path = Path(current) / name
            mode = path.lstat().st_mode
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise CurationTransformError(
                    "Official operation source contains a symlink or special file"
                )
        for name in files:
            path = Path(current) / name
            result[str(path.relative_to(root))] = _digest(path)
    return result


def _load_source(root: Path, dataset_class):
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        raise CurationTransformError(
            "Official adapter only accepts v3.0; version conversion is never implicit"
        )
    for field in ("data_path", "video_path"):
        template = info.get(field)
        if template and (Path(template).is_absolute() or ".." in Path(template).parts):
            raise CurationTransformError("Unsafe dataset metadata path")
    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    if not tasks.index.is_unique or not all(
        isinstance(value, str) for value in tasks.index
    ):
        raise CurationTransformError(
            "v3 Task metadata lacks the official string index; regenerate this derived dataset"
        )
    if "task_index" not in tasks or tasks["task_index"].duplicated().any():
        raise CurationTransformError("v3 Task indices are missing or duplicated")
    return dataset_class("datasetui/local", root=root, video_backend="pyav")


def _video_signature(path: Path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return (
            stream.codec_context.codec.canonical_name,
            stream.codec_context.format.name,
            stream.average_rate,
        )


def _check_split_codecs(dataset) -> dict[str, tuple]:
    signatures = {}
    for key in dataset.meta.video_keys:
        if "/" in key or "\\" in key or key in {"", ".", ".."}:
            raise CurationTransformError("Unsafe video feature key")
        paths = sorted((dataset.root / "videos" / key).rglob("*.mp4"))
        observed = {_video_signature(path) for path in paths}
        if len(observed) != 1:
            raise CurationTransformError(
                f"Mixed or missing video streams need a per-shard codec policy: {key}"
            )
        signature = observed.pop()
        info = dataset.meta.features[key].get("info", {})
        from lerobot.configs.video import encoder_config_from_video_info

        encoder = encoder_config_from_video_info(info)
        encoder_codec = av.codec.Codec(encoder.vcodec, "w").canonical_name
        if signature != (encoder_codec, encoder.pix_fmt, dataset.meta.fps):
            raise CurationTransformError(
                f"Source stream and declared encoder/FPS differ: {key}; refusing codec fallback"
            )
        signatures[key] = signature
    return signatures


def _event(callback, item):
    if callback:
        callback(
            {
                "stage": "write",
                "completed": 0,
                "total": 0,
                "unit": "items",
                "current_item": item,
                "_force": True,
            }
        )


def _built(lengths: list[int], source_ids: list[int], operation: str):
    return {
        "lineage": [
            {
                "source_episode_index": source_id,
                "output_episode_index": index,
                "source_start_frame": 0,
                "source_end_frame": length,
                "output_length": length,
                "trim_method": "disabled",
                "task_overridden": False,
                "persistent_annotations": 0,
                "event_annotations": 0,
            }
            for index, (source_id, length) in enumerate(
                zip(source_ids, lengths, strict=True)
            )
        ],
        "relative_action": {"enabled": False, "dimensions": [], "statistics": {}},
        "processing": provenance(operation),
        "statistics": {
            "policy": MERGE_STATISTICS_POLICY,
            "source": "pinned-official-episode-aggregate",
            "fallback": False,
        },
    }


def write_official_merge(
    *, sources, destination: Path, robot_type: str, on_progress=None
):
    from datasetui.processing_sources import private_sources
    from datasetui.merge.normalization import plan_merge_normalization, normalize_private_merge_sources

    roots = [source.root for source in sources]
    _validate_destination(destination, [root.resolve() for root in roots])
    plan = plan_merge_normalization(roots, on_progress=on_progress)
    with private_sources(roots, destination.parent, on_progress=on_progress) as copies:
        normalize_private_merge_sources(copies, plan, on_progress=on_progress)
        return _merge_private(
            sources=[SimpleNamespace(root=root) for root in copies],
            destination=destination,
            robot_type=robot_type,
            on_progress=on_progress,
        )


def _merge_private(*, sources, destination: Path, robot_type: str, on_progress=None):
    dataset_class, api = require_runtime()
    roots = [source.root.resolve() for source in sources]
    _validate_destination(destination, roots)
    _event(on_progress, "공식 Merge 원본 보존 검사")
    before = [_inventory(root) for root in roots]
    loaded = [_load_source(root, dataset_class) for root in roots]
    from datasetui.merge.schema import inspect_compatible_data_schema, restore_merged_data_schema

    inspect_compatible_data_schema(roots)
    if any(dataset.meta.robot_type != robot_type for dataset in loaded):
        raise CurationTransformError(
            "Robot type cannot be overridden to bypass official Merge compatibility"
        )
    _event(on_progress, "공식 LeRobot Merge · 원본 영상 복사")
    output = api.merge_datasets(
        loaded,
        "datasetui/merged",
        output_dir=destination,
        concatenate_videos=False,
        concatenate_data=False,
    )
    schema_repair = restore_merged_data_schema(
        source_roots=roots, output_root=destination
    )
    # LeRobot creates default scalar features (timestamp=float32) when it
    # constructs output metadata. Copied Parquet remains lossless float64.
    canonical_info = json.loads((roots[0] / "meta/info.json").read_text())
    if "timestamp" in canonical_info["features"]:
        from datasetui.dataset_io.files import write_json_atomic

        output_info_path = destination / "meta/info.json"
        output_info = json.loads(output_info_path.read_text())
        output_info["features"]["timestamp"] = canonical_info["features"]["timestamp"]
        write_json_atomic(output_info_path, output_info)
    expected = Counter(
        value
        for manifest in before
        for key, value in manifest.items()
        if key.endswith(".mp4")
    )
    actual = Counter(_digest(path) for path in destination.rglob("*.mp4"))
    if expected != actual:
        raise CurationTransformError(
            "Official Merge did not preserve all source video file hashes"
        )
    if before != [_inventory(root) for root in roots]:
        raise CurationTransformError(
            "Source changed during official Merge; result will not be published"
        )
    lengths = [
        int(episode["length"])
        for dataset in loaded
        for episode in dataset.meta.episodes
    ]
    if len(output) != sum(lengths):
        raise CurationTransformError("Official Merge output frame count mismatch")
    _bind_official_statistics(destination, operation="merge_datasets")
    # Keep the global and per-episode statistics emitted by the pinned official
    # merge. In particular, its quantile summaries are not exact global quantiles.
    # Do not append DatasetUI's full numeric/RGB/episode recomputation here.
    from datasetui.output_metadata import copy_modality_metadata

    copy_modality_metadata(roots, destination)
    built = _built(lengths, list(range(len(lengths))), "merge_datasets")
    built["schema_repair"] = schema_repair
    return built


def _validate_destination(destination: Path, roots: list[Path]):
    if destination.exists() or destination.is_symlink():
        raise CurationTransformError("Official operation destination already exists")
    if any(
        destination.resolve().is_relative_to(root)
        or root.is_relative_to(destination.resolve())
        for root in roots
    ):
        raise CurationTransformError(
            "Official operation output must be separate from every source"
        )


def write_official_subset(
    *, source, destination: Path, source_indices: list[int], on_progress=None
):
    from datasetui.processing_sources import private_sources

    _validate_destination(destination, [source.root.resolve()])
    with private_sources(
        [source.root], destination.parent, on_progress=on_progress
    ) as copies:
        return _subset_private(
            source=SimpleNamespace(root=copies[0]),
            destination=destination,
            source_indices=source_indices,
            on_progress=on_progress,
        )


def _subset_private(
    *, source, destination: Path, source_indices: list[int], on_progress=None
):
    dataset_class, api = require_runtime()
    root = source.root.resolve()
    _validate_destination(destination, [root])
    if (
        not source_indices
        or any(type(index) is not int for index in source_indices)
        or source_indices != sorted(set(source_indices))
    ):
        raise CurationTransformError(
            "Official subset requires nonempty, unique, ordered episode IDs"
        )
    _event(on_progress, "공식 Subset 원본·코덱 검사")
    before = _inventory(root)
    loaded = _load_source(root, dataset_class)
    if source_indices[0] < 0 or source_indices[-1] >= loaded.meta.total_episodes:
        raise CurationTransformError("Subset episode index is outside the dataset")
    signatures = _check_split_codecs(loaded)
    _event(on_progress, "공식 LeRobot Subset · 필요한 공유 영상 구간 처리")
    # Output parent is newly reserved; never permit an upstream split name to
    # become a caller-controlled filesystem path.
    private = destination.with_name(destination.name + ".official-split")
    private.mkdir(exist_ok=False)
    result = api.split_dataset(
        loaded, {"selected": source_indices}, output_dir=private
    )["selected"]
    for key, signature in signatures.items():
        paths = list((result.root / "videos" / key).rglob("*.mp4"))
        if not paths or any(_video_signature(path) != signature for path in paths):
            raise CurationTransformError(
                "Official Subset changed source codec, pixel format or FPS"
            )
    if before != _inventory(root):
        raise CurationTransformError(
            "Source changed during official Subset; result will not be published"
        )
    lengths = [int(loaded.meta.episodes[index]["length"]) for index in source_indices]
    if len(result) != sum(lengths):
        raise CurationTransformError("Official Subset output frame count mismatch")
    shutil.move(str(result.root), str(destination))
    private.rmdir()
    # The pinned official split already aggregates global statistics from the
    # selected episode summaries.  Recomputing here would scan all data/video
    # again and would replace the official (conservative) quantile contract.
    _bind_official_statistics(destination, operation="split_dataset")
    from datasetui.output_metadata import copy_modality_metadata

    copy_modality_metadata([root], destination)
    return _built(lengths, source_indices, "split_dataset")
