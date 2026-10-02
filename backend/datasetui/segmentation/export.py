"""Export of approved previews to a new dataset, videos and statistics."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from datasetui.config import Settings
from datasetui.content_integrity import (
    ContentIntegrityError,
    dataset_content_fingerprint,
    dataset_content_manifest,
)
from datasetui.database import Database, RecipeRevisionMismatchError
from datasetui.dataset_io.files import read_regular_bytes, write_json_atomic
from datasetui.dataset_io.source import DatasetSource
from datasetui.datasets import inspect_dataset, scan_storage_area
from datasetui.job_progress import JobProgressReporter, report_progress
from datasetui.segmentation.errors import SegmentationError
from datasetui.segmentation.media import iter_video_arrays
from datasetui.segmentation.paths import output_lock
from datasetui.segmentation.preview import (
    canonical_hash,
    MAX_MANIFEST_BYTES,
    read_manifest,
)
from datasetui.segmentation.selection import has_keep_objects, retained_mask
from datasetui.segmentation.source import load_source
from datasetui.validation import validate_dataset_root



def export_preview(
    database: Database,
    settings: Settings,
    *,
    job_id: str,
    worker_id: str,
    preview_id: str | None = None,
    output_name: str,
    preview_ids: list[str] | None = None,
    previews: list[dict[str, Any]] | None = None,
    recompute_statistics: bool = False,
) -> dict[str, Any]:
    from datasetui.segmentation.preview import verified_preview

    progress = JobProgressReporter(
        database,
        job_id=job_id,
        worker_id=worker_id,
        output_name=output_name,
    )
    report_progress(
        progress,
        stage="preparing",
        completed=0,
        total=1,
        unit="items",
        current_item="승인된 미리보기 확인",
        force=True,
    )

    selected_ids = (
        preview_ids
        or [str(item["preview_id"]) for item in previews or []]
        or ([str(preview_id)] if preview_id else [])
    )
    if not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise SegmentationError("Select distinct approved previews for export")
    selections = []
    for selected_id in selected_ids:
        try:
            _, expected_result = verified_preview(
                database, settings, selected_id, verify_source=False
            )
        except (ValueError, ContentIntegrityError) as exc:
            raise SegmentationError("Approved preview is stale") from exc
        preview_root = settings.jobs_root / "segmentation" / selected_id
        manifest = read_manifest(preview_root)
        for key in ("recipe_hash", "fingerprint"):
            if expected_result.get(key) != manifest.get(key):
                raise SegmentationError("Approved preview metadata is stale")
        artifact_fingerprint = dataset_content_fingerprint(preview_root, reuse_file_digests=True)
        if artifact_fingerprint != expected_result["artifact_fingerprint"]:
            raise SegmentationError("Approved preview artifacts were modified")
        if manifest.get("selection_required"):
            raise SegmentationError("Select text candidates before exporting")
        if manifest.get("review_blocked"):
            raise SegmentationError(
                "The complete clip has an empty selected mask; correct the prompts"
            )
        selections.append((preview_root, manifest, artifact_fingerprint))
    preview_root, manifest, _ = selections[0]
    if (
        len(
            {
                (item[1]["source"]["dataset_id"], item[1]["fingerprint"])
                for item in selections
            }
        )
        != 1
    ):
        raise SegmentationError(
            "All previews must belong to the same source dataset revision"
        )
    view_keys = [
        (item[1]["spec"]["episode_index"], item[1]["spec"]["video_key"])
        for item in selections
    ]
    if len(view_keys) != len(set(view_keys)):
        raise SegmentationError(
            "Only one approved preview is allowed for each episode and camera"
        )
    preview_id = selected_ids[0]
    artifact_fingerprint = canonical_hash([item[2] for item in selections])
    export_recipe_hash = canonical_hash(
        {
            "previews": [item[1]["recipe_hash"] for item in selections],
            "recompute_statistics": recompute_statistics,
        }
    )

    dataset_id = manifest["source"]["dataset_id"]
    source_root, _ = load_source(
        database, settings, dataset_id, manifest["fingerprint"]
    )
    report_progress(
        progress,
        stage="preparing",
        completed=1,
        total=1,
        unit="items",
        current_item="승인된 미리보기 확인",
    )
    derived = settings.nas_root / "derived"
    if derived.is_symlink() or not derived.is_dir():
        raise SegmentationError("Derived dataset storage is unavailable")
    final = derived / output_name
    provenance_path = settings.nas_root / "manifests/segmentation" / f"{job_id}.json"
    with output_lock(settings, output_name):
        database.assert_job_lease(job_id, worker_id=worker_id)
        if final.exists() or final.is_symlink():
            result = _reuse_export(
                database,
                settings,
                final=final,
                provenance_path=provenance_path,
                job_id=job_id,
                preview_id=str(preview_id),
                output_name=output_name,
                recipe_hash=export_recipe_hash,
                artifact_fingerprint=artifact_fingerprint,
            )
            report_progress(
                progress,
                stage="complete",
                completed=1,
                total=1,
                unit="items",
                current_item="기존 Segmentation 결과 재사용",
                force=True,
            )
            return result
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".incoming-segmentation-{job_id}-{uuid.uuid4().hex}-",
                dir=derived,
            )
        )
        destination = staging / output_name
        try:
            database.assert_job_lease(job_id, worker_id=worker_id)
            report_progress(
                progress,
                stage="publish",
                completed=0,
                total=0,
                unit="bytes",
                current_item="원본 데이터셋 복사",
                force=True,
            )
            shutil.copytree(source_root, destination)
            copied_fingerprint = dataset_content_fingerprint(destination)
            if copied_fingerprint != manifest["fingerprint"]:
                raise SegmentationError("Source dataset copy verification failed")
            if dataset_content_fingerprint(source_root, reuse_file_digests=True) != manifest["fingerprint"]:
                raise RecipeRevisionMismatchError(dataset_id)

            def lease() -> None:
                database.assert_job_lease(job_id, worker_id=worker_id)


            # Untouched episodes keep their exact source bytes; each approved
            # episode/camera gets a new source-codec file and repointed metadata.
            written_videos = write_segmented_videos(
                destination,
                [(item_root, item_manifest) for item_root, item_manifest, _ in selections],
                lease,
                on_progress=progress,
            )
            report_progress(
                progress,
                stage="statistics",
                completed=0,
                total=0,
                unit="frames",
                current_item="영상 통계 재계산",
                force=True,
            )
            from datasetui.deferred_statistics import (
                preserve_deferred_statistics,
                read_deferred_statistics,
            )

            # Relative artifacts embed normalization data and cannot be deferred.
            relative_required = (destination / "meta/relative_action.json").exists()
            if recompute_statistics or relative_required:
                if read_deferred_statistics(source_root):
                    from datasetui.output_statistics import write_output_statistics

                    write_output_statistics(destination)
                else:
                    for _, item_manifest, _ in selections:
                        _update_image_statistics(
                            destination,
                            item_manifest["spec"]["video_key"],
                            int(item_manifest["spec"]["episode_index"]),
                        )
                statistics = {"status": "recomputed"}
            else:
                statistics = preserve_deferred_statistics(source_root, destination)
            report_progress(
                progress,
                stage="statistics",
                completed=1,
                total=1,
                unit="items",
                current_item="영상 통계 재계산"
                if statistics["status"] == "recomputed"
                else "분포 통계 미재계산 표시",
                force=True,
            )
            report_progress(
                progress,
                stage="validate",
                completed=0,
                total=2,
                unit="items",
                current_item="전체 검사",
                force=True,
            )
            full_gate = validate_dataset_root(destination, mode="full")
            report_progress(
                progress,
                stage="validate",
                completed=1,
                total=2,
                unit="items",
                current_item="전체 검사",
            )
            export_gate = validate_dataset_root(destination, mode="export_gate")
            if not full_gate["passed"] or not export_gate["passed"]:
                raise SegmentationError("Segmented dataset failed the export gate")
            report_progress(
                progress,
                stage="validate",
                completed=2,
                total=2,
                unit="items",
                current_item="Export Gate 검사",
            )
            for item_root, _, item_fingerprint in selections:
                if dataset_content_fingerprint(item_root, reuse_file_digests=True) != item_fingerprint:
                    raise SegmentationError(
                        "Approved preview artifacts changed during export"
                    )
            if dataset_content_fingerprint(source_root, reuse_file_digests=True) != manifest["fingerprint"]:
                raise RecipeRevisionMismatchError(dataset_id)
            output_manifest = dataset_content_manifest(destination)
            provenance = {
                "schema_version": 1,
                "job_id": job_id,
                "preview_id": str(preview_id),
                "preview_ids": selected_ids,
                "recipe_hash": export_recipe_hash,
                "source_dataset_id": dataset_id,
                "source_fingerprint": manifest["fingerprint"],
                "artifact_fingerprint": artifact_fingerprint,
                "output_name": output_name,
                "output_manifest": output_manifest,
                "model": manifest["model"],
                "statistics": statistics,
                "video_policy": POLICY,
                "segmented_videos": written_videos,
            }
            write_json_atomic(provenance_path, provenance)
            database.assert_job_lease(job_id, worker_id=worker_id)
            if final.exists() or final.is_symlink():
                raise SegmentationError("Segmentation output name already exists")
            report_progress(
                progress,
                stage="publish",
                completed=0,
                total=1,
                unit="items",
                current_item="Segmentation 데이터셋 게시",
                force=True,
            )
            database.begin_job_finalization(job_id, worker_id=worker_id)
            destination.rename(final)
            report_progress(
                progress,
                stage="publish",
                completed=1,
                total=1,
                unit="items",
                current_item="Segmentation 데이터셋 게시",
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    report_progress(
        progress,
        stage="register",
        completed=0,
        total=1,
        unit="items",
        current_item="라이브러리 갱신",
        force=True,
    )
    result = _register_export(database, settings, output_name, output_manifest)
    report_progress(
        progress,
        stage="register",
        completed=1,
        total=1,
        unit="items",
        current_item="라이브러리 갱신",
    )
    report_progress(
        progress,
        stage="complete",
        completed=1,
        total=1,
        unit="items",
        current_item="Segmentation 데이터셋 완료",
        force=True,
    )
    return result


def _update_image_statistics(root: Path, video_key: str, episode_index: int) -> None:
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    source = DatasetSource(root, info)
    paths: dict[Path, list[tuple[int, int]]] = {}
    for index in range(int(info["total_episodes"])):
        data, metadata = source.episode(index)
        path, start = source.video_source(index, video_key, metadata)
        paths.setdefault(path, []).append((start, len(data)))
    episode_image_stats = [
        _image_stats([path], ranges={path: [(start, length)]})
        for path, selections in paths.items()
        for start, length in selections
    ]
    global_stats = _aggregate_image_stats(episode_image_stats)
    stats_path = root / "meta/stats.json"
    stats = (
        json.loads(stats_path.read_text(encoding="utf-8"))
        if stats_path.is_file()
        else {}
    )
    if not isinstance(stats, dict):
        raise SegmentationError("Dataset statistics are invalid")
    stats[video_key] = global_stats
    write_json_atomic(stats_path, stats)

    episode_stats_path = root / "meta/episodes_stats.jsonl"
    if episode_stats_path.is_file():
        rows = [
            json.loads(line)
            for line in episode_stats_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        data, metadata = source.episode(episode_index)
        video_path, start = source.video_source(episode_index, video_key, metadata)
        selected_stats = _image_stats(
            [video_path], ranges={video_path: [(start, len(data))]}
        )
        matched = False
        for row in rows:
            if int(row.get("episode_index", -1)) == episode_index:
                row.setdefault("stats", {})[video_key] = selected_stats
                matched = True
        if not matched:
            raise SegmentationError("Per-episode statistics are incomplete")
        text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        episode_stats_path.write_text(text, encoding="utf-8")
    if str(info.get("codebase_version")) == "v3.0":
        data, metadata = source.episode(episode_index)
        video_path, start = source.video_source(episode_index, video_key, metadata)
        selected_stats = _image_stats(
            [video_path], ranges={video_path: [(start, len(data))]}
        )
        _update_v3_flattened_stats(root, episode_index, video_key, selected_stats)
    episode_stats_root = root / "meta/episodes_stats"
    if episode_stats_root.is_dir() and not episode_stats_root.is_symlink():
        data, metadata = source.episode(episode_index)
        video_path, start = source.video_source(episode_index, video_key, metadata)
        selected_stats = _image_stats(
            [video_path], ranges={video_path: [(start, len(data))]}
        )
        matched = False
        for parquet_path in sorted(episode_stats_root.rglob("*.parquet")):
            if parquet_path.is_symlink():
                raise SegmentationError("Per-episode statistics path is unsafe")
            frame = pd.read_parquet(parquet_path)
            if "episode_index" not in frame.columns:
                raise SegmentationError("Per-episode statistics are invalid")
            rows = frame.index[frame["episode_index"] == episode_index].tolist()
            for row_index in rows:
                if "stats" in frame.columns and isinstance(
                    frame.at[row_index, "stats"], dict
                ):
                    value = dict(frame.at[row_index, "stats"])
                    value[video_key] = selected_stats
                    frame.at[row_index, "stats"] = value
                elif video_key in frame.columns:
                    frame.at[row_index, video_key] = selected_stats
                else:
                    raise SegmentationError(
                        "Per-episode image statistics schema is unsupported"
                    )
                matched = True
            if rows:
                temporary = parquet_path.with_name(f".{parquet_path.name}.tmp")
                frame.to_parquet(temporary, index=False)
                temporary.replace(parquet_path)
        if not matched:
            raise SegmentationError("Per-episode statistics are incomplete")


def _image_stats(
    paths: Any, *, ranges: dict[Path, list[tuple[int, int]]] | None = None
) -> dict[str, Any]:
    minimum = np.full(3, 255, dtype=np.uint8)
    maximum = np.zeros(3, dtype=np.uint8)
    sums = np.zeros(3, dtype=np.float64)
    sums_squared = np.zeros(3, dtype=np.float64)
    histograms = np.zeros((3, 256), dtype=np.int64)
    pixel_count = 0
    frame_count = 0
    for path in paths:
        for index, frame in enumerate(iter_video_arrays(path)):
            selected_ranges = (ranges or {}).get(path)
            if selected_ranges is not None and not any(
                start <= index < start + length for start, length in selected_ranges
            ):
                continue
            flat = frame.reshape(-1, 3)
            minimum = np.minimum(minimum, flat.min(axis=0))
            maximum = np.maximum(maximum, flat.max(axis=0))
            sums += flat.sum(axis=0)
            sums_squared += np.square(flat.astype(np.float64)).sum(axis=0)
            for channel in range(3):
                histograms[channel] += np.bincount(flat[:, channel], minlength=256)
            pixel_count += len(flat)
            frame_count += 1
    if frame_count == 0:
        raise SegmentationError("Video statistics could not be computed")
    mean = sums / pixel_count
    std = np.sqrt(np.maximum(0, sums_squared / pixel_count - np.square(mean)))
    result: dict[str, Any] = {
        "min": (minimum.astype(float) / 255.0).reshape(3, 1, 1).tolist(),
        "max": (maximum.astype(float) / 255.0).reshape(3, 1, 1).tolist(),
        "mean": (mean / 255.0).reshape(3, 1, 1).tolist(),
        "std": (std / 255.0).reshape(3, 1, 1).tolist(),
        "count": [frame_count],
    }
    for name, quantile in (
        ("q01", 0.01),
        ("q10", 0.1),
        ("q50", 0.5),
        ("q90", 0.9),
        ("q99", 0.99),
    ):
        threshold = max(0, int(np.ceil(quantile * pixel_count)) - 1)
        values = [
            int(np.searchsorted(np.cumsum(histograms[channel]), threshold + 1))
            for channel in range(3)
        ]
        result[name] = (
            (np.asarray(values, dtype=float) / 255.0).reshape(3, 1, 1).tolist()
        )
    return result


def _aggregate_image_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise SegmentationError("Video statistics could not be aggregated")
    counts = np.asarray([item["count"][0] for item in items], dtype=np.float64)
    total = float(counts.sum())
    means = np.asarray([item["mean"] for item in items], dtype=np.float64)
    deviations = np.asarray([item["std"] for item in items], dtype=np.float64)
    weights = counts.reshape((-1, 1, 1, 1))
    mean = np.sum(means * weights, axis=0) / total
    variance = (
        np.sum((np.square(deviations) + np.square(means - mean)) * weights, axis=0)
        / total
    )
    aggregated: dict[str, Any] = {
        "min": np.min(np.asarray([item["min"] for item in items]), axis=0).tolist(),
        "max": np.max(np.asarray([item["max"] for item in items]), axis=0).tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "count": [int(total)],
    }
    for name in ("q01", "q10", "q50", "q90", "q99"):
        values = np.asarray([item[name] for item in items], dtype=np.float64)
        aggregated[name] = (np.sum(values * weights, axis=0) / total).tolist()
    return aggregated


def _update_v3_flattened_stats(
    root: Path, episode_index: int, video_key: str, stats: dict[str, Any]
) -> None:
    metadata_root = root / "meta/episodes"
    matched = False
    stats_present = False
    for path in sorted(metadata_root.rglob("*.parquet")):
        if path.is_symlink():
            raise SegmentationError("Episode metadata path is unsafe")
        frame = pd.read_parquet(path)
        stat_columns = [name for name in frame.columns if name.startswith("stats/")]
        stats_present = stats_present or bool(stat_columns)
        if "episode_index" not in frame.columns:
            raise SegmentationError("Episode metadata is invalid")
        rows = frame.index[frame["episode_index"] == episode_index].tolist()
        if not rows:
            continue
        required = {f"stats/{video_key}/{name}" for name in stats}
        if stat_columns and not required.issubset(frame.columns):
            raise SegmentationError(
                "Flattened v3 image statistics schema is incomplete"
            )
        if not stat_columns:
            continue
        for row_index in rows:
            for stat_name, value in stats.items():
                frame.at[row_index, f"stats/{video_key}/{stat_name}"] = value
        temporary = path.with_name(f".{path.name}.tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
        matched = True
    if stats_present and not matched:
        raise SegmentationError("Flattened v3 image statistics are incomplete")


def _reuse_export(
    database: Database,
    settings: Settings,
    *,
    final: Path,
    provenance_path: Path,
    job_id: str,
    preview_id: str,
    output_name: str,
    recipe_hash: str,
    artifact_fingerprint: str,
) -> dict[str, Any]:
    if final.is_symlink() or not final.is_dir() or not provenance_path.is_file():
        raise SegmentationError("Segmentation output name already exists")
    provenance = json.loads(
        read_regular_bytes(provenance_path, max_bytes=MAX_MANIFEST_BYTES)
    )
    if (
        provenance.get("job_id") != job_id
        or provenance.get("preview_id") != preview_id
        or provenance.get("output_name") != output_name
        or provenance.get("recipe_hash") != recipe_hash
        or provenance.get("artifact_fingerprint") != artifact_fingerprint
    ):
        raise SegmentationError("Segmentation output name already exists")
    output_manifest = dataset_content_manifest(final)
    if output_manifest != provenance.get("output_manifest"):
        raise SegmentationError("Published segmentation output was modified")
    return _register_export(database, settings, output_name, output_manifest)


def _register_export(
    database: Database,
    settings: Settings,
    output_name: str,
    output_manifest: dict[str, Any],
) -> dict[str, Any]:
    derived = settings.nas_root / "derived"
    generation = database.begin_dataset_scan("derived")
    database.synchronize_datasets(
        storage_area="derived",
        records=scan_storage_area(
            settings.nas_root,
            "derived",
            max_depth=settings.dataset_scan_max_depth,
        ),
        scan_generation=generation,
    )
    candidate = inspect_dataset(
        area_root=derived, storage_area="derived", relative_path=output_name
    )
    dataset = database.get_dataset(
        next(
            item["id"]
            for item in database.list_datasets(storage_area="derived")
            if item["relative_path"] == output_name
            and item["fingerprint"] == candidate.fingerprint
        )
    )
    return {
        "dataset_id": dataset["id"],
        "output_name": output_name,
        "relative_path": output_name,
        "fingerprint": dataset["fingerprint"],
        "manifest_sha256": output_manifest["tree_sha256"],
    }


POLICY = "source-codec-new-files-v1"


_V3_VIDEO = re.compile(r"^chunk-(\d+)/file-(\d+)\.mp4$")


def _high_quality_options(encoder: str, gop: int) -> dict[str, str]:
    """Visually lossless re-encode with the source GOP (random-access cost)."""
    g = str(max(1, gop))
    if encoder == "libsvtav1":
        return {"preset": "8", "crf": "18", "g": g}
    if encoder == "libaom-av1":
        return {"cpu-used": "6", "crf": "18", "row-mt": "1", "g": g}
    if encoder == "libx264":
        return {"preset": "medium", "crf": "14", "g": g}
    if encoder == "libx265":
        return {"preset": "medium", "crf": "16", "x265-params": f"keyint={g}"}
    if encoder == "libvpx-vp9":
        return {"cpu-used": "4", "crf": "18", "b": "0", "g": g}
    return {"g": g}


def _source_stream(path: Path) -> tuple[str, str]:
    import av

    from datasetui.dataset_io.video import normalize_video_codec

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("Source video has no video stream")
        stream = container.streams.video[0]
        pixel_format = (
            stream.codec_context.format.name
            if stream.codec_context.format is not None
            else "yuv420p"
        )
        return normalize_video_codec(stream.codec_context.name), pixel_format


class _EpisodeWriter:
    def __init__(self, path: Path, codec: str, pixel_format: str, fps: float, width: int, height: int, gop: int):
        import av

        from datasetui.dataset_io.video import compatible_pixel_format, video_encoder

        self.encoder = video_encoder(
            codec, width=width, height=height, pixel_format=pixel_format
        )
        self.path = path
        self.container = av.open(str(path), mode="w")
        rate = Fraction(str(fps)).limit_denominator(1_000_000)
        self.stream = self.container.add_stream(self.encoder, rate=rate)
        self.stream.width, self.stream.height = width, height
        self.stream.pix_fmt = compatible_pixel_format(self.encoder, pixel_format)
        self.stream.options = _high_quality_options(self.encoder, gop)
        self.stream.thread_count = max(1, min(4, os.cpu_count() or 1))
        self.time_base = 1 / rate
        self.stream.time_base = self.time_base
        self.count = 0

    def write(self, array: np.ndarray) -> None:
        import av

        frame = av.VideoFrame.from_ndarray(
            np.asarray(array, dtype=np.uint8), format="rgb24"
        )
        frame.pts = self.count
        frame.time_base = self.time_base
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.count += 1

    def close(self) -> None:
        if self.container is None:
            return
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()
            self.container = None


def _feature_gop(info: dict[str, Any], key: str) -> int:
    feature = info.get("features", {}).get(key, {})
    details = feature.get("info") if isinstance(feature, dict) else None
    value = (details or {}).get("video.g") if isinstance(details, dict) else None
    try:
        return int(value) if value is not None else 2
    except (TypeError, ValueError):
        return 2


def _next_v3_slots(destination: Path, key: str, chunks_size: int):
    used = set()
    root = destination / "videos" / key
    for path in root.glob("chunk-*/file-*.mp4"):
        match = _V3_VIDEO.match(path.relative_to(root).as_posix())
        if match:
            used.add((int(match.group(1)), int(match.group(2))))
    chunk, file_index = max(used, default=(0, -1))
    while True:
        file_index += 1
        if file_index >= chunks_size:
            chunk, file_index = chunk + 1, 0
        if (chunk, file_index) not in used:
            used.add((chunk, file_index))
            yield chunk, file_index


def write_segmented_videos(
    destination: Path,
    selections: list[tuple[Path, dict[str, Any]]],
    check_lease,
    *,
    on_progress=None,
) -> list[dict[str, Any]]:
    from datasetui.segmentation.errors import SegmentationError
    from datasetui.segmentation.media import iter_video_arrays, video_frame_count
    from datasetui.segmentation.paths import safe_regular_path
    from datasetui.segmentation.selection import read_mask
    from datasetui.job_progress import report_progress

    info = json.loads((destination / "meta/info.json").read_text(encoding="utf-8"))
    version = str(info.get("codebase_version"))
    fps = float(info["fps"])
    chunks_size = int(info.get("chunks_size", 1000))
    total = sum(int(manifest["frame_count"]) for _, manifest in selections)
    done = 0
    slots: dict[str, Any] = {}
    grouped: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for item in selections:
        grouped.setdefault(item[1]["source"]["video_path"], []).append(item)
    written: list[dict[str, Any]] = []
    for relative_video, segments in grouped.items():
        source_video = safe_regular_path(destination, destination / relative_video)
        codec, pixel_format = _source_stream(source_video)
        ranges = []
        for preview_root, manifest in sorted(
            segments, key=lambda item: int(item[1]["source"]["start_frame"])
        ):
            width, height = int(manifest["width"]), int(manifest["height"])
            start, count = int(manifest["source"]["start_frame"]), int(manifest["frame_count"])
            if start < 0 or count < 1:
                raise SegmentationError("Selected video range is invalid")
            with Image.open(preview_root / "background.png") as image:
                image.load()
                if image.size != (width, height):
                    raise SegmentationError("Approved background dimensions are invalid")
                background = np.asarray(image.convert("RGB"), dtype=np.uint8)
            key = manifest["spec"]["video_key"]
            if version == "v3.0":
                if key not in slots:
                    slots[key] = _next_v3_slots(destination, key, chunks_size)
                chunk, file_index = next(slots[key])
                target = destination / "videos" / key / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise SegmentationError("Segmented video target already exists")
                location = {"chunk_index": chunk, "file_index": file_index}
            else:
                if start != 0:
                    raise SegmentationError("v2 episode videos must start at frame 0")
                target = source_video.with_name(f".{source_video.name}.segmented.mp4")
                location = None
            ranges.append(
                {
                    "start": start,
                    "end": start + count,
                    "root": preview_root,
                    "manifest": manifest,
                    "background": background,
                    "target": target,
                    "location": location,
                    "size": (width, height),
                    "writer": None,
                }
            )
        if any(left["end"] > right["start"] for left, right in zip(ranges, ranges[1:])):
            raise SegmentationError("Approved episode ranges overlap in the shared video")
        last = max(item["end"] for item in ranges)
        try:
            for index, array in enumerate(iter_video_arrays(source_video)):
                if index >= last:
                    break
                current = next(
                    (item for item in ranges if item["start"] <= index < item["end"]), None
                )
                if current is None:
                    continue
                width, height = current["size"]
                if array.shape[:2] != (height, width):
                    raise SegmentationError("Video dimensions changed before export")
                manifest = current["manifest"]
                relative = index - current["start"]
                keep = retained_mask(
                    manifest["spec"]["mode"],
                    read_mask(current["root"], "protect", relative, width, height),
                    read_mask(current["root"], "replace", relative, width, height),
                    has_keep=has_keep_objects(manifest["spec"]),
                )
                if current["writer"] is None:
                    current["writer"] = _EpisodeWriter(
                        current["target"], codec, pixel_format, fps, width, height,
                        _feature_gop(info, manifest["spec"]["video_key"]),
                    )
                current["writer"].write(np.where(keep[:, :, None], array, current["background"]))
                done += 1
                report_progress(
                    on_progress,
                    stage="video",
                    completed=done,
                    total=total,
                    unit="frames",
                    current_item="분할 에피소드 인코딩 (원본 코덱)",
                )
                if done % 32 == 0:
                    check_lease()
        finally:
            for item in ranges:
                if item["writer"] is not None:
                    item["writer"].close()
        for item in ranges:
            expected = item["end"] - item["start"]
            if item["writer"] is None or item["writer"].count != expected:
                raise SegmentationError("Selected video does not cover the episode")
            if video_frame_count(item["target"]) != expected:
                raise SegmentationError("Segmented video frame count changed")
            if item["location"] is None:
                item["target"].replace(source_video)
                video_path = relative_video
            else:
                video_path = item["target"].relative_to(destination).as_posix()
            written.append(
                {
                    "episode_index": int(item["manifest"]["spec"]["episode_index"]),
                    "video_key": item["manifest"]["spec"]["video_key"],
                    "video_path": video_path,
                    "frames": expected,
                    "codec": codec,
                    "encoder": item["writer"].encoder,
                    **(item["location"] or {}),
                }
            )
        check_lease()
    if version == "v3.0":
        _repoint_v3_metadata(destination, written, fps)
    return written


def _repoint_v3_metadata(destination: Path, written: list[dict[str, Any]], fps: float) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from datasetui.segmentation.errors import SegmentationError

    updates = {(item["episode_index"], item["video_key"]): item for item in written}
    matched = set()
    for path in sorted((destination / "meta/episodes").rglob("*.parquet")):
        if path.is_symlink():
            raise SegmentationError("Episode metadata path is unsafe")
        table = pq.read_table(path)
        episodes = table.column("episode_index").to_pylist()
        hits = [(row, key) for row, episode in enumerate(episodes) for (e, key) in updates if e == episode]
        if not hits:
            continue
        metadata = table.schema.metadata
        for key in {key for _, key in hits}:
            rows = {row: updates[(episodes[row], key)] for row, k in hits if k == key}
            values = {
                f"videos/{key}/chunk_index": lambda item: item["chunk_index"],
                f"videos/{key}/file_index": lambda item: item["file_index"],
                f"videos/{key}/from_timestamp": lambda item: 0.0,
                f"videos/{key}/to_timestamp": lambda item: item["frames"] / fps,
            }
            for name, value in values.items():
                position = table.schema.get_field_index(name)
                if position < 0:
                    raise SegmentationError("v3 episode video metadata is incomplete")
                field = table.schema.field(name)
                column = table.column(name).to_pylist()
                for row, item in rows.items():
                    column[row] = value(item)
                table = table.set_column(position, field, pa.array(column, type=field.type))
            matched |= {(episodes[row], key) for row in rows}
        table = table.replace_schema_metadata(metadata)
        temporary = path.with_name(f".{path.name}.tmp")
        pq.write_table(table, temporary)
        temporary.replace(path)
    if matched != set(updates):
        raise SegmentationError("Segmented episodes are missing from episode metadata")
