"""Write segmented episodes to new video files; never re-encode untouched data.

Untouched episodes (and every other byte of the copied dataset) stay identical
to the source. Each approved episode/camera is rendered into its own new file
using the source camera's codec at high quality, and only the episode metadata
that locates that video is repointed:

* v3.0: shared shards are left untouched; the segmented episode gets a new
  ``videos/{key}/chunk-XXX/file-YYY.mp4`` and its ``videos/{key}/*`` metadata
  (chunk/file index, from/to timestamp) is updated.
* v2.x: the per-episode video file is replaced by its segmented version.
"""

from __future__ import annotations

import json
import os
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from datasetui.segmentation_selection import has_keep_objects, retained_mask

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

    from datasetui.transforms import _normalize_video_codec

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("Source video has no video stream")
        stream = container.streams.video[0]
        pixel_format = (
            stream.codec_context.format.name
            if stream.codec_context.format is not None
            else "yuv420p"
        )
        return _normalize_video_codec(stream.codec_context.name), pixel_format


class _EpisodeWriter:
    def __init__(self, path: Path, codec: str, pixel_format: str, fps: float, width: int, height: int, gop: int):
        import av

        from datasetui.transforms import _compatible_pixel_format, _video_encoder

        self.encoder = _video_encoder(
            codec, width=width, height=height, pixel_format=pixel_format
        )
        self.path = path
        self.container = av.open(str(path), mode="w")
        rate = Fraction(str(fps)).limit_denominator(1_000_000)
        self.stream = self.container.add_stream(self.encoder, rate=rate)
        self.stream.width, self.stream.height = width, height
        self.stream.pix_fmt = _compatible_pixel_format(self.encoder, pixel_format)
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
    from datasetui.segmentation import (
        SegmentationError,
        _iter_video_arrays,
        _read_mask,
        _report_progress,
        _safe_regular_path,
        _video_frame_count,
    )

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
        source_video = _safe_regular_path(destination, destination / relative_video)
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
            for index, array in enumerate(_iter_video_arrays(source_video)):
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
                    _read_mask(current["root"], "protect", relative, width, height),
                    _read_mask(current["root"], "replace", relative, width, height),
                    has_keep=has_keep_objects(manifest["spec"]),
                )
                if current["writer"] is None:
                    current["writer"] = _EpisodeWriter(
                        current["target"], codec, pixel_format, fps, width, height,
                        _feature_gop(info, manifest["spec"]["video_key"]),
                    )
                current["writer"].write(np.where(keep[:, :, None], array, current["background"]))
                done += 1
                _report_progress(
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
            if _video_frame_count(item["target"]) != expected:
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

    from datasetui.segmentation import SegmentationError

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
