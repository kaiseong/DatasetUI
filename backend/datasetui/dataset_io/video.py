"""Video codec probing, encoder selection and exact frame-range slicing."""

from __future__ import annotations

import os
import stat
from fractions import Fraction
from pathlib import Path
from typing import Any

from datasetui.dataset_io.files import copy_open_regular_file
from datasetui.job_progress import ProgressCallback, report_progress
from datasetui.transform_errors import CurationTransformError


def slice_video(
    source: Path,
    destination: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    expected_frames: int,
    *,
    codec: str = "h264",
    expected_source_codec: str | None = None,
    on_progress: ProgressCallback | None = None,
    progress_base: int = 0,
    progress_total: int | None = None,
    current_item: str | None = None,
) -> None:
    try:
        import av
    except ImportError as exc:
        raise CurationTransformError("Video transform support is unavailable") from exc
    try:
        descriptor = os.open(
            source, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
        )
    except OSError as exc:
        raise CurationTransformError("Source episode video is unavailable") from exc
    try:
        source_metadata = os.fstat(descriptor)
        if not stat.S_ISREG(source_metadata.st_mode):
            raise CurationTransformError("Source episode video is unavailable")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            input_container = av.open(stream)
            try:
                if not input_container.streams.video:
                    raise CurationTransformError(
                        "Source episode video has no video stream"
                    )
                if expected_source_codec is not None:
                    source_codec = normalize_video_codec(
                        input_container.streams.video[0].name
                    )
                    if source_codec != expected_source_codec:
                        raise CurationTransformError(
                            "Source video codec changed while the trim was running"
                        )
                frames = []
                copy_whole_file = (
                    expected_source_codec is not None
                    and start_frame == 0
                    and end_frame == expected_frames
                )
                for index, frame in enumerate(input_container.decode(video=0)):
                    if index >= end_frame:
                        copy_whole_file = False
                        break
                    if index < start_frame:
                        continue
                    frames.append(frame)
                    report_progress(
                        on_progress,
                        stage="video",
                        completed=progress_base + len(frames),
                        total=progress_total or expected_frames * 2,
                        unit="frame_operations",
                        current_item=(
                            f"{current_item} · 디코딩" if current_item else "디코딩"
                        ),
                    )
            finally:
                input_container.close()
        if len(frames) != expected_frames:
            raise CurationTransformError("Video and data frame counts do not match")
        if copy_whole_file:
            copy_open_regular_file(
                descriptor,
                destination,
                source_metadata=source_metadata,
            )
            report_progress(
                on_progress,
                stage="video",
                completed=progress_base + expected_frames * 2,
                total=progress_total or expected_frames * 2,
                unit="frame_operations",
                current_item=(
                    f"{current_item} · 원본 영상 복사 (재인코딩 없음)"
                    if current_item
                    else "원본 영상 복사 (재인코딩 없음)"
                ),
                force=True,
            )
            return
    finally:
        os.close(descriptor)
    source_format = frames[0].format.name
    encoder = video_encoder(
        codec,
        width=frames[0].width,
        height=frames[0].height,
        pixel_format=source_format,
    )
    output = av.open(str(destination), mode="w")
    try:
        frame_rate = Fraction(str(fps)).limit_denominator(1_000_000)
        stream = output.add_stream(encoder, rate=frame_rate)
        stream.width = frames[0].width
        stream.height = frames[0].height
        stream.pix_fmt = compatible_pixel_format(encoder, source_format)
        stream.options = video_encoder_options(encoder)
        stream.thread_count = max(1, min(4, os.cpu_count() or 1))
        frame_time_base = 1 / frame_rate
        stream.time_base = frame_time_base
        for encoded_count, frame in enumerate(frames, start=1):
            frame.pts = encoded_count - 1
            frame.time_base = frame_time_base
            for packet in stream.encode(frame):
                output.mux(packet)
            report_progress(
                on_progress,
                stage="video",
                completed=progress_base + expected_frames + encoded_count,
                total=progress_total or expected_frames * 2,
                unit="frame_operations",
                current_item=f"{current_item} · 인코딩" if current_item else "인코딩",
            )
        for packet in stream.encode():
            output.mux(packet)
    finally:
        output.close()


def probe_video_codec(path: Path) -> str:
    try:
        import av
    except ImportError as exc:
        raise CurationTransformError("Video transform support is unavailable") from exc
    try:
        descriptor = os.open(
            path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
        )
    except OSError as exc:
        raise CurationTransformError("Source episode video is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise CurationTransformError("Source episode video is unavailable")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            container = av.open(stream)
            try:
                if not container.streams.video:
                    raise CurationTransformError(
                        "Source episode video has no video stream"
                    )
                return normalize_video_codec(container.streams.video[0].name)
            finally:
                container.close()
    finally:
        os.close(descriptor)


def normalize_video_codec(name: str) -> str:
    normalized = name.lower().replace("_", "-")
    aliases = {
        "av1": "av1",
        "libdav1d": "av1",
        "libaom-av1": "av1",
        "libsvtav1": "av1",
        "h264": "h264",
        "avc1": "h264",
        "libx264": "h264",
        "hevc": "hevc",
        "h265": "hevc",
        "hev1": "hevc",
        "hvc1": "hevc",
        "libx265": "hevc",
        "vp9": "vp9",
        "libvpx-vp9": "vp9",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise CurationTransformError(
            f"Source video codec is not supported for trimming: {name}"
        ) from exc


def video_encoder(
    codec: str,
    *,
    width: int | None = None,
    height: int | None = None,
    pixel_format: str | None = None,
) -> str:
    try:
        import av
    except ImportError as exc:
        raise CurationTransformError("Video transform support is unavailable") from exc
    candidates = {
        "av1": ("libsvtav1", "libaom-av1"),
        "h264": ("libx264",),
        "hevc": ("libx265",),
        "vp9": ("libvpx-vp9",),
    }.get(codec, ())
    for candidate in candidates:
        if candidate == "libsvtav1" and (
            (width is not None and width < 64)
            or (height is not None and height < 64)
            or (pixel_format is not None and not pixel_format.startswith("yuv420p"))
        ):
            continue
        try:
            encoder = av.codec.Codec(candidate, "w")
        except (ValueError, av.error.FFmpegError):
            continue
        if pixel_format is not None and pixel_format not in {
            item.name for item in (encoder.video_formats or [])
        }:
            continue
        return candidate
    raise CurationTransformError(
        f"No encoder is available to preserve source video codec: {codec}"
    )


def compatible_pixel_format(encoder: str, source_format: str) -> str:
    import av

    codec = av.codec.Codec(encoder, "w")
    supported = {item.name for item in (codec.video_formats or [])}
    if source_format in supported:
        return source_format
    raise CurationTransformError(
        f"Encoder cannot preserve source pixel format {source_format}: {encoder}"
    )


def update_video_feature_codec(feature: dict[str, Any], codec: str) -> None:
    dotted_settings = {
        "video.crf",
        "video.encoder",
        "video.fast_decode",
        "video.g",
        "video.preset",
    }
    plain_settings = {"crf", "encoder", "fast_decode", "g", "preset"}
    for location in ("info", "video_info"):
        details = feature.get(location)
        if details is None and location == "info":
            details = feature.setdefault(location, {})
        if isinstance(details, dict):
            for name in dotted_settings:
                details.pop(name, None)
            details["video.codec"] = codec
    video = feature.get("video")
    if isinstance(video, dict):
        for name in plain_settings:
            video.pop(name, None)
        video["codec"] = codec
    for name in dotted_settings:
        feature.pop(name, None)
    feature["video.codec"] = codec


def video_encoder_options(encoder: str) -> dict[str, str]:
    if encoder == "libsvtav1":
        return {"preset": "8", "crf": "30"}
    if encoder == "libaom-av1":
        return {"cpu-used": "8", "crf": "30", "row-mt": "1"}
    if encoder == "libx265":
        return {"preset": "medium", "crf": "28"}
    if encoder == "libvpx-vp9":
        return {"cpu-used": "4", "crf": "30", "b": "0"}
    return {}
