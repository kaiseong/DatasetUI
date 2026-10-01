"""Video decoding/encoding and frame I/O shared by sample, preview and export."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from datasetui.segmentation.errors import SegmentationError
from datasetui.transforms import (
    _report_progress,
)

MAX_IMAGE_PIXELS = 16_000_000


def _write_episode_clip(
    source_path: Path,
    start: int,
    count: int,
    fps: float,
    output_path: Path,
    check_lease: Callable[[], None],
    *,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    current_item: str | None = None,
) -> tuple[int, int]:
    frames = _iter_video_arrays(source_path)
    output = None
    stream = None
    written = 0
    width = height = 0
    _report_progress(
        on_progress,
        stage="read",
        completed=0,
        total=count,
        unit="frames",
        current_item=current_item,
        force=True,
    )
    try:
        for absolute_index, array in enumerate(frames):
            if absolute_index < start:
                continue
            if absolute_index >= start + count:
                break
            if output is None:
                height, width = array.shape[:2]
                if width * height > MAX_IMAGE_PIXELS:
                    raise SegmentationError("Video frame exceeds the image pixel limit")
                output, stream = _open_video_writer(output_path, fps, width, height)
            _encode_array(output, stream, array)
            written += 1
            _report_progress(
                on_progress,
                stage="read",
                completed=written,
                total=count,
                unit="frames",
                current_item=current_item,
            )
            if written % 32 == 0:
                check_lease()
    finally:
        if output is not None and stream is not None:
            _close_video_writer(output, stream)
    if written != count:
        raise SegmentationError("Source video does not cover the full episode")
    return width, height


def _iter_video_arrays(path: Path):
    if path.is_symlink() or not path.is_file():
        raise SegmentationError("Video file is unavailable")
    import av

    container = av.open(str(path))
    try:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")
    finally:
        container.close()


def _decode_one(path: Path, index: int) -> np.ndarray:
    if path.is_symlink() or not path.is_file():
        raise SegmentationError("Video file is unavailable")
    import av

    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        rate = float(stream.average_rate or stream.base_rate or 0)
        time_base = float(stream.time_base)
        if rate <= 0 or time_base <= 0:
            raise SegmentationError("Video timing metadata is unavailable")
        start_pts = int(stream.start_time or 0)
        target_pts = start_pts + int((index / rate) / time_base)
        container.seek(target_pts, stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_index = round((int(frame.pts) - start_pts) * time_base * rate)
            if frame_index == index:
                return frame.to_ndarray(format="rgb24")
            if frame_index > index:
                break
    finally:
        container.close()
    raise SegmentationError("Video frame is unavailable")


def _video_frame_count(path: Path) -> int:
    return sum(1 for _ in _iter_video_arrays(path))


def _open_video_writer(path: Path, fps: float, width: int, height: int):
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=Fraction(str(fps)))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    return container, stream


def _encode_array(container: Any, stream: Any, array: np.ndarray) -> None:
    import av

    frame = av.VideoFrame.from_ndarray(
        np.asarray(array, dtype=np.uint8), format="rgb24"
    )
    for packet in stream.encode(frame):
        container.mux(packet)


def _close_video_writer(container: Any, stream: Any) -> None:
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _png_bytes(image: Image.Image) -> bytes:
    from io import BytesIO

    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
