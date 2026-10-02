"""Export never re-encodes untouched data and keeps the source codec."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from test_trim_source_codec import _source

from datasetui.segmentation.export import write_segmented_videos
from datasetui.segmentation.media import iter_video_arrays
from datasetui.validation.run import validate_dataset_root

FRONT = "observation.images.front"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _preview(tmp_path: Path, episode: int, video_path: str, start: int) -> tuple[Path, dict]:
    root = tmp_path / f"preview-{episode}"
    keep = np.zeros((64, 64), dtype=np.uint8)
    keep[:, :32] = 255
    for target, mask in (("protect", keep), ("replace", np.zeros_like(keep))):
        (root / target).mkdir(parents=True)
        for index in range(10):
            Image.fromarray(mask).save(root / target / f"{index:06d}.png")
    Image.new("RGB", (64, 64), "black").save(root / "background.png")
    return root, {
        "width": 64,
        "height": 64,
        "fps": 10,
        "frame_count": 10,
        "source": {"video_path": video_path, "start_frame": start, "total_frames": 20},
        "spec": {
            "mode": "object_selection",
            "video_key": FRONT,
            "episode_index": episode,
            "prompts": [{"target": "protect"}],
        },
    }


def _codec(path: Path) -> str:
    with av.open(str(path)) as container:
        return container.streams.video[0].codec_context.name


@pytest.mark.parametrize("version", ["v3.0", "v2.1"])
def test_only_selected_episode_is_rewritten_in_source_codec(tmp_path, version):
    source = tmp_path / "source"
    _source(source, version)
    output = tmp_path / "output"
    shutil.copytree(source, output)
    before = {
        path.relative_to(output).as_posix(): _sha(path)
        for path in output.rglob("*")
        if path.is_file()
    }
    if version == "v3.0":
        shard = f"videos/{FRONT}/chunk-000/file-000.mp4"
        selection = _preview(tmp_path, 1, shard, 10)
    else:
        shard = f"videos/chunk-000/{FRONT}/episode_000001.mp4"
        selection = _preview(tmp_path, 1, shard, 0)
    original = list(iter_video_arrays(source / shard))[(10 if version == "v3.0" else 0):][:10]

    [written] = write_segmented_videos(output, [selection], lambda: None)

    changed = {
        name
        for name, digest in before.items()
        if not (output / name).is_file() or _sha(output / name) != digest
    }
    if version == "v3.0":
        new_video = output / f"videos/{FRONT}/chunk-000/file-001.mp4"
        assert written["video_path"] == new_video.relative_to(output).as_posix()
        # The shared shard and every other video byte is untouched.
        assert changed == {"meta/episodes/chunk-000/file-000.parquet"}
        rows = pd.read_parquet(output / "meta/episodes/chunk-000/file-000.parquet")
        moved = rows.set_index("episode_index").loc[1]
        kept = rows.set_index("episode_index").loc[0]
        assert (moved[f"videos/{FRONT}/file_index"], moved[f"videos/{FRONT}/from_timestamp"]) == (1, 0.0)
        assert moved[f"videos/{FRONT}/to_timestamp"] == pytest.approx(1.0)
        assert (kept[f"videos/{FRONT}/file_index"], kept[f"videos/{FRONT}/from_timestamp"]) == (0, 0.0)
    else:
        new_video = output / shard
        assert changed == {shard}
    assert _codec(new_video) in {"libdav1d", "av1", "libaom-av1"}
    frames = list(iter_video_arrays(new_video))
    assert len(frames) == 10
    for frame, source_frame in zip(frames, original):
        assert np.abs(frame[:, 40:].astype(int)).mean() < 6
        assert np.abs(frame[:, :24].astype(int) - source_frame[:, :24].astype(int)).mean() < 6
    # The fixture ships without statistics; everything else must stay as valid
    # as the source, including video decoding and segment coverage.
    report = validate_dataset_root(output, mode="full")
    baseline = validate_dataset_root(source, mode="full")
    codes = lambda value: sorted(issue["code"] for issue in value["issues"])
    assert codes(report) == codes(baseline), json.dumps(report, default=str)[:1500]
    assert {check["id"]: check["status"] for check in report["checks"]}["videos"] == "passed"
