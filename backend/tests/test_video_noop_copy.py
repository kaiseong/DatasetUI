from __future__ import annotations

import hashlib
from pathlib import Path

import av
import pandas as pd
import pytest

import datasetui.dataset_io.video as video
from datasetui.dataset_io.source import DatasetSource
from test_trim_source_codec import _encode


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("codec", ["av1", "h264"])
def test_source_policy_copies_an_exact_whole_video_without_encoding(
    tmp_path: Path, monkeypatch, codec: str
) -> None:
    source = tmp_path / f"source-{codec}.mp4"
    output = tmp_path / f"output-{codec}.mp4"
    _encode(source, codec, [20 + index * 10 for index in range(8)])

    def encoding_is_a_failure(*args, **kwargs):
        raise AssertionError(
            "whole-file source preservation must not select an encoder"
        )

    monkeypatch.setattr(video, "video_encoder", encoding_is_a_failure)
    video.slice_video(
        source,
        output,
        0,
        8,
        10,
        8,
        codec=codec,
        expected_source_codec=codec,
    )

    assert _sha256(output) == _sha256(source)


@pytest.mark.parametrize("codec", ["av1", "h264"])
def test_v3_single_episode_file_metadata_resolves_to_whole_file_copy(
    tmp_path: Path, codec: str
) -> None:
    video_key = "observation.images.top"
    source_path = tmp_path / f"videos/{video_key}/chunk-000/file-000.mp4"
    output = tmp_path / f"copied-{codec}.mp4"
    _encode(source_path, codec, [20 + index * 10 for index in range(8)])
    (tmp_path / "meta/episodes/chunk-000").mkdir(parents=True)
    metadata = {
        "episode_index": 0,
        f"videos/{video_key}/chunk_index": 0,
        f"videos/{video_key}/file_index": 0,
        f"videos/{video_key}/from_timestamp": 0.0,
        f"videos/{video_key}/to_timestamp": 0.8,
    }
    pd.DataFrame([metadata]).to_parquet(
        tmp_path / "meta/episodes/chunk-000/file-000.parquet", index=False
    )
    dataset_source = DatasetSource(
        tmp_path,
        {
            "codebase_version": "v3.0",
            "fps": 10,
            "features": {video_key: {"dtype": "video", "shape": [64, 64, 3]}},
        },
    )
    resolved_path, segment_start = dataset_source.video_source(0, video_key, metadata)

    video.slice_video(
        resolved_path,
        output,
        segment_start,
        segment_start + 8,
        dataset_source.fps,
        8,
        codec=codec,
        expected_source_codec=codec,
    )

    assert resolved_path == source_path
    assert segment_start == 0
    assert _sha256(output) == _sha256(source_path)


@pytest.mark.parametrize("codec", ["av1", "h264"])
@pytest.mark.parametrize(("start", "end"), [(0, 4), (4, 8), (2, 6)])
def test_shared_shard_or_trim_range_is_reencoded_not_whole_file_copied(
    tmp_path: Path, monkeypatch, codec: str, start: int, end: int
) -> None:
    source = tmp_path / f"source-{codec}-{start}-{end}.mp4"
    output = tmp_path / f"output-{codec}-{start}-{end}.mp4"
    _encode(source, codec, [20 + index * 10 for index in range(8)])
    original_encoder = video.video_encoder
    selected_encoders: list[str] = []

    def track_encoder(requested_codec: str, **kwargs):
        selected_encoders.append(requested_codec)
        return original_encoder(requested_codec, **kwargs)

    monkeypatch.setattr(video, "video_encoder", track_encoder)
    video.slice_video(
        source,
        output,
        start,
        end,
        10,
        end - start,
        codec=codec,
        expected_source_codec=codec,
    )

    assert selected_encoders == [codec]
    assert _sha256(output) != _sha256(source)
    with av.open(str(output)) as container:
        assert (
            container.streams.video[0].codec_context.codec.id
            == av.codec.Codec(codec, "r").id
        )
        assert len(list(container.decode(video=0))) == end - start


def test_whole_range_without_source_policy_still_uses_requested_encoder(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source-av1.mp4"
    output = tmp_path / "output-h264.mp4"
    _encode(source, "av1", [20 + index * 10 for index in range(8)])
    original_encoder = video.video_encoder
    selected_encoders: list[str] = []

    def track_encoder(requested_codec: str, **kwargs):
        selected_encoders.append(requested_codec)
        return original_encoder(requested_codec, **kwargs)

    monkeypatch.setattr(video, "video_encoder", track_encoder)
    video.slice_video(source, output, 0, 8, 10, 8, codec="h264")

    assert selected_encoders == ["h264"]
    with av.open(str(output)) as container:
        assert (
            container.streams.video[0].codec_context.codec.id
            == av.codec.Codec("h264", "r").id
        )
