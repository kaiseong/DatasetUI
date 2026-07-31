"""Deterministic, contract-derived LeRobot fixtures.

The generated datasets freeze the official v0.3.3 v2.1 core layout and the
v0.6.0 v3.0 core layout.  They are intentionally described as
*contract-derived*: official-loader compatibility is a separate compatibility
gate and is never inferred merely because these fixtures validate locally.
"""

from __future__ import annotations

import binascii
import json
import math
import random
import shutil
import struct
import zlib
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SEED = 42
FIXTURES_DIR = Path(__file__).resolve().parents[3] / "fixtures"
_GENERATED_MARKER = "CONTRACT-DERIVED"

V21_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
V21_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
V30_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
V30_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"

SAY_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "say",
        "description": "Speak a short utterance to the user via the TTS executor.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The verbatim text to speak."}
            },
            "required": ["text"],
        },
    },
}


def _prepare_generated_dir(path: Path) -> None:
    """Reset only an empty directory or one previously made by this generator."""
    if path.exists() and any(path.iterdir()):
        provenance = path / "PROVENANCE.md"
        if not provenance.exists() or _GENERATED_MARKER not in provenance.read_text(encoding="utf-8"):
            raise FileExistsError(f"Refusing to replace non-fixture directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", binascii.crc32(kind + payload))


def _rgb_png(red: int, green: int, blue: int, *, width: int = 2, height: int = 2) -> bytes:
    """Create a tiny valid RGB PNG using only the standard library."""
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    pixel = bytes((red % 256, green % 256, blue % 256))
    scanlines = b"".join(b"\x00" + pixel * width for _ in range(height))
    return signature + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", zlib.compress(scanlines, 9)) + _png_chunk(b"IEND", b"")


def _feature_map(*, cameras: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "observation.state": {"dtype": "float32", "shape": [4], "names": ["s0", "s1", "s2", "s3"]},
        "action": {"dtype": "float32", "shape": [4], "names": ["a0", "a1", "a2", "a3"]},
    }
    for camera in cameras:
        result[camera] = {"dtype": "image", "shape": [2, 2, 3], "names": ["height", "width", "channels"]}
    return result


def generate_v21_info(
    fps: int = 10,
    n_episodes: int = 2,
    n_frames: int = 20,
    robot_type: str = "so100",
) -> dict[str, Any]:
    """Return strict official-v0.3.3-compatible v2.1 core metadata."""
    return {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": n_episodes,
        "total_frames": n_frames,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1 if n_episodes else 0,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": V21_DATA_PATH,
        "video_path": None,
        "features": _feature_map(cameras=("observation.images.top",)),
    }


def generate_v30_info(
    fps: int = 10,
    n_episodes: int = 3,
    n_frames: int = 30,
    robot_type: str = "so100",
    multi_camera: bool = True,
    multi_task: bool = True,
) -> dict[str, Any]:
    cameras = ("observation.images.front", "observation.images.wrist") if multi_camera else ("observation.images.front",)
    return {
        "codebase_version": "v3.0",
        "robot_type": robot_type,
        "total_episodes": n_episodes,
        "total_frames": n_frames,
        "total_tasks": 2 if multi_task else 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": fps,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": V30_DATA_PATH,
        "video_path": None,
        "features": _feature_map(cameras=cameras),
    }


def generate_v30_annotated_info(fps: int = 10, n_episodes: int = 2, n_frames: int = 20) -> dict[str, Any]:
    info = generate_v30_info(fps=fps, n_episodes=n_episodes, n_frames=n_frames)
    info["features"]["language_persistent"] = {"dtype": "language", "shape": [1], "names": None}
    info["features"]["language_events"] = {"dtype": "language", "shape": [1], "names": None}
    info["tools"] = [SAY_TOOL_SCHEMA]
    return info


def _episode_lengths(n_frames: int, n_episodes: int) -> list[int]:
    base, remainder = divmod(n_frames, n_episodes)
    return [base + (1 if index < remainder else 0) for index in range(n_episodes)]


def _numeric_rows(info: dict[str, Any]) -> dict[str, list[Any]]:
    rng_state = random.Random(SEED)
    rng_action = random.Random(SEED + 1)
    columns: dict[str, list[Any]] = {
        "index": [],
        "episode_index": [],
        "frame_index": [],
        "timestamp": [],
        "task_index": [],
        "observation.state": [],
        "action": [],
    }
    camera_keys = [name for name, feature in info["features"].items() if feature["dtype"] == "image"]
    for key in camera_keys:
        columns[key] = []

    global_index = 0
    for episode_index, length in enumerate(_episode_lengths(info["total_frames"], info["total_episodes"])):
        for frame_index in range(length):
            columns["index"].append(global_index)
            columns["episode_index"].append(episode_index)
            columns["frame_index"].append(frame_index)
            columns["timestamp"].append(frame_index / info["fps"])
            columns["task_index"].append(episode_index % info["total_tasks"])
            columns["observation.state"].append([rng_state.uniform(-1.0, 1.0) for _ in range(4)])
            columns["action"].append([rng_action.uniform(-0.5, 0.5) for _ in range(4)])
            for camera_number, key in enumerate(camera_keys):
                columns[key].append(
                    {
                        "bytes": _rgb_png(global_index * 13, camera_number * 97, 255 - global_index * 7),
                        "path": None,
                    }
                )
            global_index += 1
    return columns


def _base_data_table(info: dict[str, Any]) -> pa.Table:
    rows = _numeric_rows(info)
    image_type = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
    arrays: list[pa.Array] = []
    names: list[str] = []
    for name, values in rows.items():
        names.append(name)
        if name in {"index", "episode_index", "frame_index", "task_index"}:
            arrays.append(pa.array(values, type=pa.int64()))
        elif name == "timestamp":
            arrays.append(pa.array(values, type=pa.float32()))
        elif name in {"observation.state", "action"}:
            arrays.append(pa.array(values, type=pa.list_(pa.float32(), 4)))
        else:
            arrays.append(pa.array(values, type=image_type))
    return pa.Table.from_arrays(arrays, names=names)


def _tool_call_type() -> pa.StructType:
    return pa.struct(
        [
            pa.field("type", pa.string(), nullable=False),
            pa.field(
                "function",
                pa.struct(
                    [
                        pa.field("name", pa.string(), nullable=False),
                        pa.field("arguments", pa.struct([pa.field("text", pa.string(), nullable=False)]), nullable=False),
                    ]
                ),
                nullable=False,
            ),
        ]
    )


def _persistent_type() -> pa.StructType:
    return pa.struct(
        [
            pa.field("role", pa.string(), nullable=False),
            pa.field("content", pa.string(), nullable=True),
            pa.field("style", pa.string(), nullable=True),
            pa.field("timestamp", pa.float32(), nullable=False),
            pa.field("camera", pa.string(), nullable=True),
            pa.field("tool_calls", pa.list_(_tool_call_type()), nullable=True),
        ]
    )


def _event_type() -> pa.StructType:
    return pa.struct(
        [
            pa.field("role", pa.string(), nullable=False),
            pa.field("content", pa.string(), nullable=True),
            pa.field("style", pa.string(), nullable=True),
            pa.field("camera", pa.string(), nullable=True),
            pa.field("tool_calls", pa.list_(_tool_call_type()), nullable=True),
        ]
    )


def _speech(text: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "style": None,
        "camera": None,
        "tool_calls": [{"type": "function", "function": {"name": "say", "arguments": {"text": text}}}],
    }


def _vqa_pair(question: str, answer: dict[str, Any], camera: str) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": question, "style": "vqa", "camera": camera, "tool_calls": None},
        {
            "role": "assistant",
            "content": json.dumps(answer, sort_keys=True, separators=(",", ":")),
            "style": "vqa",
            "camera": camera,
            "tool_calls": None,
        },
    ]


def _annotated_data_table(info: dict[str, Any]) -> pa.Table:
    table = _base_data_table(info)
    persistent = [
        {"role": "user", "content": "pick up the object", "style": "task_aug", "timestamp": 0.0, "camera": None, "tool_calls": None},
        {"role": "assistant", "content": "approach and grasp", "style": "subtask", "timestamp": 0.0, "camera": None, "tool_calls": None},
        {"role": "assistant", "content": "1. approach\n2. grasp\n3. lift", "style": "plan", "timestamp": 0.0, "camera": None, "tool_calls": None},
        {"role": "assistant", "content": "object grasped", "style": "memory", "timestamp": 0.5, "camera": None, "tool_calls": None},
        {"role": "assistant", "content": "1. hold position\n2. lift", "style": "plan", "timestamp": 0.1, "camera": None, "tool_calls": None},
    ]
    per_frame_persistent = [persistent for _ in range(table.num_rows)]
    per_frame_events: list[list[dict[str, Any]]] = [[] for _ in range(table.num_rows)]
    per_frame_events[0] = [_speech("Starting now.")]
    per_frame_events[1] = [
        {"role": "user", "content": "lift more slowly", "style": "interjection", "camera": None, "tool_calls": None},
        _speech("I will lift more slowly."),
    ]
    camera = "observation.images.front"
    answers = [
        ("Where is the cup?", {"detections": [{"label": "cup", "bbox_format": "xyxy", "bbox": [0, 0, 1, 1]}]}),
        ("Where is the grasp point?", {"label": "handle", "point_format": "xy", "point": [1, 1]}),
        ("How many cups?", {"label": "cup", "count": 1}),
        ("What color is the cup?", {"label": "cup", "attribute": "color", "value": "red"}),
        ("Where is the cup relative to the tray?", {"subject": "cup", "relation": "inside", "object": "tray"}),
    ]
    for frame_index, (question, answer) in enumerate(answers, start=2):
        per_frame_events[frame_index] = _vqa_pair(question, answer, camera)

    return table.append_column(
        "language_persistent",
        pa.array(per_frame_persistent, type=pa.list_(_persistent_type())),
    ).append_column(
        "language_events",
        pa.array(per_frame_events, type=pa.list_(_event_type())),
    )


def _numeric_stats(table: pa.Table) -> dict[str, dict[str, list[float] | list[int]]]:
    stats: dict[str, dict[str, list[float] | list[int]]] = {}
    for key in ("observation.state", "action"):
        rows = table.column(key).to_pylist()
        dimensions = list(zip(*rows, strict=True))
        means = [sum(values) / len(values) for values in dimensions]
        stats[key] = {
            "min": [min(values) for values in dimensions],
            "max": [max(values) for values in dimensions],
            "mean": means,
            "std": [
                math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
                for values, mean in zip(dimensions, means, strict=True)
            ],
            "count": [len(rows)],
        }
    return stats


def _write_one_row_group_per_episode(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    episodes = table.column("episode_index").to_pylist()
    writer = pq.ParquetWriter(path, table.schema, compression="snappy", use_dictionary=True)
    try:
        start = 0
        while start < table.num_rows:
            episode = episodes[start]
            end = start + 1
            while end < table.num_rows and episodes[end] == episode:
                end += 1
            writer.write_table(table.slice(start, end - start))
            start = end
    finally:
        writer.close()


def _v3_episode_table(info: dict[str, Any], data: pa.Table) -> pa.Table:
    rows: list[dict[str, Any]] = []
    start = 0
    global_stats = _numeric_stats(data)
    for episode_index, length in enumerate(_episode_lengths(info["total_frames"], info["total_episodes"])):
        row: dict[str, Any] = {
            "episode_index": episode_index,
            "tasks": ["pick up object" if episode_index % 2 == 0 else "place object"],
            "length": length,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": start,
            "dataset_to_index": start + length,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for feature, feature_stats in global_stats.items():
            for statistic, value in feature_stats.items():
                row[f"stats/{feature}/{statistic}"] = value
        rows.append(row)
        start += length
    return pa.Table.from_pylist(rows)


def _tasks_table(total_tasks: int) -> pa.Table:
    tasks = ["pick up object", "place object"][:total_tasks]
    return pa.table({"task_index": pa.array(range(total_tasks), type=pa.int64()), "task": tasks})


def _write_provenance(path: Path, format_name: str) -> None:
    path.write_text(
        "# Fixture provenance\n\n"
        f"- Format: {format_name}\n"
        "- Origin: CONTRACT-DERIVED from the pinned LeRobot source contracts.\n"
        "- Generator: `lerobot_dataset_editor.fixtures` with deterministic seed 42.\n"
        "- NOT generated by the official `lerobot` package.\n"
        "- Official loader validation: not claimed; compatibility tests record that separately.\n",
        encoding="utf-8",
    )


def materialize_v21_fixture(output_dir: Path | None = None) -> Path:
    output_dir = output_dir or FIXTURES_DIR / "v21_valid"
    _prepare_generated_dir(output_dir)
    info = generate_v21_info()
    _write_json(output_dir / "meta" / "info.json", info)
    full_table = _base_data_table(info)
    episodes: list[dict[str, Any]] = []
    episode_stats: list[dict[str, Any]] = []
    start = 0
    for episode_index, length in enumerate(_episode_lengths(info["total_frames"], info["total_episodes"])):
        episode_table = full_table.slice(start, length)
        path = output_dir / V21_DATA_PATH.format(episode_chunk=0, episode_index=episode_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(episode_table, path, compression="snappy", use_dictionary=True)
        episodes.append({"episode_index": episode_index, "tasks": ["pick up object"], "length": length})
        episode_stats.append({"episode_index": episode_index, "stats": _numeric_stats(episode_table)})
        start += length
    _write_jsonl(output_dir / "meta" / "episodes.jsonl", episodes)
    _write_jsonl(output_dir / "meta" / "episodes_stats.jsonl", episode_stats)
    _write_jsonl(output_dir / "meta" / "tasks.jsonl", [{"task_index": 0, "task": "pick up object"}])
    _write_json(output_dir / "meta" / "stats.json", _numeric_stats(full_table))
    _write_provenance(output_dir / "PROVENANCE.md", "LeRobot v2.1 (official v0.3.3 contract)")
    return output_dir


def _materialize_v30(output_dir: Path, *, annotated: bool) -> Path:
    _prepare_generated_dir(output_dir)
    info = generate_v30_annotated_info() if annotated else generate_v30_info()
    _write_json(output_dir / "meta" / "info.json", info)
    table = _annotated_data_table(info) if annotated else _base_data_table(info)
    _write_one_row_group_per_episode(table, output_dir / V30_DATA_PATH.format(chunk_index=0, file_index=0))
    episode_table = _v3_episode_table(info, table)
    episodes_path = output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        episode_table,
        episodes_path,
        compression="snappy",
        use_dictionary=True,
    )
    pq.write_table(_tasks_table(info["total_tasks"]), output_dir / "meta" / "tasks.parquet", compression="snappy")
    _write_json(output_dir / "meta" / "stats.json", _numeric_stats(table))
    label = "LeRobot v3.0 + v3.1 language extension" if annotated else "LeRobot v3.0 (official v0.6.0 contract)"
    _write_provenance(output_dir / "PROVENANCE.md", label)
    return output_dir


def materialize_v30_fixture(output_dir: Path | None = None) -> Path:
    return _materialize_v30(output_dir or FIXTURES_DIR / "v30_valid", annotated=False)


def materialize_v30_annotated_fixture(output_dir: Path | None = None) -> Path:
    return _materialize_v30(output_dir or FIXTURES_DIR / "v30_annotated", annotated=True)


def materialize_corrupt_fixture(output_dir: Path | None = None) -> Path:
    output_dir = output_dir or FIXTURES_DIR / "corrupt"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "meta" / "info.json", {"codebase_version": "v99.0", "fps": -1})
    bad = output_dir / "data" / "chunk-000" / "file-000.parquet"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"NOT_A_PARQUET_FILE")
    (output_dir / "PROVENANCE.md").write_text(
        "# Fixture provenance\n\nINTENTIONALLY CORRUPT contract fixture; not a valid dataset.\n",
        encoding="utf-8",
    )
    return output_dir


def materialize_all(base_dir: Path | None = None) -> dict[str, Path]:
    base_dir = base_dir or FIXTURES_DIR
    return {
        "v21_valid": materialize_v21_fixture(base_dir / "v21_valid"),
        "v30_valid": materialize_v30_fixture(base_dir / "v30_valid"),
        "v30_annotated": materialize_v30_annotated_fixture(base_dir / "v30_annotated"),
        "corrupt": materialize_corrupt_fixture(base_dir / "corrupt"),
    }
