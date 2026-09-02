"""Source-safe v3.1 language annotation drafts and atomic export."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ..registry.xdg import ensure_xdg_dirs
from .common import data_paths, episode_table, finite, load_info

PERSISTENT_STYLES = {"task_aug", "subtask", "plan", "memory"}
EVENT_STYLES = {"interjection", "vqa"}
ROLES = {"user", "assistant"}
SAY_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "say",
        "description": "Speak a short utterance to the user via the TTS executor.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The verbatim text to speak."}},
            "required": ["text"],
        },
    },
}

_TOOL_CALL = pa.struct([
    pa.field("type", pa.string(), nullable=False),
    pa.field("function", pa.struct([
        pa.field("name", pa.string(), nullable=False),
        pa.field("arguments", pa.struct([pa.field("text", pa.string(), nullable=False)]), nullable=False),
    ]), nullable=False),
])
_PERSISTENT = pa.struct([
    pa.field("role", pa.string(), nullable=False), pa.field("content", pa.string()),
    pa.field("style", pa.string()), pa.field("timestamp", pa.float32(), nullable=False),
    pa.field("camera", pa.string()), pa.field("tool_calls", pa.list_(_TOOL_CALL)),
])
_EVENT = pa.struct([
    pa.field("role", pa.string(), nullable=False), pa.field("content", pa.string()),
    pa.field("style", pa.string()), pa.field("camera", pa.string()),
    pa.field("tool_calls", pa.list_(_TOOL_CALL)),
])


def _draft_path(project_id: str) -> Path:
    if not isinstance(project_id, str) or not project_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in project_id):
        raise ValueError("invalid project_id")
    root = ensure_xdg_dirs().state_dir / "annotations"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    return root / f"{project_id}.json"


def _validate_geometry(content: str | None) -> str | None:
    if content is None:
        return None
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return content
    def walk(item: Any) -> Any:
        if isinstance(item, dict):
            for key, nested in item.items():
                if key in {"bbox", "point", "keypoint"} and isinstance(nested, list):
                    expected_length = 4 if key == "bbox" else 2
                    if len(nested) != expected_length or not all(
                        isinstance(number, (int, float)) and not isinstance(number, bool)
                        and math.isfinite(number) and 0 <= number <= 1
                        for number in nested
                    ):
                        raise ValueError("annotation geometry must be normalized to [0,1]")
                    if key == "bbox" and (nested[0] > nested[2] or nested[1] > nested[3]):
                        raise ValueError("annotation bbox must use ordered xyxy coordinates")
                    item[key] = [round(float(number), 4) for number in nested]
                else:
                    item[key] = walk(nested)
        elif isinstance(item, list):
            return [walk(nested) for nested in item]
        return item
    return json.dumps(walk(value), sort_keys=True, separators=(",", ":"))


def _construct(atom: dict[str, Any]) -> list[dict[str, Any]]:
    kind = atom.get("kind")
    if kind is None:
        return [atom]
    timestamp = atom.get("timestamp", 0)
    content = atom.get("content", atom.get("text"))
    if kind == "task_aug":
        return [{**atom, "role": "user", "style": "task_aug", "timestamp": 0, "content": content}]
    if kind in {"subtask", "plan", "memory"}:
        return [{**atom, "role": "assistant", "style": kind, "content": content}]
    if kind == "interjection":
        return [{**atom, "role": "user", "style": "interjection", "content": content}]
    if kind == "speech":
        return [{**atom, "role": "assistant", "style": None, "content": None, "camera": None,
                 "tool_calls": [{"type": "function", "function": {"name": "say",
                                  "arguments": {"text": str(content or "")}}}]}]
    if kind == "vqa":
        camera = atom.get("camera")
        answer = atom.get("answer")
        answer_content = answer if isinstance(answer, str) else json.dumps(answer, sort_keys=True, separators=(",", ":"))
        return [
            {"role": "user", "content": atom.get("question", content), "style": "vqa",
             "timestamp": timestamp, "camera": camera},
            {"role": "assistant", "content": answer_content, "style": "vqa",
             "timestamp": timestamp, "camera": camera},
        ]
    raise ValueError("unsupported annotation constructor")


def _normalize_atom(atom: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(atom, dict):
        raise ValueError("annotation atom must be an object")
    style = atom.get("style")
    role = atom.get("role")
    timestamp = atom.get("timestamp", 0.0)
    if role not in ROLES:
        raise ValueError("annotation role must be user or assistant")
    if style not in PERSISTENT_STYLES | EVENT_STYLES | {None}:
        raise ValueError("unsupported annotation style")
    if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool) or not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("annotation timestamp must be finite and non-negative")
    content = atom.get("content")
    if content is not None and not isinstance(content, str):
        raise ValueError("annotation content must be a string or null")
    camera = atom.get("camera")
    if camera is not None and not isinstance(camera, str):
        raise ValueError("annotation camera must be a string or null")
    tool_calls = atom.get("tool_calls")
    if tool_calls is not None and not isinstance(tool_calls, list):
        raise ValueError("tool_calls must be a list or null")
    expected_role = {"task_aug": "user", "subtask": "assistant", "plan": "assistant",
                     "memory": "assistant", "interjection": "user"}.get(style)
    if expected_role and role != expected_role:
        raise ValueError(f"style={style} requires role={expected_role}")
    if style == "task_aug" and float(timestamp) != 0:
        raise ValueError("task_aug timestamp must be 0")
    if style != "vqa" and camera is not None:
        raise ValueError("camera must be null for non-vqa annotations")
    if style is not None and tool_calls:
        raise ValueError("styled annotations cannot contain tool_calls")
    if style is not None and (content is None or not content.strip()):
        raise ValueError("styled annotations require non-empty content")
    if style is None:
        if role != "assistant" or content is not None or camera is not None:
            raise ValueError("speech atoms require assistant role and null content/style/camera")
        if not isinstance(tool_calls, list) or len(tool_calls) != 1:
            raise ValueError("speech atoms require exactly one say tool call")
        call = tool_calls[0]
        function = call.get("function") if isinstance(call, dict) else None
        arguments = function.get("arguments") if isinstance(function, dict) else None
        if not isinstance(call, dict) or call.get("type") != "function" or not isinstance(function, dict) \
                or function.get("name") != "say" or not isinstance(arguments, dict) \
                or not isinstance(arguments.get("text"), str) or not arguments["text"]:
            raise ValueError("speech atom must contain canonical say({text}) call")
    if style == "vqa":
        if not camera:
            raise ValueError("vqa annotations require a camera")
        content = _validate_geometry(content)
    return {"id": str(atom.get("id") or uuid.uuid4()), "role": role, "content": content,
            "style": style, "timestamp": round(float(timestamp), 6), "camera": camera,
            "tool_calls": tool_calls}


def _validate_vqa_answer(content: str) -> None:
    try:
        answer = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("vqa assistant content must be JSON") from exc
    if not isinstance(answer, dict):
        raise ValueError("vqa answer must be an object")
    if isinstance(answer.get("detections"), list):
        for detection in answer["detections"]:
            if not isinstance(detection, dict) or not isinstance(detection.get("label"), str) \
                    or detection.get("bbox_format") != "xyxy" \
                    or not isinstance(detection.get("bbox"), list) or len(detection["bbox"]) != 4:
                raise ValueError("invalid vqa bbox answer")
            x1, y1, x2, y2 = detection["bbox"]
            if x1 > x2 or y1 > y2:
                raise ValueError("vqa bbox must use ordered xyxy coordinates")
        return
    if answer.get("point_format") == "xy" and isinstance(answer.get("point"), list) \
            and len(answer["point"]) == 2 and isinstance(answer.get("label"), str):
        return
    if isinstance(answer.get("label"), str) and isinstance(answer.get("count"), int) \
            and not isinstance(answer["count"], bool) and answer["count"] >= 0:
        return
    if all(isinstance(answer.get(key), str) and answer[key] for key in ("label", "attribute", "value")):
        return
    if all(isinstance(answer.get(key), str) and answer[key] for key in ("subject", "relation", "object")):
        return
    raise ValueError("vqa answer does not match bbox/keypoint/count/attribute/spatial schema")


def _validate_atom_set(atoms: list[dict[str, Any]], camera_keys: set[str]) -> None:
    index = 0
    while index < len(atoms):
        atom = atoms[index]
        if atom["style"] != "vqa":
            index += 1
            continue
        if atom["camera"] not in camera_keys:
            raise ValueError("vqa camera must name a dataset image/video feature")
        if atom["role"] != "user":
            raise ValueError("vqa annotations must begin with a user question")
        if index + 1 >= len(atoms):
            raise ValueError("vqa question requires an assistant answer")
        answer = atoms[index + 1]
        if answer["style"] != "vqa" or answer["role"] != "assistant" \
                or answer["camera"] != atom["camera"] or answer["timestamp"] != atom["timestamp"]:
            raise ValueError("vqa question/answer must be adjacent with matching timestamp and camera")
        _validate_vqa_answer(answer["content"])
        index += 2


def _source_atoms(source: Path, episode_index: int) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    table = episode_table(source, episode_index,
                          ["episode_index", "timestamp", "language_persistent", "language_events"])
    for row in table.to_pylist():
        timestamp = float(row.get("timestamp", 0))
        for value in row.get("language_persistent") or []:
            item = dict(value); item["timestamp"] = float(item.get("timestamp", timestamp)); atoms.append(_normalize_atom(item))
        for value in row.get("language_events") or []:
            item = dict(value); item["timestamp"] = timestamp; atoms.append(_normalize_atom(item))
    if atoms:
        # Persistent atoms are broadcast in every row. Stable de-duplication is required.
        unique: dict[str, dict[str, Any]] = {}
        for atom in atoms:
            key = json.dumps({k: atom[k] for k in atom if k != "id"}, sort_keys=True)
            unique.setdefault(key, atom)
        atoms = list(unique.values())
    return atoms


def _load_drafts(project_id: str) -> dict[str, list[dict[str, Any]]]:
    path = _draft_path(project_id)
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value.get("episodes", {}) if isinstance(value, dict) else {}


def list_annotations(source: Path, project_id: str, episode_index: int) -> dict[str, Any]:
    drafts = _load_drafts(project_id)
    key = str(episode_index)
    source_atoms = _source_atoms(source, episode_index)
    atoms = drafts[key] if key in drafts else source_atoms
    return finite({"episode_index": episode_index, "atoms": atoms, "draft": key in drafts,
                   "source_atoms": len(source_atoms)})


def save_annotations(source: Path, project_id: str, episode_index: int,
                     atoms: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(atoms, list):
        raise ValueError("atoms must be a list")
    expanded = [item for atom in atoms for item in _construct(atom)]
    normalized = [_normalize_atom(atom) for atom in expanded]
    # Snap to the nearest original frame. ``min`` keeps the first frame on a
    # tie, matching the reference Space's linear scan.
    table = episode_table(source, episode_index, ["episode_index", "timestamp"])
    timestamps = [float(value) for value in table["timestamp"].to_pylist()] \
        if "timestamp" in table.column_names else []
    if not timestamps:
        raise ValueError("episode does not exist or has no source timestamps")
    for atom in normalized:
        atom["timestamp"] = round(min(timestamps, key=lambda value: abs(value - atom["timestamp"])), 6)
    info = load_info(source)
    camera_keys = {key for key, spec in info.get("features", {}).items()
                   if isinstance(spec, dict) and spec.get("dtype") in {"image", "video"}}
    _validate_atom_set(normalized, camera_keys)
    drafts = _load_drafts(project_id)
    drafts[str(episode_index)] = normalized
    path = _draft_path(project_id)
    payload = json.dumps({"version": 1, "episodes": drafts}, sort_keys=True, separators=(",", ":")) + "\n"
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(payload); output.flush(); os.fsync(output.fileno())
        os.replace(temp_name, path)
        os.chmod(path, 0o600)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return {"saved": True, "episode_index": episode_index, "count": len(normalized)}


def _strip(atom: dict[str, Any], *, persistent: bool) -> dict[str, Any]:
    keys = ("role", "content", "style", "timestamp", "camera", "tool_calls") if persistent else ("role", "content", "style", "camera", "tool_calls")
    return {key: atom.get(key) for key in keys}


def _rewrite_file(path: Path, drafts: dict[str, list[dict[str, Any]]]) -> int:
    parquet = pq.ParquetFile(path)
    if "episode_index" not in parquet.schema_arrow.names or "timestamp" not in parquet.schema_arrow.names:
        return 0
    event_targets: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for raw_episode, atoms in drafts.items():
        episode = int(raw_episode)
        for atom in atoms:
            if atom.get("style") in PERSISTENT_STYLES:
                continue
            # Draft events were snapped against the source episode on save, so
            # their timestamp is already an exact source-frame key.  Re-reading
            # episode tables once per output file would only amplify I/O.
            target = round(float(atom.get("timestamp", 0)), 6)
            event_targets.setdefault((episode, target), []).append(_strip(atom, persistent=False))

    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    try:
        for batch in parquet.iter_batches(batch_size=8192):
            table = pa.Table.from_batches([batch])
            episodes = [int(value) for value in table["episode_index"].to_pylist()]
            timestamps = [float(value) for value in table["timestamp"].to_pylist()]
            existing_persistent = (table["language_persistent"].to_pylist()
                                   if "language_persistent" in table.column_names else [[] for _ in episodes])
            existing_events = (table["language_events"].to_pylist()
                               if "language_events" in table.column_names else [[] for _ in episodes])
            persistent_rows: list[list[dict[str, Any]]] = []
            event_rows: list[list[dict[str, Any]]] = []
            for row_index, (episode, timestamp) in enumerate(zip(episodes, timestamps, strict=True)):
                atoms = drafts.get(str(episode))
                if atoms is None:
                    persistent_rows.append(list(existing_persistent[row_index] or []))
                    event_rows.append(list(existing_events[row_index] or []))
                else:
                    persistent_rows.append([_strip(atom, persistent=True) for atom in atoms
                                            if atom.get("style") in PERSISTENT_STYLES])
                    event_rows.append(list(event_targets.get((episode, round(timestamp, 6)), [])))
            for name, values, arrow_type in (("language_persistent", persistent_rows, pa.list_(_PERSISTENT)),
                                              ("language_events", event_rows, pa.list_(_EVENT))):
                array = pa.array(values, type=arrow_type)
                index = table.schema.get_field_index(name)
                table = table.set_column(index, name, array) if index >= 0 else table.append_column(name, array)
            for legacy in ("subtask_index", "tools"):
                index = table.schema.get_field_index(legacy)
                if index >= 0:
                    table = table.remove_column(index)
            if writer is None:
                writer = pq.ParquetWriter(temp_name, table.schema, compression="snappy", use_dictionary=True)
            writer.write_table(table)
            rows_written += table.num_rows
        if writer is not None:
            writer.close()
            writer = None
            os.replace(temp_name, path)
        return rows_written
    finally:
        if writer is not None:
            writer.close()
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def export_annotations(source: Path, project_id: str, output: Path) -> dict[str, Any]:
    source, output = source.resolve(), output.resolve()
    if output.exists() or output == source or source in output.parents:
        raise ValueError("output must be a new directory outside the source dataset")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    drafts = _load_drafts(project_id)
    try:
        symlink = next((path for path in source.rglob("*") if path.is_symlink()), None)
        if symlink is not None:
            raise ValueError(f"source dataset contains an unsupported symlink: {symlink.relative_to(source)}")
        shutil.copytree(source, temp, dirs_exist_ok=True, copy_function=shutil.copy2)
        for legacy in ("subtasks.parquet", "tasks_high_level.parquet"):
            (temp / "meta" / legacy).unlink(missing_ok=True)
        # Every data file must expose the language columns declared below in
        # info.json, even when only a subset of episodes has drafts.  Stream
        # each file in bounded batches rather than leaving schema drift behind.
        rewrite_paths = set(data_paths(temp))
        rows = sum(_rewrite_file(path, drafts) for path in sorted(rewrite_paths))
        info_path = temp / "meta" / "info.json"
        info = load_info(temp)
        features = info.setdefault("features", {})
        features.pop("subtask_index", None)
        features.pop("tools", None)
        features["language_persistent"] = {"dtype": "language", "shape": [1], "names": None}
        features["language_events"] = {"dtype": "language", "shape": [1], "names": None}
        existing_tools = info.get("tools") if isinstance(info.get("tools"), list) else []
        non_say_tools = [tool for tool in existing_tools
                         if not (isinstance(tool, dict) and isinstance(tool.get("function"), dict)
                                 and tool["function"].get("name") == "say")]
        canonical_existing = next((tool for tool in existing_tools if tool == SAY_TOOL_SCHEMA), None)
        info["tools"] = [*non_say_tools, canonical_existing or SAY_TOOL_SCHEMA]
        info_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, output)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return {"exported": True, "output_path": str(output), "episodes": len(drafts), "rows": rows}
