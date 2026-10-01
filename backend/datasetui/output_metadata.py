"""Keep explicit consumer modality mappings without inventing robot semantics."""

import json
from pathlib import Path

from datasetui.transform_errors import CurationTransformError


def copy_modality_metadata(roots: list[Path], destination: Path) -> None:
    from datasetui.transforms import _read_regular_bytes, _write_json

    values = []
    for root in roots:
        path = root / "meta/modality.json"
        if not path.exists() and not path.is_symlink():
            values.append(None)
            continue
        value = json.loads(_read_regular_bytes(path, max_bytes=1024 * 1024))
        if not isinstance(value, dict):
            raise CurationTransformError("Consumer modality metadata must be an object")
        values.append(value)
    if any(value != values[0] for value in values[1:]):
        raise CurationTransformError(
            "Merge consumer modality mappings differ or are missing in some sources"
        )
    if values and values[0] is not None:
        _write_json(destination / "meta/modality.json", values[0])
    elif values:
        # Explicit, versioned RBY1 contract from the actual GR00T consumer,
        # not a guess based only on vector width. Other robots get no mapping.
        if any(
            not (root / "meta/info.json").is_file() for root in [*roots, destination]
        ):
            return
        infos = [
            json.loads(
                _read_regular_bytes(root / "meta/info.json", max_bytes=16 * 1024 * 1024)
            )
            for root in [*roots, destination]
        ]
        if infos[-1].get("codebase_version") == "v2.1" and all(
            _is_verified_rby1(info) for info in infos
        ):
            groups = {
                "right_arm": {"start": 0, "end": 7},
                "left_arm": {"start": 7, "end": 14},
                "right_gripper": {"start": 14, "end": 15},
                "left_gripper": {"start": 15, "end": 16},
            }
            mapping = {
                "state": groups,
                "action": groups,
                "video": {
                    alias: {"original_key": f"observation.images.{camera}"}
                    for alias, camera in (
                        ("cam_front_head", "front"),
                        ("cam_right_wrist", "right"),
                        ("cam_left_wrist", "left"),
                    )
                },
                "annotation": {
                    "human.task_description": {"original_key": "task_index"}
                },
            }
            _write_json(destination / "meta/modality.json", mapping)
            _write_json(
                destination / "meta/datasetui_consumer_mapping.json",
                {
                    "policy": "rby1-gr00t-explicit-columns-v1",
                    "source": "rby1-gr00t/rby1/modality.json",
                    "consumer_repo_commit": "e1b59f35d26eb0e05a6eabb86ae7d8bdeac142b7",
                    "source_dataset_modified": False,
                },
            )


def _is_verified_rby1(info: dict) -> bool:
    names = (
        [f"right_arm_{i}" for i in range(7)]
        + [f"left_arm_{i}" for i in range(7)]
        + ["right_gripper_0", "left_gripper_0"]
    )
    features = info.get("features", {})
    return (
        info.get("robot_type") == "rby1"
        and all(
            features.get(key, {}).get("names") == names
            and features[key].get("shape") == [16]
            and features[key].get("dtype") == "float32"
            for key in ("action", "observation.state")
        )
        and all(
            features.get(f"observation.images.{camera}", {}).get("dtype") == "video"
            for camera in ("front", "right", "left")
        )
    )
