"""Fail-closed identity gate for the isolated, pinned processing runtime."""

import hashlib
import importlib.util
import json
from pathlib import Path

from datasetui.transform_errors import CurationTransformError


UPSTREAM_COMMIT = "30074f7f1358b3c015ae1750017200e86e9c4eb6"
ENGINE_POLICY = "lerobot-v3-source-v1"


def verify_source_tree(root: Path, expected: dict[str, str]) -> None:
    actual = {str(path.relative_to(root)) for path in root.rglob("*.py")}
    if actual != set(expected):
        raise CurationTransformError(
            "LeRobot processing source file set differs from the pinned version"
        )
    for relative, expected_hash in expected.items():
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise CurationTransformError(
                "LeRobot processing source contains an unsafe path"
            )
        value = path.read_bytes()
        blob = b"blob " + str(len(value)).encode() + b"\0" + value
        if hashlib.sha1(blob).hexdigest() != expected_hash:
            raise CurationTransformError(
                f"LeRobot processing source differs from the pinned version: {relative}"
            )


def require_runtime():
    spec = importlib.util.find_spec("lerobot")
    if spec is None or spec.origin is None:
        raise CurationTransformError(
            "Pinned LeRobot processing runtime is missing; no legacy fallback was used"
        )
    lock = json.loads(Path(__file__).with_name("lerobot_source_lock.json").read_text())
    if lock["commit"] != UPSTREAM_COMMIT:
        raise CurationTransformError("LeRobot source lock has the wrong revision")
    verify_source_tree(Path(spec.origin).parent, lock["git_blob_sha1"])
    from huggingface_hub.constants import HF_HUB_OFFLINE

    if not HF_HUB_OFFLINE:
        raise CurationTransformError("Processing worker must run with HF_HUB_OFFLINE=1")
    from lerobot.datasets import dataset_tools
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset, dataset_tools
