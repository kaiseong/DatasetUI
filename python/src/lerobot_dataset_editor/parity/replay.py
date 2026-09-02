"""Robot replay normalization shared by the SVG/URDF renderer."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .common import dimension_names, finite, load_info

SUPPORTED = {
    "so100": {"family": "so", "scale": 10.0},
    "so_100": {"family": "so", "scale": 10.0},
    "so101": {"family": "so", "scale": 10.0},
    "so_101": {"family": "so", "scale": 10.0},
    "so_follower": {"family": "so", "scale": 10.0},
    "openarm": {"family": "openarm", "scale": 3.0},
    "open_arm": {"family": "openarm", "scale": 3.0},
    "unitree_g1": {"family": "g1", "scale": 1.0},
    "g1": {"family": "g1", "scale": 1.0},
}

G1_SDK_TO_URDF = {
    "klefthippitch.q": "left_hip_pitch_joint", "klefthiproll.q": "left_hip_roll_joint",
    "klefthipyaw.q": "left_hip_yaw_joint", "kleftknee.q": "left_knee_joint",
    "kleftanklepitch.q": "left_ankle_pitch_joint", "kleftankleroll.q": "left_ankle_roll_joint",
    "krighthippitch.q": "right_hip_pitch_joint", "krighthiproll.q": "right_hip_roll_joint",
    "krighthipyaw.q": "right_hip_yaw_joint", "krightknee.q": "right_knee_joint",
    "krightanklepitch.q": "right_ankle_pitch_joint", "krightankleroll.q": "right_ankle_roll_joint",
    "kwaistyaw.q": "waist_yaw_joint", "kwaistroll.q": "waist_roll_joint",
    "kwaistpitch.q": "waist_pitch_joint", "kleftshoulderpitch.q": "left_shoulder_pitch_joint",
    "kleftshoulderroll.q": "left_shoulder_roll_joint", "kleftshoulderyaw.q": "left_shoulder_yaw_joint",
    "kleftelbow.q": "left_elbow_joint", "kleftwristroll.q": "left_wrist_roll_joint",
    "kleftwristpitch.q": "left_wrist_pitch_joint", "kleftwristyaw.q": "left_wrist_yaw_joint",
    "krightshoulderpitch.q": "right_shoulder_pitch_joint", "krightshoulderroll.q": "right_shoulder_roll_joint",
    "krightshoulderyaw.q": "right_shoulder_yaw_joint", "krightelbow.q": "right_elbow_joint",
    "krightwristroll.q": "right_wrist_roll_joint", "krightwristpitch.q": "right_wrist_pitch_joint",
    "krightwristyaw.q": "right_wrist_yaw_joint",
}


def _normalize_robot(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _angles(values: list[float]) -> list[float]:
    """Use one unit mode for the current frame, matching the Space renderer."""
    maximum = max((abs(value) for value in values), default=0.0)
    if maximum > 360:
        return [(value - 2048.0) / 2048.0 * math.pi for value in values]
    if maximum > 6.3:
        return [math.radians(value) for value in values]
    return values


def _suffix(value: str) -> str:
    return value.lower().split(".")[-1].replace("_pos", "").replace("_position", "")


def _match(names: list[str], joints: list[str]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for index, name in enumerate(names):
        suffix = _suffix(name)
        exact = next((joint for joint in joints if _suffix(joint) == suffix), None)
        if exact:
            mapping[index] = exact
            continue
        g1_target = G1_SDK_TO_URDF.get(suffix)
        if g1_target in joints:
            mapping[index] = g1_target
            continue
        arm = next((joint for joint in joints
                    if (match := re.match(r"openarm_(left|right)_joint(\d+)$", joint.lower()))
                    and f"{match.group(1)}_joint_{match.group(2)}" in suffix), None)
        if arm:
            mapping[index] = arm
            continue
        finger = next((joint for joint in joints if re.match(r"openarm_(left|right)_finger_joint1$", joint.lower())
                       and joint.lower().split("_")[1] + "_gripper" in suffix), None)
        if finger:
            mapping[index] = finger
            continue
        fuzzy = next((joint for joint in joints if suffix in _suffix(joint) or _suffix(joint) in suffix), None)
        if fuzzy:
            mapping[index] = fuzzy
    return mapping


def map_replay(source: Path, values: list[float], joint_names: list[str] | None = None,
               urdf_joints: list[str] | None = None,
               joint_ranges: list[dict[str, float]] | None = None) -> dict[str, Any]:
    info = load_info(source)
    version = str(info.get("codebase_version", ""))
    robot = _normalize_robot(str(info.get("robot_type", "")))
    config = SUPPORTED.get(robot)
    if not version.startswith("v3") or config is None:
        return {"supported": False, "robot_type": robot, "reason": "replay supports v3 SO100/SO101, OpenArm, and Unitree G1"}
    names = joint_names or dimension_names(info, "observation.state", len(values))
    joints = [joint for joint in (urdf_joints or names) if not joint.endswith("finger_joint2")]
    mapping = _match(names, joints)
    positions: dict[str, float] = {}
    revolute: list[tuple[str, float]] = []
    for index, joint in mapping.items():
        value = float(values[index])
        if "gripper" in joint or "finger" in joint:
            range_item = joint_ranges[index] if joint_ranges and index < len(joint_ranges) else None
            minimum = float(range_item["min"]) if range_item else 0.0
            maximum = float(range_item["max"]) if range_item else 0.0
            value = ((value - minimum) / (maximum - minimum) if minimum < maximum else value / 100.0) * .044
            positions[joint] = value
        else:
            revolute.append((joint, value))
    converted = _angles([value for _, value in revolute])
    for (joint, _), value in zip(revolute, converted, strict=True):
        positions[joint] = value
    for joint in urdf_joints or []:
        if joint.lower().endswith("finger_joint2"):
            first = re.sub(r"finger_joint2$", "finger_joint1", joint, flags=re.IGNORECASE)
            if first in positions:
                positions[joint] = positions[first]
    return finite({"supported": True, "robot_type": robot, "family": config["family"],
                   "scale": config["scale"], "positions": positions,
                   "trail": {"seconds": 1.0, "max_points": 300}})
