"""Frozen characterization of the existing ``dataset_tools.sh`` surface.

The contract is source-fingerprinted so behavior claims cannot silently drift
when the parent repository changes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EDITOR_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = EDITOR_ROOT
DATASET_TOOLS_SH = REPO_ROOT / "dataset_tools.sh"
CHARACTERIZATION_PATH = EDITOR_ROOT / "contracts" / "dataset-tools-characterization.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contract() -> dict[str, Any]:
    return json.loads(CHARACTERIZATION_PATH.read_text(encoding="utf-8"))


def source_fingerprint_status() -> tuple[bool, dict[str, dict[str, str | bool]]]:
    contract = _contract()
    expected = {contract["launcher"]["path"]: contract["launcher"]["sha256"], **contract["sources"]}
    results: dict[str, dict[str, str | bool]] = {}
    for relative_path, expected_hash in expected.items():
        path = REPO_ROOT / relative_path
        actual = _sha256(path) if path.is_file() else "MISSING"
        results[relative_path] = {
            "expected": expected_hash,
            "actual": actual,
            "matches": actual == expected_hash,
        }
    return all(bool(item["matches"]) for item in results.values()), results


@dataclass(frozen=True)
class DatasetToolsCharacterization:
    commands: list[str] = field(default_factory=lambda: ["trim", "merge", "help"])
    aliases: dict[str, str] = field(
        default_factory=lambda: {"trim-stationary": "trim", "trim_stationary": "trim"}
    )
    defaults: dict[str, Any] = field(
        default_factory=lambda: {
            "trim_video_copy_mode": "copy",
            "merge_remux_policy": "never",
            "workers": "auto",
            "push_to_hub": False,
            "dry_run": False,
            "validate_only": False,
            "state_epsilon": 5e-4,
        }
    )
    trim_no_reencode: bool = True
    merge_no_reencode: bool = True
    trim_range_computation: str = (
        "Stationary prefix/suffix detection via maximum absolute component displacement "
        "from the first/last frame with an inclusive epsilon (default 5e-4); retained "
        "boundary seconds are rounded to frames."
    )
    merge_validation_rules: list[str] = field(
        default_factory=lambda: [
            "At least two source datasets required",
            "All sources must have same fps",
            "All sources must have same robot_type",
            "All sources must have same feature schema (keys, dtypes, shapes)",
            "All sources must have same video keys",
        ]
    )

    def script_exists(self) -> bool:
        return DATASET_TOOLS_SH.is_file()

    def to_report(self) -> dict[str, Any]:
        fingerprints_match, fingerprints = source_fingerprint_status()
        return {
            "status": "current" if fingerprints_match else "stale",
            "source_fingerprints_match": fingerprints_match,
            "source_fingerprints": fingerprints,
            "commands": self.commands,
            "aliases": self.aliases,
            "defaults": self.defaults,
            "trim_no_reencode": self.trim_no_reencode,
            "merge_no_reencode": self.merge_no_reencode,
            "trim_range_computation": self.trim_range_computation,
            "merge_validation_rules": self.merge_validation_rules,
            "script_exists": self.script_exists(),
        }


def get_characterization() -> DatasetToolsCharacterization:
    return DatasetToolsCharacterization()
