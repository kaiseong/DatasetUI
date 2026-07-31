"""Acceptance tests for exact official LeRobot baseline compatibility evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from lerobot_dataset_editor.cli import _v21_summary, _v30_summary
from lerobot_dataset_editor.schemas import validate_info_v21, validate_info_v30

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "tests" / "fixtures" / "official"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.mark.parametrize(
    ("name", "package", "version", "generation_script", "adapter"),
    [
        ("v21_v033", "lerobot", "0.3.3", "scripts/compat/generate_v21_v033.py", "v033"),
        ("v30_v060", "lerobot", "0.6.0", "scripts/compat/generate_v30_v060.py", "v060"),
    ],
)
def test_official_fixture_provenance_is_exact_and_artifacts_match(
    name: str,
    package: str,
    version: str,
    generation_script: str,
    adapter: str,
) -> None:
    root = OFFICIAL / name
    manifest = _json(root / "PROVENANCE.json")
    assert manifest["origin"] == "official-writer-generated"
    assert manifest["generator"] == {"package": package, "version": version}
    assert manifest["python_version"] == "3.12.3"
    assert manifest["generation_script"] == generation_script
    assert manifest["metadata_only_adapter"] == {
        "used": True,
        "path": f"tests/compat/stubs/{adapter}",
        "no_torch_cuda_runtime": True,
    }
    assert manifest["dependencies"]
    assert all("==" in value for value in manifest["dependencies"])
    assert manifest["artifacts"]
    for relative, expected in manifest["artifacts"].items():
        artifact = root / relative
        assert artifact.is_file(), relative
        assert _sha256(artifact) == expected, relative


def test_official_v21_fixture_freezes_actual_v033_writer_layout() -> None:
    root = OFFICIAL / "v21_v033"
    info = _json(root / "meta" / "info.json")
    assert validate_info_v21(info) == []
    assert info["codebase_version"] == "v2.1"
    assert info["video_path"] is None
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 20
    assert not (root / "meta" / "stats.json").exists()
    assert {path.name for path in (root / "meta").iterdir()} >= {
        "info.json",
        "episodes.jsonl",
        "episodes_stats.jsonl",
        "tasks.jsonl",
    }
    summary = _v21_summary(root)
    assert summary["valid"] is True
    assert summary["global_stats_present"] is False


def test_official_v30_fixture_freezes_actual_v060_writer_layout() -> None:
    root = OFFICIAL / "v30_v060"
    info = _json(root / "meta" / "info.json")
    assert validate_info_v30(info) == []
    assert info["codebase_version"] == "v3.0"
    assert info["video_path"] is None
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 10
    summary = _v30_summary(root, annotated=False)
    assert summary["valid"] is True
    assert summary["provenance"] == "official-writer-generated"


def test_contract_fixture_stats_are_computed_from_data(v30_fixture: Path) -> None:
    table = pq.read_table(v30_fixture / "data" / "chunk-000" / "file-000.parquet")
    stats = _json(v30_fixture / "meta" / "stats.json")
    for feature in ("observation.state", "action"):
        rows = table.column(feature).to_pylist()
        for dimension, values in enumerate(zip(*rows, strict=True)):
            mean = sum(values) / len(values)
            expected = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
            assert stats[feature]["std"][dimension] == pytest.approx(expected)
            assert stats[feature]["std"][dimension] > 0


def test_compatibility_scripts_and_opt_in_gate_are_preserved() -> None:
    required = [
        ROOT / "scripts" / "compat" / "generate_v21_v033.py",
        ROOT / "scripts" / "compat" / "generate_v30_v060.py",
        ROOT / "scripts" / "compat" / "validate_official.py",
        ROOT / "scripts" / "compat" / "README.md",
        ROOT / "tests" / "compat" / "test_official_loaders.py",
        ROOT / "contracts" / "official-compatibility.json",
    ]
    assert all(path.is_file() for path in required)
    evidence = _json(ROOT / "contracts" / "official-compatibility.json")
    assert evidence["baselines"] == {"v2.1": "lerobot==0.3.3", "v3.0": "lerobot==0.6.0"}
    assert evidence["validations"]["v21_valid"]["all_frames_loaded"] == 20
    assert evidence["validations"]["v30_valid"]["all_frames_loaded"] == 30
    assert evidence["validations"]["v30_annotated"]["all_frames_loaded"] == 20
    assert evidence["validations"]["v21_valid"]["timestamp_validation_passed"] is True
    assert evidence["validations"]["v30_valid"]["full_loader_passed"] is True
