from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "contracts" / "appimage-spike.schema.json"
EVIDENCE_PATH = ROOT / "appimage-spike" / "evidence" / "spike-result.json"
BUILDER_PATH = ROOT / "electron-builder.yml"
VERIFIER_PATH = ROOT / "scripts" / "verify-appimage.mjs"

REQUIRED_RESOURCES = {
    "resources/app.asar",
    "resources/python/src/lerobot_dataset_editor/rpc/__main__.py",
    "resources/contracts/rpc.schema.json",
    "resources/contracts/rpc-transport.schema.json",
    "resources/fixtures/v21_valid/meta/info.json",
    "resources/tests/fixtures/official/v21_v033/meta/info.json",
    "resources/third_party/visualize_dataset.REVISION",
    "resources/tools/lerobot_dataset_tools/cli.py",
    "resources/dataset_tools.sh",
}


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_spike_evidence_is_schema_valid_and_truthful() -> None:
    assert SCHEMA_PATH.is_file(), "add the executable AppImage spike evidence schema"
    schema = _json(SCHEMA_PATH)
    jsonschema.Draft202012Validator.check_schema(schema)

    assert EVIDENCE_PATH.is_file(), "generate and retain the AppImage spike evidence"
    evidence = _json(EVIDENCE_PATH)
    jsonschema.Draft202012Validator(schema).validate(evidence)

    assert evidence["status"] == "passed"
    # The retained spike records the host that actually produced the artifact.
    # Do not make the evidence untruthful by pinning a developer workstation
    # release; the schema already requires every host field and this contract
    # only depends on Linux/x86_64 for the selected build target.
    assert evidence["host"]["os"] == "linux"
    assert evidence["host"]["distribution"]
    assert evidence["host"]["version"]
    assert evidence["host"]["architecture"] == "x86_64"
    assert evidence["artifact"]["target"] == "AppImage"
    assert evidence["artifact"]["architecture"] == "x86_64"
    assert evidence["artifact"]["name"] == "DatasetUI-0.2.0-linux-x86_64.AppImage"

    extraction = evidence["extraction"]
    assert extraction["method"] == "--appimage-extract"
    assert extraction["fuse_required"] is False
    assert extraction["exit_code"] == 0
    resources = {item["path"]: item["present"] for item in extraction["resources"]}
    assert REQUIRED_RESOURCES <= resources.keys()
    assert all(resources[path] is True for path in REQUIRED_RESOURCES)

    smoke = evidence["rpc_smoke"]
    assert smoke["exit_code"] == 0
    assert smoke["marker"] == "DATASETUI_RPC_SMOKE"
    assert smoke["service"] == "lerobot-dataset-editor"
    assert smoke["protocol_version"] == 1

    runtime = evidence["python_runtime"]
    assert runtime["python_mode"] == "host-python-spike"
    assert runtime["self_contained"] is False
    assert runtime["dependency_source"] == "host-environment"
    assert runtime["production_packaging_task"] == 22
    assert "not self-contained" in runtime["limitation"].lower()


def test_builder_packages_the_exact_runtime_surface_for_x64() -> None:
    assert BUILDER_PATH.is_file(), "add the electron-builder AppImage configuration"
    config = yaml.safe_load(BUILDER_PATH.read_text(encoding="utf-8"))

    assert config["appId"] == "io.github.kaiseong.datasetui"
    assert config["productName"] == "DatasetUI"
    assert config["artifactName"] == "DatasetUI-${version}-linux-x86_64.${ext}"
    assert config["asar"] is True
    assert set(config["files"]) >= {"out/**/*", "package.json"}
    assert config["linux"]["target"] == [{"target": "AppImage", "arch": ["x64"]}]

    resource_mappings = {
        (item["from"], item["to"])
        for item in config["extraResources"]
    }
    assert resource_mappings >= {
        ("python/src", "python/src"),
        ("contracts", "contracts"),
        ("fixtures", "fixtures"),
        ("tests/fixtures/official", "tests/fixtures/official"),
        ("third_party", "third_party"),
        ("tools", "tools"),
        ("dataset_tools.sh", "dataset_tools.sh"),
    }


def test_spike_command_builds_then_runs_the_verifier() -> None:
    package = _json(ROOT / "package.json")
    scripts = package["scripts"]
    assert scripts["appimage:build"] == (
        "npm run build && electron-builder --linux AppImage --x64 --config electron-builder.yml"
    )
    assert scripts["appimage:verify"] == "node scripts/verify-appimage.mjs"
    assert scripts["appimage:spike"] == "npm run appimage:build && npm run appimage:verify"
    assert VERIFIER_PATH.is_file(), "implement the no-FUSE extraction and RPC smoke verifier"
