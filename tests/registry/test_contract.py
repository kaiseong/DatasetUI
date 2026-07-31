from __future__ import annotations

import json
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "contracts" / "project-registry.schema.json"


def test_project_registry_schema_freezes_runtime_compatibility_fields() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    doctor = schema["$defs"]["doctor_report"]
    assert set(doctor["required"]) >= {
        "mode",
        "compatible",
        "issues",
        "python",
        "lerobot",
        "ffmpeg",
        "compute",
        "device_policy_result",
        "probe_environment_secret_seen",
    }


def test_embedded_doctor_result_validates_against_contract() -> None:
    from lerobot_dataset_editor.runtime.doctor import run_doctor

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    wrapper = {
        "$schema": schema["$schema"],
        "$defs": schema["$defs"],
        "$ref": "#/$defs/doctor_report",
    }
    report = run_doctor(device_policy="cpu-only")
    jsonschema.Draft202012Validator(wrapper).validate(report)
