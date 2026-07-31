"""Schema loading and validation for frozen LeRobot contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "contracts"


def _load_schema(name: str) -> dict[str, Any]:
    path = CONTRACTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Contract schema not found: {path}")
    schema = json.loads(path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    return schema


def v21_schema() -> dict[str, Any]:
    return _load_schema("lerobot-v21.schema.json")


def v30_schema() -> dict[str, Any]:
    return _load_schema("lerobot-v30.schema.json")


def language_v31_schema() -> dict[str, Any]:
    return _load_schema("language-v31.schema.json")


def rpc_schema() -> dict[str, Any]:
    return _load_schema("rpc.schema.json")


def _errors(schema: dict[str, Any], value: Any) -> list[str]:
    validator = jsonschema.Draft202012Validator(schema)
    return [error.message for error in sorted(validator.iter_errors(value), key=lambda item: list(item.path))]


def validate_info_v21(info: dict[str, Any]) -> list[str]:
    return _errors(v21_schema(), info)


def validate_info_v30(info: dict[str, Any]) -> list[str]:
    return _errors(v30_schema(), info)


def validate_language_row(row: dict[str, Any], column: str = "persistent") -> list[str]:
    """Validate one logical row while preserving the schema's root references."""
    if column not in {"persistent", "event"}:
        raise ValueError("column must be 'persistent' or 'event'")
    document = {
        "language_persistent": [row] if column == "persistent" else [],
        "language_events": [row] if column == "event" else [],
    }
    return _errors(language_v31_schema(), document)


def validate_vqa_answer(answer: dict[str, Any]) -> list[str]:
    schema = language_v31_schema()
    wrapper = {
        "$schema": schema["$schema"],
        "$defs": schema["$defs"],
        "$ref": "#/$defs/vqa_answer",
    }
    return _errors(wrapper, answer)
