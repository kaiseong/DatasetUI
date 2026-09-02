from __future__ import annotations

from collections.abc import Callable
from typing import Any


JobHandler = Callable[[dict[str, Any]], dict[str, Any]]


def _phase2_smoke(payload: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True}


def _validate_empty_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload:
        raise ValueError("phase2.smoke does not accept payload fields")
    return {}


JOB_HANDLERS: dict[str, tuple[str, JobHandler]] = {
    "phase2.smoke": ("cpu", _phase2_smoke),
}


def queue_for_kind(kind: str) -> str | None:
    registered = JOB_HANDLERS.get(kind)
    return registered[0] if registered else None


def validate_job_payload(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    if kind == "phase2.smoke":
        return _validate_empty_payload(payload)
    raise ValueError(f"Unsupported job kind: {kind}")


def run_registered_job(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    registered = JOB_HANDLERS.get(kind)
    if registered is None:
        raise ValueError(f"Unsupported job kind: {kind}")
    return registered[1](payload)
