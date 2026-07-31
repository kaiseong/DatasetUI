"""Device selection policy with safe CPU fallback and explicit GPU-only failure."""

from __future__ import annotations

from typing import Any, Callable

_VALID_POLICIES = {"auto", "cpu-only", "gpu-only"}


def default_cuda_probe() -> dict[str, Any]:
    """Probe the actual current PyTorch CUDA runtime and initialize device 0."""
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": f"PyTorch import failed: {exc}"}
    try:
        if not torch.cuda.is_available():
            return {"available": False, "error": "PyTorch reports CUDA unavailable"}
        properties = torch.cuda.get_device_properties(0)
        # A tiny allocation exercises driver/runtime/context initialization and
        # turns initialization or VRAM failures into a normal probe failure.
        probe_tensor = torch.empty(1, device="cuda:0")
        del probe_tensor
        return {
            "available": True,
            "device": "cuda:0",
            "name": properties.name,
            "vram_mb": int(properties.total_memory // (1024 * 1024)),
        }
    except Exception as exc:
        return {"available": False, "error": f"CUDA initialization failed: {exc}"}


def resolve_device(
    policy: str,
    *,
    cuda_probe: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve ``auto``/``cpu-only``/``gpu-only`` without silent GPU-only fallback."""
    if policy not in _VALID_POLICIES:
        raise ValueError(f"device policy must be one of {sorted(_VALID_POLICIES)}")
    if policy == "cpu-only":
        return {"requested": policy, "selected": "cpu", "fallback_reason": None}

    result = (cuda_probe or default_cuda_probe)()
    if result.get("available"):
        return {
            "requested": policy,
            "selected": result.get("device", "cuda:0"),
            "fallback_reason": None,
        }

    reason = str(result.get("error", "CUDA not available"))
    if policy == "gpu-only":
        raise RuntimeError(
            f"GPU-only policy failed: {reason}. No fallback to CPU is allowed with gpu-only policy."
        )
    return {"requested": policy, "selected": "cpu", "fallback_reason": reason}
