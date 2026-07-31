"""Runtime environment doctor for embedded and user-selected external runtimes."""

from __future__ import annotations

import json
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from .device_policy import default_cuda_probe, resolve_device

REQUIRED_PYTHON_MINOR = "3.12"
REQUIRED_LEROBOT_VERSION = "0.6.0"
_PROBE_TIMEOUT_SECONDS = 10
_EXTERNAL_PROBE = r'''
import importlib.metadata
import json
import os
import sys


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None

lerobot_version = package_version("lerobot")
torch_version = package_version("torch")
cuda_runtime = None
cuda_available = False
torch_error = None
if torch_version is not None:
    try:
        import torch
        cuda_runtime = getattr(getattr(torch, "version", None), "cuda", None)
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            torch.empty(1, device="cuda:0")
    except Exception as exc:
        torch_error = f"{type(exc).__name__}: {exc}"
        cuda_available = False
print(json.dumps({
    "python": {"path": sys.executable, "version": ".".join(map(str, sys.version_info[:3])), "available": True},
    "lerobot": {"version": lerobot_version, "available": lerobot_version is not None, "import_error": None if lerobot_version is not None else "distribution not installed"},
    "torch": {"version": torch_version, "cuda_runtime": cuda_runtime, "cuda_available": cuda_available, "error": torch_error},
    "secret_seen": any(name in os.environ for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")),
}))
'''


def _safe_probe_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for name in ("HOME", "LANG", "LC_ALL", "PATH"):
        value = os.environ.get(name)
        if value is not None:
            env[name] = value
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    return env


def run_doctor(
    *,
    mode: str = "embedded",
    python_path: str | None = None,
    ffmpeg_path: str | None = None,
    device_policy: str = "auto",
    cuda_probe: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Probe the selected runtime and return compatibility plus diagnostics.

    External probes execute only fixed argument vectors with a sanitized
    environment. ``gpu-only`` failures propagate as errors and are never
    represented as successful fallback results.
    """
    if mode not in {"embedded", "external"}:
        raise ValueError("mode must be 'embedded' or 'external'")

    secret_seen = False
    torch_info: dict[str, Any]
    if mode == "external":
        external = _probe_external_python(python_path)
        python_info = external["python"]
        lerobot_info = external["lerobot"]
        torch_info = external["torch"]
        secret_seen = bool(external.get("secret_seen"))
    else:
        python_info = _probe_python()
        lerobot_info = _probe_lerobot()
        torch_info = _probe_torch()

    ffmpeg_info = _probe_ffmpeg(ffmpeg_path)
    compute = _probe_compute(torch_info)

    effective_cuda_probe = cuda_probe
    if effective_cuda_probe is None and mode == "external":
        effective_cuda_probe = lambda: {
            "available": bool(torch_info.get("cuda_available")),
            "device": "cuda:0",
            "error": torch_info.get("error") or "external runtime CUDA initialization unavailable",
        }
    device_result = resolve_device(
        device_policy,
        cuda_probe=effective_cuda_probe or default_cuda_probe,
    )

    issues = _compatibility_issues(
        python_info=python_info,
        lerobot_info=lerobot_info,
        ffmpeg_info=ffmpeg_info,
        secret_seen=secret_seen,
    )
    return {
        "mode": mode,
        "compatible": not issues,
        "issues": issues,
        "python": python_info,
        "lerobot": lerobot_info,
        "ffmpeg": ffmpeg_info,
        "compute": compute,
        "device_policy_result": device_result,
        "probe_environment_secret_seen": secret_seen,
    }


def _unavailable_python(path: str | None, error: str) -> dict[str, Any]:
    return {
        "python": {"path": path, "version": None, "available": False, "error": error},
        "lerobot": {"version": None, "available": False, "import_error": error},
        "torch": {"version": None, "cuda_runtime": None, "cuda_available": False, "error": error},
        "secret_seen": False,
    }


def _probe_external_python(python_path: str | None) -> dict[str, Any]:
    if not python_path:
        return _unavailable_python(None, "external Python path is required")
    path = Path(python_path)
    if not path.is_file():
        return _unavailable_python(str(path), "external Python executable does not exist")
    if not os.access(path, os.X_OK):
        return _unavailable_python(str(path), "external Python path is not executable")

    try:
        result = subprocess.run(
            [str(path), "-I", "-c", _EXTERNAL_PROBE],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            env=_safe_probe_env(),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _unavailable_python(str(path), f"external Python probe failed: {exc}")
    if result.returncode != 0:
        detail = result.stderr.strip()[:500] or f"exit code {result.returncode}"
        return _unavailable_python(str(path), f"external Python probe failed: {detail}")
    try:
        payload = json.loads(result.stdout.strip())
        if not isinstance(payload, dict):
            raise TypeError("probe result is not an object")
        for key in ("python", "lerobot", "torch"):
            if not isinstance(payload.get(key), dict):
                raise TypeError(f"probe result has no {key} object")
    except (json.JSONDecodeError, TypeError) as exc:
        return _unavailable_python(str(path), f"invalid external Python probe result: {exc}")
    payload["python"]["path"] = str(path)
    return payload


def _probe_python() -> dict[str, Any]:
    version = sys.version_info
    return {
        "path": sys.executable,
        "version": f"{version.major}.{version.minor}.{version.micro}",
        "available": True,
    }


def _probe_lerobot() -> dict[str, Any]:
    try:
        import importlib.metadata

        version = importlib.metadata.version("lerobot")
        return {"version": version, "available": True, "import_error": None}
    except Exception as exc:
        return {"version": None, "available": False, "import_error": str(exc)}


def _probe_torch() -> dict[str, Any]:
    try:
        import importlib.metadata

        version = importlib.metadata.version("torch")
    except Exception as exc:
        return {
            "version": None,
            "cuda_runtime": None,
            "cuda_available": False,
            "error": str(exc),
        }
    try:
        import torch

        available = bool(torch.cuda.is_available())
        if available:
            torch.empty(1, device="cuda:0")
        return {
            "version": version,
            "cuda_runtime": getattr(torch.version, "cuda", None),
            "cuda_available": available,
            "error": None,
        }
    except Exception as exc:
        return {
            "version": version,
            "cuda_runtime": None,
            "cuda_available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _probe_ffmpeg(explicit_path: str | None = None) -> dict[str, Any]:
    path = explicit_path or shutil.which("ffmpeg")
    if not path or not Path(path).is_file() or not os.access(path, os.X_OK):
        return {"path": path, "version": None, "available": False, "codecs": []}
    try:
        result = subprocess.run(
            [path, "-version"],
            capture_output=True,
            text=True,
            timeout=5,
            env=_safe_probe_env(),
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return {"path": path, "version": None, "available": False, "codecs": []}
    if result.returncode != 0:
        return {"path": path, "version": None, "available": False, "codecs": []}
    first_line = result.stdout.splitlines()[0] if result.stdout else ""
    parts = first_line.split()
    version = parts[2].split("-")[0] if len(parts) >= 3 else None
    return {
        "path": path,
        "version": version,
        "available": True,
        "codecs": _probe_ffmpeg_codecs(path),
    }


def _probe_ffmpeg_codecs(ffmpeg_path: str) -> list[str]:
    try:
        result = subprocess.run(
            [ffmpeg_path, "-encoders", "-hide_banner"],
            capture_output=True,
            text=True,
            timeout=10,
            env=_safe_probe_env(),
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    known = {
        "libx264", "libx265", "h264_nvenc", "hevc_nvenc",
        "libvpx", "libvpx-vp9", "libaom-av1",
    }
    return sorted({codec for line in result.stdout.splitlines() for codec in known if codec in line})


def _probe_compute(torch_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "cpu": {
            "model": platform.processor() or platform.machine(),
            "cores": multiprocessing.cpu_count() or 0,
        },
        "cuda": {
            "available": bool(torch_info.get("cuda_available")),
            "version": torch_info.get("cuda_runtime"),
            "driver": _probe_driver(),
            "runtime_error": torch_info.get("error"),
        },
        "gpu": _probe_gpus(),
        "torch_version": torch_info.get("version"),
    }


def _probe_driver() -> str | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            env=_safe_probe_env(),
            check=False,
        )
        return result.stdout.strip().splitlines()[0] if result.returncode == 0 and result.stdout.strip() else None
    except (subprocess.TimeoutExpired, OSError):
        return None


def _probe_gpus() -> list[dict[str, Any]]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            env=_safe_probe_env(),
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if result.returncode != 0:
        return []
    gpus: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            try:
                gpus.append({"name": parts[0], "vram_mb": int(parts[1])})
            except ValueError:
                continue
    return gpus


def _compatibility_issues(
    *,
    python_info: dict[str, Any],
    lerobot_info: dict[str, Any],
    ffmpeg_info: dict[str, Any],
    secret_seen: bool,
) -> list[str]:
    issues: list[str] = []
    python_version = python_info.get("version")
    if not python_info.get("available") or not isinstance(python_version, str) or not (
        python_version == REQUIRED_PYTHON_MINOR or python_version.startswith(f"{REQUIRED_PYTHON_MINOR}.")
    ):
        issues.append(f"Python 3.12 is required; detected {python_version or 'unavailable'}")
    lerobot_version = lerobot_info.get("version")
    if not lerobot_info.get("available") or lerobot_version != REQUIRED_LEROBOT_VERSION:
        issues.append(
            f"lerobot=={REQUIRED_LEROBOT_VERSION} is required; detected {lerobot_version or 'unavailable'}"
        )
    if not ffmpeg_info.get("available"):
        issues.append("FFmpeg executable is required and must pass the version probe")
    if secret_seen:
        issues.append("external runtime probe observed a forbidden secret environment variable")
    return issues
