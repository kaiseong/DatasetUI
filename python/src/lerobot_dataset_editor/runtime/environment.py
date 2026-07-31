"""Runtime environment resolution: embedded vs external Python/FFmpeg paths."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any


def resolve_runtime_paths(
    *,
    mode: str,
    app_root: str,
    python_path: str | None = None,
    ffmpeg_path: str | None = None,
) -> dict[str, Any]:
    """Resolve runtime executable paths based on mode.

    Args:
        mode: "embedded" or "external"
        app_root: Application root directory (for embedded mode venv lookup)
        python_path: Explicit Python path (external mode)
        ffmpeg_path: Explicit FFmpeg path (external mode)

    Returns:
        Dict with resolved "python_path" and "ffmpeg_path".

    Raises:
        FileNotFoundError: If a specified external path does not exist.
        ValueError: If mode is invalid or embedded venv not found.
    """
    if mode == "embedded":
        return _resolve_embedded(app_root)
    elif mode == "external":
        return _resolve_external(python_path=python_path, ffmpeg_path=ffmpeg_path)
    else:
        raise ValueError(f"Invalid runtime mode: {mode!r}. Must be 'embedded' or 'external'.")


def _resolve_embedded(app_root: str) -> dict[str, Any]:
    """Resolve embedded (venv) Python and system FFmpeg."""
    venv_python = Path(app_root) / ".venv" / "bin" / "python"
    if not venv_python.exists():
        raise ValueError(
            f"Embedded Python not found at {venv_python}. "
            "Ensure the virtual environment is set up."
        )

    ffmpeg = shutil.which("ffmpeg")
    return {
        "python_path": str(venv_python),
        "ffmpeg_path": ffmpeg,
    }


def _resolve_external(
    *,
    python_path: str | None = None,
    ffmpeg_path: str | None = None,
) -> dict[str, Any]:
    """Resolve external user-specified executable paths."""
    resolved_python = None
    resolved_ffmpeg = None

    if python_path:
        if not os.path.isfile(python_path):
            raise FileNotFoundError(
                f"External Python not found: {python_path}"
            )
        if not os.access(python_path, os.X_OK):
            raise ValueError(
                f"External Python is not executable: {python_path}"
            )
        resolved_python = python_path

    if ffmpeg_path:
        if not os.path.isfile(ffmpeg_path):
            raise FileNotFoundError(
                f"External FFmpeg not found: {ffmpeg_path}"
            )
        if not os.access(ffmpeg_path, os.X_OK):
            raise ValueError(
                f"External FFmpeg is not executable: {ffmpeg_path}"
            )
        resolved_ffmpeg = ffmpeg_path

    return {
        "python_path": resolved_python,
        "ffmpeg_path": resolved_ffmpeg,
    }
