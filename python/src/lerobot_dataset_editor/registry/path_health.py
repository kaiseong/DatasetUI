"""Path health detection for registered project directories."""

from __future__ import annotations

import os
from typing import Any


def check_path(path: str) -> dict[str, Any]:
    """Check whether a registered path is healthy.

    Returns a dict with:
      - status: "ok" | "missing" | "read_only"
      - path: the checked path
    """
    if not os.path.exists(path):
        return {"status": "missing", "path": path}

    if not os.access(path, os.W_OK):
        return {"status": "read_only", "path": path}

    return {"status": "ok", "path": path}
