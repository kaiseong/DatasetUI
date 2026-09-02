from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    database_path: Path
    redis_url: str
    allowed_origins: tuple[str, ...]
    nas_root: Path
    dataset_scan_max_depth: int = 6
    job_timeout_seconds: int = 900

    @classmethod
    def from_env(cls) -> "Settings":
        host = os.environ.get("DATASETUI_HOST", "192.168.0.3")
        configured_origins = os.environ.get("DATASETUI_ALLOWED_ORIGINS")
        allowed_origins = tuple(
            origin.strip()
            for origin in (configured_origins or f"https://{host}").split(",")
            if origin.strip()
        )
        return cls(
            database_path=Path(
                os.environ.get(
                    "DATASETUI_DB_PATH",
                    "/data/registry/datasetui.sqlite3",
                )
            ),
            redis_url=os.environ.get("REDIS_URL", "redis://redis:6379/0"),
            allowed_origins=allowed_origins,
            nas_root=Path(
                os.environ.get(
                    "DATASETUI_NAS_ROOT",
                    "/mnt/datasetui-nas/DatasetUI",
                )
            ),
            dataset_scan_max_depth=int(os.environ.get("DATASETUI_SCAN_MAX_DEPTH", "6")),
            job_timeout_seconds=int(
                os.environ.get("DATASETUI_JOB_TIMEOUT_SECONDS", "900")
            ),
        )
