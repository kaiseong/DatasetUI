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
    cache_root: Path
    staging_root: Path
    jobs_root: Path
    hf_read_token: str | None = None
    hf_import_max_bytes: int = 2_000_000_000_000
    dataset_scan_max_depth: int = 6
    job_timeout_seconds: int = 900
    io_job_timeout_seconds: int = 86_400
    job_lease_seconds: int = 120
    job_heartbeat_seconds: int = 20

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
            cache_root=Path(os.environ.get("DATASETUI_CACHE_ROOT", "/data/cache")),
            staging_root=Path(
                os.environ.get("DATASETUI_STAGING_ROOT", "/data/staging")
            ),
            jobs_root=Path(os.environ.get("DATASETUI_JOBS_ROOT", "/data/jobs")),
            hf_read_token=os.environ.get("HF_READ_TOKEN") or None,
            hf_import_max_bytes=int(
                os.environ.get("DATASETUI_HF_IMPORT_MAX_BYTES", "2000000000000")
            ),
            dataset_scan_max_depth=int(os.environ.get("DATASETUI_SCAN_MAX_DEPTH", "6")),
            job_timeout_seconds=int(
                os.environ.get("DATASETUI_JOB_TIMEOUT_SECONDS", "900")
            ),
            io_job_timeout_seconds=int(
                os.environ.get("DATASETUI_IO_JOB_TIMEOUT_SECONDS", "86400")
            ),
            job_lease_seconds=int(os.environ.get("DATASETUI_JOB_LEASE_SECONDS", "120")),
            job_heartbeat_seconds=int(
                os.environ.get("DATASETUI_JOB_HEARTBEAT_SECONDS", "20")
            ),
        )
