"""Values every route module needs from the application factory."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from datasetui.config import Settings
from datasetui.database import Database
from datasetui.huggingface import HuggingFaceGateway
from datasetui.queueing import QueueDispatcher


@dataclass(frozen=True)
class RouterContext:
    database: Database
    dispatcher: QueueDispatcher
    settings: Settings
    hf_gateway: HuggingFaceGateway
    dispatch_job: Callable[[dict[str, Any]], dict[str, Any]]
    dispatch_library_job: Callable[[dict[str, Any]], dict[str, Any]]
    recover_expired_jobs: Callable[[], None]
    dataset_job_payload: Callable[[str], dict[str, str]]
