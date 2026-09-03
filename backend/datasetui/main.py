from __future__ import annotations

from app import app as legacy_app

from datasetui.application import create_application
from datasetui.config import Settings
from datasetui.database import Database
from datasetui.queueing import RQDispatcher


settings = Settings.from_env()
database = Database(settings.database_path)
database.initialize()
dispatcher = RQDispatcher(
    redis_url=settings.redis_url,
    timeout_seconds=settings.job_timeout_seconds,
    io_timeout_seconds=settings.io_job_timeout_seconds,
)
app = create_application(
    database=database,
    dispatcher=dispatcher,
    allowed_origins=settings.allowed_origins,
    legacy_app=legacy_app,
    settings=settings,
)
