from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasetui.api import create_router  # noqa: E402
from datasetui.database import Database  # noqa: E402
from datasetui.queueing import RecordingDispatcher  # noqa: E402


@pytest.fixture
def database(tmp_path: Path) -> Database:
    db = Database(tmp_path / "datasetui.sqlite3")
    db.initialize()
    return db


@pytest.fixture
def dispatcher() -> RecordingDispatcher:
    return RecordingDispatcher()


@pytest.fixture
def client(database: Database, dispatcher: RecordingDispatcher) -> TestClient:
    app = FastAPI()
    app.include_router(create_router(database, dispatcher))
    return TestClient(app)
