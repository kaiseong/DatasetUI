from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from datasetui.application import create_application
from datasetui.database import Database
from datasetui.queueing import RecordingDispatcher


def test_legacy_wildcard_cors_does_not_reach_workbench(
    database: Database, dispatcher: RecordingDispatcher
) -> None:
    legacy_app = FastAPI()
    legacy_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app = create_application(
        database=database,
        dispatcher=dispatcher,
        allowed_origins=("https://192.168.0.3",),
        legacy_app=legacy_app,
    )
    client = TestClient(app)

    response = client.options(
        "/api/v1/profiles",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code in {400, 403}
    assert "access-control-allow-origin" not in response.headers

    mutation = client.post(
        "/api/v1/profiles",
        headers={"Origin": "https://evil.example"},
        json={"name": "Blocked"},
    )
    assert mutation.status_code == 403
    assert database.list_profiles() == []


def test_configured_datasetui_origin_is_allowed(
    database: Database, dispatcher: RecordingDispatcher
) -> None:
    app = create_application(
        database=database,
        dispatcher=dispatcher,
        allowed_origins=("https://192.168.0.3",),
        legacy_app=FastAPI(),
    )
    response = TestClient(app).post(
        "/api/v1/profiles",
        headers={"Origin": "https://192.168.0.3"},
        json={"name": "Researcher"},
    )
    assert response.status_code == 201
    assert response.headers["access-control-allow-origin"] == "https://192.168.0.3"
