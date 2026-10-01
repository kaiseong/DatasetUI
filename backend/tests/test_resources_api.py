from dataclasses import replace
from datetime import datetime, timezone
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from datasetui.api import create_router
from datasetui.config import Settings
from datasetui.resource_admission import STATUS_FILE, empty_status


def test_resource_endpoint_exposes_fresh_status_and_fails_closed(database, dispatcher, tmp_path, monkeypatch):
    monkeypatch.setenv('DATASETUI_ADAPTIVE_ENABLED', '1')
    app = FastAPI()
    settings = replace(Settings.from_env(), jobs_root=tmp_path)
    app.include_router(create_router(database, dispatcher, settings=settings))
    with TestClient(app) as client:
        result = client.get('/api/v1/system/resources')
        assert result.status_code == 200
        assert result.json()['healthy'] is False
        assert result.json()['cpu_available_percent'] is None
        status = empty_status(enabled=True)
        status.update(healthy=True, decision='ready', cpu_available_percent=80,
                      memory_available_percent=90, sampled_at=datetime.now(timezone.utc).isoformat())
        (tmp_path/STATUS_FILE).write_text(json.dumps(status))
        assert client.get('/api/v1/system/resources').json()['cpu_available_percent'] == 80
        status['active_jobs'] = 'invalid'
        (tmp_path/STATUS_FILE).write_text(json.dumps(status))
        assert client.get('/api/v1/system/resources').json()['healthy'] is False
