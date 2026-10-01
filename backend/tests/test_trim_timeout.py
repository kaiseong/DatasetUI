import pytest

from test_curation_api import _profile, _register


@pytest.mark.parametrize("trim_enabled,relative_enabled,expected", [
    (True, False, 86400), (True, True, 86400), (False, False, None),
])
def test_trim_dispatch_timeout(client, database, monkeypatch, trim_enabled,
                               relative_enabled, expected):
    from datasetui.queueing import RecordingDispatcher
    seen = []
    original = RecordingDispatcher.enqueue

    def capture(self, **kwargs):
        seen.append(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(RecordingDispatcher, "enqueue", capture)
    dataset = _register(database)
    profile = _profile(client, "Trim timeout")
    recipe = client.post(f"/api/v1/datasets/{dataset['id']}/recipes", json={
        "profile_id": profile["id"], "name": "Trim", "selection_mode": "all",
        "trim_config": {"enabled": trim_enabled},
        "relative_action": {"enabled": relative_enabled,
                            "dimensions": ["joint_1"] if relative_enabled else []},
    })
    assert recipe.status_code == 201
    response = client.post(f"/api/v1/recipes/{recipe.json()['id']}/runs", json={
        "profile_id": profile["id"], "output_name": "trim-timeout",
        "idempotency_key": "trim-timeout",
    })
    assert response.status_code == 202
    assert seen[-1].get("job_timeout") == expected


def test_trim_timeout_environment(monkeypatch):
    from datasetui.config import Settings
    monkeypatch.setenv("DATASETUI_TRIM_TIMEOUT_SECONDS", "7200")
    assert Settings.from_env().trim_timeout_seconds == 7200
