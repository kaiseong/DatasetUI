from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from lerobot_dataset_editor.registry.repository import register_project
from lerobot_dataset_editor.rpc.protocol import RpcRequest
from lerobot_dataset_editor.rpc.server import RpcServer


def _server(dataset: Path) -> tuple[RpcServer, str]:
    server = RpcServer()
    server.dispatch(RpcRequest(id=1, method="initialize", params={
        "protocolVersion": 1, "client": {"name": "pytest"},
    }))
    registered = register_project(server._get_registry(), name="Parity", source_path=str(dataset),
                                  target_format="v3", runtime_mode="embedded")
    return server, registered["id"]


def test_rpc_episode_and_analytics_match_renderer_contract(
    v30_fixture: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    dataset = tmp_path / "dataset"; shutil.copytree(v30_fixture, dataset)
    server, project_id = _server(dataset)
    episode = server.dispatch(RpcRequest(id=2, method="dataset.episode",
                                         params={"project_id": project_id, "episode_index": 0}))
    assert set(episode) >= {"episode_index", "episode_count", "duration", "fps", "timestamps",
                            "media", "action", "state", "progress", "robot"}
    assert all(set(series) == {"name", "values"} for series in episode["action"] + episode["state"])

    analytics = server.dispatch(RpcRequest(id=3, method="dataset.analytics",
                                           params={"project_id": project_id}))
    assert set(analytics) >= {"statistics", "histograms", "episodes", "variance", "autocorrelation",
                              "suggested_chunk_size", "velocity", "jerk", "speed_cv", "alignment"}
    assert len(analytics["variance"]) == 50
    assert analytics["doctor"]["status"] == "ok"
    assert {check["name"] for check in analytics["doctor"]["checks"]} >= {
        "Dataset structure", "Episodes", "Action data", "Media", "Progress sidecar",
    }


def test_media_resolve_rejects_absolute_parent_and_symlink_escape(
    v30_fixture: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    dataset = tmp_path / "dataset"; shutil.copytree(v30_fixture, dataset)
    media = dataset / "preview.png"; media.write_bytes(b"png")
    outside = tmp_path / "outside.png"; outside.write_bytes(b"png")
    (dataset / "escape.png").symlink_to(outside)
    server, project_id = _server(dataset)
    resolved = server.dispatch(RpcRequest(id=2, method="media.resolve",
                                          params={"project_id": project_id, "relative_path": "preview.png"}))
    assert resolved["mime_type"] == "image/png"
    for unsafe in (str(outside), "../outside.png", "escape.png"):
        try:
            server.dispatch(RpcRequest(id=3, method="media.resolve",
                                       params={"project_id": project_id, "relative_path": unsafe}))
        except Exception as exc:
            assert "outside" in str(exc) or "relative" in str(exc)
        else:
            raise AssertionError(f"unsafe media path accepted: {unsafe}")


@pytest.mark.parametrize("method,params", [
    ("hub.search", {"query": "robots", "token": "hf_secret", "endpoint": "http://127.0.0.1:9999"}),
    ("hub.info", {"repo_id": "org/data", "token": "hf_secret", "endpoint": "http://127.0.0.1:9999"}),
    ("hub.import", {"repo_id": "org/data", "destination": "/tmp/data", "token": "hf_secret",
                    "endpoint": "http://127.0.0.1:9999"}),
])
def test_public_rpc_never_forwards_renderer_endpoint_overrides(method: str, params: dict) -> None:
    server = RpcServer()
    server.dispatch(RpcRequest(id=1, method="initialize", params={
        "protocolVersion": 1, "client": {"name": "pytest"},
    }))
    # If the untrusted renderer endpoint reached the helper, connection errors
    # would mention loopback. Production dispatch pins requests to HF instead;
    # mock the helpers at their source in dedicated unit tests rather than
    # exposing a credential-bearing SSRF capability in the bridge.
    import lerobot_dataset_editor.parity.hub as hub
    seen: list[tuple] = []
    original_search, original_info, original_import = hub.hub_search, hub.hub_info, hub.hub_import
    try:
        hub.hub_search = lambda *args, **kwargs: seen.append((args, kwargs)) or {"results": []}
        hub.hub_info = lambda *args, **kwargs: seen.append((args, kwargs)) or {"files": []}
        hub.hub_import = lambda *args, **kwargs: seen.append((args, kwargs)) or {"destination": "/tmp/data"}
        server.dispatch(RpcRequest(id=2, method=method, params=params))
    finally:
        hub.hub_search, hub.hub_info, hub.hub_import = original_search, original_info, original_import
    assert seen and "endpoint" not in seen[0][1]
    assert "http://127.0.0.1:9999" not in seen[0][0]
