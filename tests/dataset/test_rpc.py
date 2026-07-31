"""Tests for dataset.open / dataset.browse / dataset.validate RPC methods."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from lerobot_dataset_editor.rpc.protocol import SERVER_NOT_INITIALIZED
from lerobot_dataset_editor.rpc.server import RpcDispatchError, RpcServer


@pytest.fixture
def server(tmp_path: Path) -> RpcServer:
    """Initialized RPC server with XDG isolation."""
    with patch.dict(os.environ, {
        "XDG_DATA_HOME": str(tmp_path / "xdg_data"),
        "XDG_CACHE_HOME": str(tmp_path / "xdg_cache"),
    }):
        s = RpcServer()
        from lerobot_dataset_editor.rpc.protocol import RpcRequest
        s.dispatch(RpcRequest(
            id=1,
            method="initialize",
            params={"protocolVersion": 1, "client": {"name": "test", "version": "1"}},
        ))
        return s


@pytest.fixture
def registered_v30_project(server: RpcServer, v30_fixture: Path) -> str:
    """Register a project pointing at v30_valid fixture, return project_id."""
    from lerobot_dataset_editor.rpc.protocol import RpcRequest

    result = server.dispatch(RpcRequest(
        id=10,
        method="project.register",
        params={"name": "TestV30", "source_path": str(v30_fixture), "target_format": "v3", "runtime_mode": "embedded"},
    ))
    return result["id"]


class TestDatasetRpc:
    """Tests for dataset.* RPC methods."""

    def test_dataset_open_returns_document(self, server: RpcServer, registered_v30_project: str) -> None:
        from lerobot_dataset_editor.rpc.protocol import RpcRequest

        result = server.dispatch(RpcRequest(
            id=20,
            method="dataset.open",
            params={"project_id": registered_v30_project},
        ))
        assert result["version"]["version"] == "v3.0"
        assert result["total_frames"] == 30
        assert result["fps"] == 10
        assert len(result["episodes"]) == 3

    def test_dataset_open_fails_on_corrupt(self, server: RpcServer, corrupt_fixture: Path) -> None:
        from lerobot_dataset_editor.rpc.protocol import RpcRequest

        reg = server.dispatch(RpcRequest(
            id=30,
            method="project.register",
            params={"name": "Corrupt", "source_path": str(corrupt_fixture), "target_format": "v3", "runtime_mode": "embedded"},
        ))
        with pytest.raises(RpcDispatchError) as exc_info:
            server.dispatch(RpcRequest(
                id=31,
                method="dataset.open",
                params={"project_id": reg["id"]},
            ))
        assert exc_info.value.code != 0

    def test_dataset_browse_returns_episodes(self, server: RpcServer, registered_v30_project: str) -> None:
        from lerobot_dataset_editor.rpc.protocol import RpcRequest

        result = server.dispatch(RpcRequest(
            id=40,
            method="dataset.browse",
            params={"project_id": registered_v30_project, "offset": 0, "limit": 2},
        ))
        assert result["total"] == 3
        assert len(result["episodes"]) == 2
        assert result["episodes"][0]["index"] == 0
        assert result["episodes"][0]["frame_start"] == 0
        assert result["episodes"][0]["frame_end"] == 10

    def test_dataset_validate_returns_result(self, server: RpcServer, registered_v30_project: str) -> None:
        from lerobot_dataset_editor.rpc.protocol import RpcRequest

        result = server.dispatch(RpcRequest(
            id=50,
            method="dataset.validate",
            params={"project_id": registered_v30_project},
        ))
        assert result["valid"] is True
        assert result["errors"] == []

    def test_dataset_open_before_initialize_fails(self, tmp_path: Path) -> None:
        from lerobot_dataset_editor.rpc.protocol import RpcRequest

        with patch.dict(os.environ, {"XDG_DATA_HOME": str(tmp_path / "xdg_data")}):
            s = RpcServer()
            with pytest.raises(RpcDispatchError) as exc_info:
                s.dispatch(RpcRequest(
                    id=60,
                    method="dataset.open",
                    params={"project_id": "abc"},
                ))
            assert exc_info.value.code == SERVER_NOT_INITIALIZED

    def test_dataset_methods_in_method_list(self, server: RpcServer) -> None:
        from lerobot_dataset_editor.rpc.protocol import RpcRequest

        result = server.dispatch(RpcRequest(
            id=70,
            method="system.ping",
            params=None,
        ))
        # Methods list is available from initialize response
        from lerobot_dataset_editor.rpc.server import METHODS
        assert "dataset.open" in METHODS
        assert "dataset.browse" in METHODS
        assert "dataset.validate" in METHODS
