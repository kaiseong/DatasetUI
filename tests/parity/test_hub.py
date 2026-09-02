from __future__ import annotations

import json
import io
import urllib.request
from pathlib import Path

from lerobot_dataset_editor.parity.hub import hub_import, hub_info, hub_search


class Response(io.BytesIO):
    def __init__(self, body: bytes, content_type: str = "application/json"):
        super().__init__(body); self.headers = {"Content-Type": content_type}
    def __enter__(self): return self
    def __exit__(self, *_args): self.close()


class FakeUrlOpen:
    seen_authorization = []
    def __call__(self, request, timeout=30):
        type(self).seen_authorization.append(request.get_header("Authorization"))
        path = request.full_url
        if "/api/quicksearch" in path:
            payload = [{"id": "org/data", "private": True, "downloads": 2}]
        elif "/api/datasets/org/data" in path:
            payload = {"id": "org/data", "sha": "abc", "private": True,
                       "siblings": [{"rfilename": "meta/info.json"}]}
        elif "/datasets/org/data/resolve/" in path:
            return Response(b'{"codebase_version":"v3.0"}')
        else:
            raise AssertionError(path)
        return Response(json.dumps(payload).encode())


def test_hub_search_info_import_use_transient_bearer(tmp_path: Path, monkeypatch) -> None:
    opener = FakeUrlOpen(); monkeypatch.setattr(urllib.request, "urlopen", opener)
    endpoint = "http://127.0.0.1:9999"
    token = "hf_" + "x" * 32
    assert hub_search("robot", token, endpoint)["results"][0]["repo_id"] == "org/data"
    assert hub_info("org/data", token, endpoint)["files"] == ["meta/info.json"]
    destination = tmp_path / "download"
    assert hub_import("org/data", destination, token, endpoint)["files"] == 1
    assert (destination / "meta" / "info.json").is_file()
    assert all(value == f"Bearer {token}" for value in FakeUrlOpen.seen_authorization)
    assert token not in repr(hub_info("org/data", token, endpoint))


def test_hub_rejects_non_loopback_endpoint_override() -> None:
    try:
        hub_search("robot", endpoint="https://example.com")
    except ValueError as exc:
        assert "loopback" in str(exc)
    else:
        raise AssertionError("SSRF-capable endpoint override accepted")
