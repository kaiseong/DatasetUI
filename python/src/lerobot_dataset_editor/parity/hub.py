"""Minimal Hugging Face datasets client with transient credentials only."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

from .common import finite

DEFAULT_ENDPOINT = "https://huggingface.co"


def _endpoint(value: str | None) -> str:
    endpoint = (value or DEFAULT_ENDPOINT).rstrip("/")
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("endpoint must be an http(s) origin")
    hostname = (parsed.hostname or "").lower()
    if endpoint != DEFAULT_ENDPOINT and hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("endpoint override is limited to loopback test servers")
    if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise ValueError("endpoint must be an origin without credentials or path")
    return endpoint


def _request(url: str, token: str | None, *, timeout: float = 30) -> tuple[Any, dict[str, str]]:
    headers = {"Accept": "application/json", "User-Agent": "DatasetUI/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response), dict(response.headers.items())
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        # Never include request headers or a potentially credential-bearing URL.
        raise ValueError(f"Hub request failed ({type(exc).__name__})") from None


def hub_search(query: str, token: str | None = None, endpoint: str | None = None) -> dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    url = _endpoint(endpoint) + "/api/quicksearch?" + urllib.parse.urlencode({"q": query, "type": "dataset"})
    payload, _ = _request(url, token)
    items = payload if isinstance(payload, list) else payload.get("datasets", payload.get("items", []))
    results = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict):
            repo_id = item.get("id") or item.get("repo_id") or item.get("name")
            if isinstance(repo_id, str):
                results.append({"id": repo_id, "repo_id": repo_id,
                                "private": bool(item.get("private", False)),
                                "downloads": item.get("downloads"), "likes": item.get("likes"),
                                "updated_at": item.get("lastModified") or item.get("updated_at")})
    return finite({"results": results, "total": len(results)})


def hub_info(repo_id: str, token: str | None = None, endpoint: str | None = None,
             revision: str | None = None) -> dict[str, Any]:
    if not isinstance(repo_id, str) or "/" not in repo_id or repo_id.startswith("/"):
        raise ValueError("repo_id must be in owner/name form")
    encoded = "/".join(urllib.parse.quote(part, safe="") for part in repo_id.split("/"))
    suffix = "?" + urllib.parse.urlencode({"revision": revision}) if revision else ""
    payload, _ = _request(f"{_endpoint(endpoint)}/api/datasets/{encoded}{suffix}", token)
    if not isinstance(payload, dict):
        raise ValueError("Hub returned invalid dataset metadata")
    siblings = payload.get("siblings", [])
    return finite({"repo_id": payload.get("id", repo_id), "sha": payload.get("sha"),
                   "private": bool(payload.get("private", False)),
                   "gated": payload.get("gated", False),
                   "files": [item.get("rfilename") for item in siblings
                             if isinstance(item, dict) and isinstance(item.get("rfilename"), str)]})


def _safe_repo_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("Hub tree contains unsafe path")
    return path


def hub_import(repo_id: str, destination: Path, token: str | None = None,
               endpoint: str | None = None, revision: str | None = None) -> dict[str, Any]:
    if destination.exists():
        raise ValueError("destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    info = hub_info(repo_id, token, endpoint, revision)
    files = [_safe_repo_path(value) for value in info["files"]]
    base = _endpoint(endpoint)
    rev = revision or info.get("sha") or "main"
    encoded_repo = "/".join(urllib.parse.quote(part, safe="") for part in repo_id.split("/"))
    temp = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for relative in files:
            target = temp.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            url = (f"{base}/datasets/{encoded_repo}/resolve/{urllib.parse.quote(str(rev), safe='')}/"
                   + "/".join(urllib.parse.quote(part, safe="") for part in relative.parts))
            headers = {"User-Agent": "DatasetUI/0.1"}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(url, headers=headers), timeout=30
                ) as response, target.open("wb") as output:
                    shutil.copyfileobj(response, output)
            except (OSError, urllib.error.URLError):
                raise ValueError("Hub download failed") from None
        os.replace(temp, destination)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return {"destination": str(destination.resolve()), "repo_id": repo_id,
            "revision": rev, "files": len(files)}
