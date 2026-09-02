from __future__ import annotations

from fastapi.testclient import TestClient


def test_profile_lifecycle(client: TestClient) -> None:
    created = client.post("/api/v1/profiles", json={"name": "  Kim   GS  "})
    assert created.status_code == 201
    profile = created.json()
    assert profile["name"] == "Kim GS"

    assert client.get("/api/v1/profiles").json() == [profile]

    archived = client.patch(
        f"/api/v1/profiles/{profile['id']}", json={"archived": True}
    )
    assert archived.status_code == 200
    assert archived.json()["archived_at"] is not None
    assert client.get("/api/v1/profiles").json() == []
    assert len(client.get("/api/v1/profiles?include_archived=true").json()) == 1


def test_profile_names_are_case_insensitively_unique(client: TestClient) -> None:
    assert (
        client.post("/api/v1/profiles", json={"name": "Researcher"}).status_code == 201
    )
    duplicate = client.post("/api/v1/profiles", json={"name": "researcher"})
    assert duplicate.status_code == 409


def test_profile_input_is_strict(client: TestClient) -> None:
    assert client.post("/api/v1/profiles", json={"name": ""}).status_code == 422
    assert (
        client.post(
            "/api/v1/profiles", json={"name": "OK", "password": "no"}
        ).status_code
        == 422
    )
